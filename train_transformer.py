"""
# Train the Set Transformer model
python train_transformer.py --input ticl_dataset.h5 --batch-size 32 --epochs 50

# Adjust hyperparameters
python train_transformer.py --input ticl_dataset.h5 --hidden-dim 128 --lr 0.0005 --heads 8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.metrics import classification_report
import argparse
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# 1. LABEL MAPPING

class_labels = {
    22: 0, 111: 0,   # photon, pion0
    11: 1, -11: 1,   # electron/positron
    13: 2, -13: 2,   # muon
    1011: 3,
    211: 4, -211: 4, 321: 4, -321: 4,  # charged hadron
    310: 5, 130: 5,  # neutral hadron
    -1: 6,           # unknown
    0: 7             # noise/empty
}

num_classes = 8


# 2. OPTIMIZED DATASET FOR COLUMNAR HDF5

class TracksterDataset(Dataset):
    
    def __init__(self, h5_file, split='train', train_ratio=0.7, val_ratio=0.15, 
                 transform=None, seed=42, cache_metadata=True):
        
        self.h5_file = h5_file
        self.transform = transform
        self.split = split
        self.cache_metadata = cache_metadata
        
        # Open file once to get metadata
        with h5py.File(h5_file, 'r') as f:
            self.num_tracksters = f.attrs['num_tracksters']
            
            # Pre-filter valid tracksters (those with clusters)
            if 'num_clusters' in f:
                num_clusters = f['num_clusters'][:]
                self.valid_mask = num_clusters > 0
                self.valid_indices = np.where(self.valid_mask)[0]
            else:
                # Fallback if no num_clusters dataset
                self.valid_indices = np.arange(self.num_tracksters)
            
            print(f"Total tracksters: {self.num_tracksters:,}, Valid: {len(self.valid_indices):,}")
        
        # Cache cluster lengths for faster batching 
        if cache_metadata and 'num_clusters' in locals():
            self.cluster_lengths = num_clusters[self.valid_indices]
        
        # Create reproducible indices for splits
        rng = np.random.RandomState(seed)
        shuffled_indices = self.valid_indices.copy()
        rng.shuffle(shuffled_indices)
        
        n_valid = len(self.valid_indices)
        n_train = int(train_ratio * n_valid)
        n_val = int(val_ratio * n_valid)
        
        if split == 'train':
            self.indices = shuffled_indices[:n_train]
        elif split == 'val':
            self.indices = shuffled_indices[n_train:n_train + n_val]
        else:  # test
            self.indices = shuffled_indices[n_train + n_val:]
        
        print(f"Created {split} dataset with {len(self.indices):,} tracksters")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        with h5py.File(self.h5_file, 'r') as f:
            # Load trackster features [eta, phi, energy]
            trackster = torch.tensor(f['features'][actual_idx], dtype=torch.float32)
            
            # Load labels
            pid = torch.tensor(class_labels.get(int(f['true_pid'][actual_idx]), 6), 
                              dtype=torch.long)
            
            clusters_flat = f['clusters'][actual_idx]  # shape: [N*4]
            
            if 'num_clusters' in f:
                n_clusters = int(f['num_clusters'][actual_idx])
            else:
                n_clusters = len(clusters_flat) // 4
            
            if n_clusters > 0:
                clusters = clusters_flat.reshape(-1, 4)  # [N, 4]
            else:
                clusters = np.zeros((0, 4), dtype=np.float32)
            
            clusters = torch.tensor(clusters, dtype=torch.float32)
        
        # Apply transforms if any
        if self.transform:
            clusters = self.transform(clusters)
        
        return clusters, pid, trackster


def collate_sets(batch):
    """Optimized collate for variable-length clusters"""
    cluster_sets = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    tracksters = torch.stack([item[2] for item in batch])
    return cluster_sets, labels, tracksters


# 3. SET TRANSFORMER MODEL

class SetTransformer(nn.Module):
    def __init__(self, 
                 node_features=4,
                 hidden_dim=128,
                 num_heads=8,
                 num_layers=3,
                 num_classes=8,
                 dropout=0.1,
                 use_trackster_features=True):
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_trackster_features = use_trackster_features
        
        # Encode each cluster independently
        self.encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # trackster feature encoder
        if use_trackster_features:
            self.trackster_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),  # [eta, phi, energy]
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            )
            pool_input_dim = hidden_dim * 2 + hidden_dim // 2
        else:
            pool_input_dim = hidden_dim * 2
        
        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
            self.ff_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),  # GELU often works better than ReLU
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ))
        
        # Pooling and classification
        self.classifier = nn.Sequential(
            nn.LayerNorm(pool_input_dim),
            nn.Linear(pool_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, cluster_sets, trackster_features=None):
        batch_size = len(cluster_sets)
        
        # Find max clusters in this batch for padding
        max_clusters = max(c.size(0) for c in cluster_sets)
        device = cluster_sets[0].device
        
        # Pad all sets to same size for batch processing
        padded_clusters = []
        attention_masks = []
        
        for clusters in cluster_sets:
            n = clusters.size(0)
            if n < max_clusters:
                padding = torch.zeros(max_clusters - n, clusters.size(1), device=device)
                padded = torch.cat([clusters, padding], dim=0)
                mask = torch.cat([torch.ones(n, device=device), 
                                 torch.zeros(max_clusters - n, device=device)])
            else:
                padded = clusters
                mask = torch.ones(max_clusters, device=device)
            
            padded_clusters.append(padded)
            attention_masks.append(mask)
        
        # Stack into batch tensor
        x = torch.stack(padded_clusters, dim=0)  # [batch_size, max_clusters, 4]
        mask = torch.stack(attention_masks, dim=0)  # [batch_size, max_clusters]
        
        # Encode each cluster
        x = self.encoder(x)  # [batch_size, max_clusters, hidden_dim]
        
        # Self-attention layers
        for attn, norm, ff in zip(self.attention_layers, self.norm_layers, self.ff_layers):
            # Self-attention with masking
            attn_out, _ = attn(x, x, x, key_padding_mask=~mask.bool())
            x = norm(x + attn_out)
            ff_out = ff(x)
            x = x + ff_out
        
        # Global pooling (only over real clusters)
        expanded_mask = mask.unsqueeze(-1).float()  # [batch_size, max_clusters, 1]
        
        # Mean pool
        x_masked = x * expanded_mask
        x_mean = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Max pool (with -inf for padded positions)
        x_max = (x_masked + (1 - expanded_mask) * -1e9).max(dim=1)[0]
        
        # Concatenate pooled features
        x_pooled = torch.cat([x_mean, x_max], dim=1)  # [batch_size, hidden_dim*2]
        
        # Add trackster features if available
        if self.use_trackster_features and trackster_features is not None:
            trackster_encoded = self.trackster_encoder(trackster_features)
            x_pooled = torch.cat([x_pooled, trackster_encoded], dim=1)
        
        # Classification
        out = self.classifier(x_pooled)
        
        return out


# 4. HIERARCHICAL LOSS
class HierarchicalLoss(nn.Module):
    def __init__(self, 
                 em_classes=[0, 1],
                 had_classes=[4, 5],
                 other_classes=[2, 3, 6, 7],
                 within_group_penalty=0.5,
                 cross_group_penalty=5.0,
                 other_penalty=2.0):
        super().__init__()
        
        self.register_buffer('em_classes', torch.tensor(em_classes))
        self.register_buffer('had_classes', torch.tensor(had_classes))
        self.register_buffer('other_classes', torch.tensor(other_classes))
        
        self.within_group_penalty = within_group_penalty
        self.cross_group_penalty = cross_group_penalty
        self.other_penalty = other_penalty
        
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def _get_group(self, class_ids):
        groups = torch.full_like(class_ids, 2, device=class_ids.device)
        
        # EM group (0)
        for c in self.em_classes:
            groups[class_ids == c] = 0
        
        # HAD group (1)
        for c in self.had_classes:
            groups[class_ids == c] = 1
        
        return groups
    
    def forward(self, pred, target):
        base_loss = self.ce(pred, target)
        pred_class = pred.argmax(dim=1)
        
        true_groups = self._get_group(target)
        pred_groups = self._get_group(pred_class)
        
        # Vectorized penalty calculation 
        same_group = (true_groups == pred_groups)
        em_had_cross = ((true_groups == 0) & (pred_groups == 1)) | ((true_groups == 1) & (pred_groups == 0))
        other_involved = ~(same_group | em_had_cross)
        
        penalties = torch.ones_like(base_loss)
        penalties[same_group] = self.within_group_penalty
        penalties[em_had_cross] = self.cross_group_penalty
        penalties[other_involved] = self.other_penalty
        
        return (base_loss * penalties).mean()
    
    def get_group_accuracy(self, pred, target):
        pred_class = pred.argmax(dim=1)
        true_groups = self._get_group(target)
        pred_groups = self._get_group(pred_class)
        return (true_groups == pred_groups).float().mean().item()
    
    def get_em_vs_had_confusion(self, pred, target):
        if len(pred) == 0:
            return np.array([[0, 0], [0, 0]])
        
        pred_class = pred.argmax(dim=1) if pred.dim() > 1 else pred
        true_groups = self._get_group(target)
        pred_groups = self._get_group(pred_class)
        
        em_had_mask = (true_groups != 2)
        if em_had_mask.sum() == 0:
            return np.array([[0, 0], [0, 0]])
        
        true_em_had = true_groups[em_had_mask].cpu().numpy()
        pred_em_had = pred_groups[em_had_mask].cpu().numpy()
        
        return confusion_matrix(true_em_had, pred_em_had, labels=[0, 1])


# 5. TRAINING FUNCTIONS

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    group_correct = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for cluster_sets, labels, tracksters in pbar:
        if len(cluster_sets) == 0:
            continue
            
        labels = labels.to(device)
        tracksters = tracksters.to(device)
        cluster_sets = [c.to(device) for c in cluster_sets]
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        out = model(cluster_sets, tracksters)
        
        # Calculate loss
        loss = criterion(out, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        group_correct += criterion.get_group_accuracy(out, labels) * labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{(pred == labels).float().mean().item():.3f}'
        })
    
    class_acc = correct / max(total, 1)
    group_acc = group_correct / max(total, 1)
    
    return total_loss / len(loader), class_acc, group_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    group_correct = 0
    all_preds = []
    all_labels = []
    
    for cluster_sets, labels, tracksters in loader:
        labels = labels.to(device)
        tracksters = tracksters.to(device)
        cluster_sets = [c.to(device) for c in cluster_sets]
        
        out = model(cluster_sets, tracksters)
        loss = criterion(out, labels)
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        group_correct += criterion.get_group_accuracy(out, labels) * labels.size(0)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    class_acc = correct / max(total, 1)
    group_acc = group_correct / max(total, 1)
    
    return total_loss / len(loader), class_acc, group_acc, all_preds, all_labels


# 6. MAIN TRAINING SCRIPT

def main():
    parser = argparse.ArgumentParser(description='Train Set Transformer on TICL cluster data')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input HDF5 file')
    parser.add_argument('--output-dir', '-o', type=str, default='./output',
                       help='Output directory for models and plots')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=60,
                       help='Number of epochs')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--layers', type=int, default=3,
                       help='Number of transformer layers')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of DataLoader workers')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use-trackster', action='store_true',
                       help='Use trackster features in model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load dataset
    print("=" * 60)
    print("Loading dataset...")
    full_dataset = TracksterDataset(args.input, seed=args.seed)
    
    # Split into train/val/test
    n_total = len(full_dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_sets,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sets,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sets,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0
    )
    
    # Create model
    model = SetTransformer(
        hidden_dim=args.hidden_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        num_classes=num_classes,
        use_trackster_features=args.use_trackster
    )
    model = model.to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    # Loss function
    criterion = HierarchicalLoss().to(args.device)
    
    # Training tracking
    best_val_group_acc = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_group_accs, val_group_accs = [], []
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss, train_acc, train_group_acc = train_epoch(
            model, train_loader, optimizer, criterion, args.device
        )
        val_loss, val_acc, val_group_acc, _, _ = evaluate(
            model, val_loader, criterion, args.device
        )
        
        scheduler.step(val_group_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_group_accs.append(train_group_acc)
        val_group_accs.append(val_group_acc)
        
        print(f"\nEpoch {epoch+1:3d}/{args.epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss={train_loss:.4f}, Class Acc={train_acc:.4f}, Group Acc={train_group_acc:.4f}")
        print(f"  Val Loss={val_loss:.4f}, Class Acc={val_acc:.4f}, Group Acc={val_group_acc:.4f}, LR={current_lr:.2e}")
        
        # Save best model based on group accuracy
        if val_group_acc > best_val_group_acc:
            best_val_group_acc = val_group_acc
            model_path = os.path.join(args.output_dir, 'best_set_transformer.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_group_acc': val_group_acc,
                'val_acc': val_acc,
                'args': args
            }, model_path)
            print(f" New best model saved! (Group Acc: {val_group_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    
    # Load best model and evaluate on test set
    model_path = os.path.join(args.output_dir, 'best_set_transformer.pt')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_group_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, args.device
    )
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Class Accuracy: {test_acc:.4f}")
    print(f"  Group Accuracy (EM/HAD/Other): {test_group_acc:.4f}")
    print("=" * 60)
    
    # EM vs HAD confusion matrix
    em_had_cm = criterion.get_em_vs_had_confusion(
        torch.tensor(test_preds), 
        torch.tensor(test_labels)
    )
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(em_had_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['EM', 'HAD'],
                yticklabels=['EM', 'HAD'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('EM vs HAD Confusion Matrix', fontsize=14, fontweight='bold')
    
    if em_had_cm.sum() > 0:
        em_had_acc = (em_had_cm[0,0] + em_had_cm[1,1]) / em_had_cm.sum()
        plt.text(0.5, -0.15, f'Accuracy: {em_had_acc:.4f}', 
                 transform=plt.gca().transAxes, ha='center', fontsize=12)
    
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, 'em_vs_had_confusion.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")
    
    # Print confusion matrix numbers
    print("\nEM vs HAD Confusion Matrix:")
    print("           Predicted")
    print("           EM    HAD")
    print(f"True EM    {em_had_cm[0,0]:5d}  {em_had_cm[0,1]:5d}")
    print(f"     HAD   {em_had_cm[1,0]:5d}  {em_had_cm[1,1]:5d}")
    
    if em_had_cm.sum() > 0:
        em_had_acc = (em_had_cm[0,0] + em_had_cm[1,1]) / em_had_cm.sum()
        print(f"\nEM vs HAD Accuracy: {em_had_acc:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Class Acc')
    plt.plot(val_accs, label='Val Class Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Class Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(train_group_accs, label='Train Group Acc')
    plt.plot(val_group_accs, label='Val Group Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('EM/HAD/Other Group Accuracy')
    
    plt.tight_layout()
    curves_path = os.path.join(args.output_dir, 'training_curves.png')
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {curves_path}")
    
    # ONNX export
    print("\nExporting to ONNX...")
    
    class SetTransformerONNX(nn.Module):
        def __init__(self, model, max_clusters=200):
            super().__init__()
            self.model = model
            self.max_clusters = max_clusters
        
        def forward(self, x, mask, trackster_features=None):
            # x: [batch_size, max_clusters, 4]
            # mask: [batch_size, max_clusters]
            
            # Encode clusters
            x = self.model.encoder(x)
            
            # Self-attention
            for attn, norm, ff in zip(self.model.attention_layers, 
                                      self.model.norm_layers, 
                                      self.model.ff_layers):
                attn_out, _ = attn(x, x, x, key_padding_mask=~mask.bool())
                x = norm(x + attn_out)
                x = x + ff(x)
            
            # Pooling
            expanded_mask = mask.unsqueeze(-1).float()
            x_masked = x * expanded_mask
            x_mean = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            x_max = (x_masked + (1 - expanded_mask) * -1e9).max(dim=1)[0]
            x_pooled = torch.cat([x_mean, x_max], dim=1)
            
            # Add trackster features if used
            if self.model.use_trackster_features and trackster_features is not None:
                trackster_encoded = self.model.trackster_encoder(trackster_features)
                x_pooled = torch.cat([x_pooled, trackster_encoded], dim=1)
            
            return self.model.classifier(x_pooled)
    
    onnx_model = SetTransformerONNX(model, max_clusters=200).eval()
    
    # Test wrapper
    with torch.no_grad():
        test_x = torch.randn(2, 200, 4)
        test_mask = torch.ones(2, 200)
        test_mask[0, 150:] = 0
        test_trackster = torch.randn(2, 3) if args.use_trackster else None
        
        output = onnx_model(test_x, test_mask, test_trackster)
        print(f"ONNX test output shape: {output.shape}")
    
    # Export
    dummy_x = torch.randn(1, 200, 4)
    dummy_mask = torch.ones(1, 200)
    dummy_trackster = torch.randn(1, 3) if args.use_trackster else None
    
    input_names = ['clusters', 'mask']
    if args.use_trackster:
        input_names.append('trackster_features')
    
    torch.onnx.export(
        onnx_model,
        (dummy_x, dummy_mask, dummy_trackster) if args.use_trackster else (dummy_x, dummy_mask),
        os.path.join(args.output_dir, 'set_transformer.onnx'),
        input_names=input_names,
        output_names=['output'],
        dynamic_axes={
            'clusters': {0: 'batch_size'},
            'mask': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=14
    )
    print(f"Model converted to {os.path.join(args.output_dir, 'set_transformer.onnx')}")


if __name__ == "__main__":
    main()
