#!/usr/bin/env python3
"""
    # Evaluate PyTorch model
    python evaluate_model.py --input ticl_dataset.h5 --checkpoint ./output/best_set_transformer.pt --device cuda

    # Evaluate ONNX model
    python evaluate_model.py --input ticl_dataset.h5 --onnx ./output/set_transformer.onnx

    # Compare both (to verify ONNX export)
    python evaluate_model.py --input ticl_dataset.h5 --checkpoint ./output/best_set_transformer.pt --onnx ./output/set_transformer.onnx --compare

    # With specific batch size
    python evaluate_model.py --input ticl_dataset.h5 --checkpoint best.pt --batch-size 64
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import argparse
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import onnxruntime as ort
from collections import defaultdict

# 1. LABEL MAPPING (same as training)

class_labels = {
    22: 0, 111: 0,     # photon, pion0
    11: 1, -11: 1,     # electron/positron
    13: 2, -13: 2,     # muon
    1011: 3,           # ?
    211: 4, -211: 4,   # charged hadron
    321: 4, -321: 4,   # charged hadron
    310: 5, 130: 5,    # neutral hadron
    -1: 6,             # unknown
    0: 7               # noise/empty
}

num_classes = 8
class_names = ['Photon', 'Electron', 'Muon', 'Other', 
               'Charged Hadron', 'Neutral Hadron', 'Unknown', 'Noise']
group_names = ['EM', 'HAD', 'Other']

# 2. DATASET (same as training)

class TracksterDataset(torch.utils.data.Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        self._handle = None
        
        with h5py.File(h5_file, 'r') as f:
            self.num_tracksters = int(f.attrs['num_tracksters'])
            num_clusters = f['num_clusters'][:]
            self.valid_indices = np.where(num_clusters > 0)[0]
        
        print(f"Dataset: {len(self.valid_indices):,} tracksters with clusters")
    
    def _get_handle(self):
        if self._handle is None:
            self._handle = h5py.File(self.h5_file, 'r', swmr=True)
        return self._handle
    
    def __del__(self):
        if self._handle is not None:
            try:
                self._handle.close()
            except:
                pass
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        f = self._get_handle()
        
        trackster = torch.from_numpy(f['features'][i].astype(np.float32))
        true_pid = int(f['true_pid'][i])
        label = torch.tensor(class_labels.get(true_pid, 6), dtype=torch.long)
        n_clusters = int(f['num_clusters'][i])
        clusters_raw = f['clusters'][i]
        
        if n_clusters > 0:
            clusters = torch.from_numpy(clusters_raw.reshape(n_clusters, 4).astype(np.float32))
        else:
            clusters = torch.zeros((0, 4), dtype=torch.float32)
        
        return clusters, label, trackster, i

def collate_sets(batch):
    cluster_sets = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    tracksters = torch.stack([item[2] for item in batch])
    indices = [item[3] for item in batch]
    return cluster_sets, labels, tracksters, indices

# 3. MODEL DEFINITION (for PyTorch evaluation)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        x2, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                          key_padding_mask=key_padding_mask)
        x = x + x2
        x = x + self.ff(self.norm2(x))
        return x

class SetTransformer(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=8, num_layers=3,
                 num_classes=8, dropout=0.1, use_trackster_features=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_trackster_features = use_trackster_features
        
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        if use_trackster_features:
            self.trackster_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
            )
            pool_input_dim = hidden_dim * 2 + hidden_dim // 2
        else:
            pool_input_dim = hidden_dim * 2
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(pool_input_dim),
            nn.Linear(pool_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, cluster_sets, trackster_features=None):
        device = cluster_sets[0].device
        max_clusters = max(c.size(0) for c in cluster_sets)
        
        padded, masks = [], []
        for c in cluster_sets:
            n = c.size(0)
            if n < max_clusters:
                pad = torch.zeros(max_clusters - n, c.size(1), device=device)
                c = torch.cat([c, pad], dim=0)
                mask = torch.cat([torch.ones(n), torch.zeros(max_clusters - n)])
            else:
                mask = torch.ones(max_clusters)
            padded.append(c)
            masks.append(mask)
        
        x = torch.stack(padded)
        mask = torch.stack(masks).to(device)
        
        x = self.encoder(x)
        pad_mask = ~mask.bool()
        
        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)
        
        exp_mask = mask.unsqueeze(-1)
        x_masked = x * exp_mask
        x_mean = x_masked.sum(1) / exp_mask.sum(1).clamp(min=1)
        x_max = (x_masked + (1 - exp_mask) * -1e9).max(1)[0]
        x_pool = torch.cat([x_mean, x_max], dim=1)
        
        if self.use_trackster_features and trackster_features is not None:
            x_pool = torch.cat([x_pool, self.trackster_encoder(trackster_features)], dim=1)
        
        return self.classifier(x_pool)

# 4. METRICS AND EVALUATION

class Metrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total = 0
        self.correct = 0
        self.loss_sum = 0
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, preds, labels, loss=None, probs=None):
        batch_size = len(labels)
        self.total += batch_size
        self.correct += (preds == labels).sum().item()
        if loss is not None:
            self.loss_sum += loss * batch_size
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        if probs is not None:
            self.all_probs.extend(probs.cpu().numpy())
    
    @property
    def accuracy(self):
        return self.correct / self.total if self.total > 0 else 0
    
    @property
    def avg_loss(self):
        return self.loss_sum / self.total if self.total > 0 else 0
    
    def get_group_labels(self, class_ids):
        groups = np.zeros_like(class_ids)
        em = (class_ids == 0) | (class_ids == 1)
        had = (class_ids == 4) | (class_ids == 5)
        groups[em] = 0
        groups[had] = 1
        groups[~(em | had)] = 2
        return groups
    
    def get_group_accuracy(self):
        if not self.all_labels:
            return 0
        true_groups = self.get_group_labels(np.array(self.all_labels))
        pred_groups = self.get_group_labels(np.array(self.all_preds))
        return (true_groups == pred_groups).mean()
    
    def get_em_vs_had_confusion(self):
        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)
        
        true_groups = self.get_group_labels(labels)
        pred_groups = self.get_group_labels(preds)
        
        em_had_mask = (true_groups != 2)
        if not em_had_mask.any():
            return np.zeros((2, 2), dtype=int)
        
        return confusion_matrix(
            true_groups[em_had_mask],
            pred_groups[em_had_mask],
            labels=[0, 1]
        )
    
    def get_per_class_accuracy(self):
        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)
        
        per_class = {}
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                per_class[c] = (preds[mask] == c).mean()
            else:
                per_class[c] = float('nan')
        return per_class

# 5. PYTORCH EVALUATION

@torch.no_grad()
def evaluate_pytorch(model, loader, device):
    model.eval()
    metrics = Metrics()
    
    for cluster_sets, labels, tracksters, indices in tqdm(loader, desc="PyTorch Evaluation"):
        labels = labels.to(device)
        tracksters = tracksters.to(device)
        cluster_sets = [c.to(device) for c in cluster_sets]
        
        outputs = model(cluster_sets, tracksters)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        metrics.update(preds, labels, probs=probs)
    
    return metrics

# 6. ONNX EVALUATION

def evaluate_onnx(onnx_path, loader, max_clusters=200):
    import onnxruntime as ort
    
    # Create session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Get input names
    input_names = [inp.name for inp in session.get_inputs()]
    print(f"ONNX Inputs: {input_names}")
    
    metrics = Metrics()
    
    for cluster_sets, labels, tracksters, indices in tqdm(loader, desc="ONNX Evaluation"):
        # Prepare padded batch
        batch_size = len(cluster_sets)
        max_n = max(c.size(0) for c in cluster_sets)
        max_n = min(max_n, max_clusters)  # Cap at max_clusters
        
        # Create padded tensors
        x_np = np.zeros((batch_size, max_n, 4), dtype=np.float32)
        mask_np = np.zeros((batch_size, max_n), dtype=np.float32)
        
        for i, c in enumerate(cluster_sets):
            n = min(c.size(0), max_n)
            x_np[i, :n] = c[:n].numpy()
            mask_np[i, :n] = 1
        
        # Prepare inputs based on what the model expects
        onnx_inputs = {}
        if 'clusters' in input_names:
            onnx_inputs['clusters'] = x_np
        if 'mask' in input_names:
            onnx_inputs['mask'] = mask_np
        if 'trackster_features' in input_names and tracksters is not None:
            onnx_inputs['trackster_features'] = tracksters.numpy()
        
        # Run inference
        outputs = session.run(None, onnx_inputs)[0]
        preds = torch.from_numpy(outputs).argmax(dim=1)
        
        metrics.update(preds, labels)
    
    return metrics

# 7. PLOTTING FUNCTIONS

def plot_all_metrics(metrics, output_dir, prefix=''):
    """Generate all evaluation plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confusion Matrix (full)
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(metrics.all_labels, metrics.all_preds)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{int(cm[i,j])}\n({cm_pct[i,j]:.1f}%)" 
                       for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    sns.heatmap(cm_pct, annot=annot, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=100, cbar_kws={'label': 'Percentage (%)'})
    #numbers plus %
    #sns.heatmap(cm_pct, annot=annot, fmt='', cmap='Blues',
    #            xticklabels=class_names, yticklabels=class_names,
    #            vmin=0, vmax=100, cbar_kws={'label': 'Percentage (%)'})
    #numbers
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #            xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Full Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}confusion_matrix_full.png'), dpi=150)
    plt.close()
    
    # 2. EM vs HAD Confusion Matrix
    em_had_cm = metrics.get_em_vs_had_confusion()
    plt.figure(figsize=(6, 5))
    em_had_cm_pct = em_had_cm / em_had_cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{int(em_had_cm[i,j])}\n({em_had_cm_pct[i,j]:.1f}%)" 
                       for j in range(2)] for i in range(2)])
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=100, cbar_kws={'label': 'Percentage (%)'})
    #numbers
    #sns.heatmap(em_had_cm, annot=True, fmt='d', cmap='Blues',
    #            xticklabels=['EM', 'HAD'], yticklabels=['EM', 'HAD'])
    #numbers + %
    sns.heatmap(em_had_cm_pct, annot=annot, fmt='', cmap='Blues',
                xticklabels=['EM', 'HAD'], yticklabels=['EM', 'HAD'],
                vmin=0, vmax=100, cbar_kws={'label': 'Percentage (%)'})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('EM vs HAD Confusion Matrix')
    
    if em_had_cm.sum() > 0:
        acc = (em_had_cm[0,0] + em_had_cm[1,1]) / em_had_cm.sum()
        plt.text(0.5, -0.15, f'Accuracy: {acc:.4f}',
                 transform=plt.gca().transAxes, ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}confusion_em_vs_had.png'), dpi=150)
    plt.close()
    
    # 3. Per-class accuracy bar chart
    per_class = metrics.get_per_class_accuracy()
    plt.figure(figsize=(12, 6))
    classes = list(per_class.keys())
    accs = [per_class[c] if not np.isnan(per_class[c]) else 0 for c in classes]
    colors = ['green' if not np.isnan(per_class[c]) else 'gray' for c in classes]
    
    plt.bar(range(len(classes)), accs, color=colors, alpha=0.7)
    plt.xticks(range(len(classes)), [class_names[c] for c in classes], rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, (acc, c) in enumerate(zip(accs, classes)):
        if not np.isnan(per_class[c]):
            plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}per_class_accuracy.png'), dpi=150)
    plt.close()
    
    # 4. Group accuracy pie chart
    true_groups = metrics.get_group_labels(np.array(metrics.all_labels))
    pred_groups = metrics.get_group_labels(np.array(metrics.all_preds))
    
    group_correct = (true_groups == pred_groups).astype(float)
    
    plt.figure(figsize=(8, 8))
    plt.pie([group_correct.mean(), 1 - group_correct.mean()],
            labels=['Correct Group', 'Wrong Group'],
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.1f%%',
            explode=(0.05, 0.05))
    plt.title(f'Group Accuracy (EM/HAD/Other): {group_correct.mean():.4f}')
    plt.savefig(os.path.join(output_dir, f'{prefix}group_accuracy.png'), dpi=150)
    plt.close()
    
    # 5. Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary - {prefix}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {metrics.accuracy:.4f}")
    print(f"Group Accuracy: {metrics.get_group_accuracy():.4f}")
    print(f"\nPer-Class Accuracy:")
    for c, acc in per_class.items():
        status = f"{acc:.4f}" if not np.isnan(acc) else "N/A (no samples)"
        print(f"  {class_names[c]:<15}: {status}")

# ============================================
# 8. MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Set Transformer model')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input HDF5 file')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='PyTorch checkpoint .pt file')
    parser.add_argument('--onnx', '-o', type=str, default=None,
                       help='ONNX model .onnx file')
    parser.add_argument('--compare', action='store_true',
                       help='Compare PyTorch and ONNX results')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of DataLoader workers')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for PyTorch evaluation')
    parser.add_argument('--output-dir', type=str, default='./evaluation',
                       help='Output directory for plots')
    parser.add_argument('--max-clusters', type=int, default=200,
                       help='Max clusters for ONNX padding')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = TracksterDataset(args.input)
    
    # Use all data for evaluation
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sets,
        pin_memory=(args.device == 'cuda')
    )
    
    results = {}
    
    # ===== PyTorch Evaluation =====
    if args.checkpoint:
        print(f"\n{'='*60}")
        print(f"Evaluating PyTorch model: {args.checkpoint}")
        print(f"{'='*60}")
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        
        # Get model parameters from checkpoint
        model_args = checkpoint.get('args', argparse.Namespace())
        hidden_dim = getattr(model_args, 'hidden_dim', 128)
        num_heads = getattr(model_args, 'heads', 8)
        num_layers = getattr(model_args, 'layers', 3)
        use_trackster = getattr(model_args, 'use_trackster', True)
        
        # Create model
        model = SetTransformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            use_trackster_features=use_trackster
        )
        
        # Load state dict
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(args.device)
        
        # Evaluate
        start_time = time.time()
        metrics_pt = evaluate_pytorch(model, loader, args.device)
        pt_time = time.time() - start_time
        
        results['pytorch'] = {
            'metrics': metrics_pt,
            'time': pt_time,
            'accuracy': metrics_pt.accuracy,
            'group_accuracy': metrics_pt.get_group_accuracy()
        }
        
        # Plot PyTorch results
        plot_all_metrics(metrics_pt, args.output_dir, prefix='pytorch_')
        
        print(f"\nPyTorch Evaluation completed in {pt_time:.2f}s")
        print(f"Accuracy: {metrics_pt.accuracy:.4f}")
        print(f"Group Accuracy: {metrics_pt.get_group_accuracy():.4f}")
    
    # ===== ONNX Evaluation =====
    if args.onnx:
        print(f"\n{'='*60}")
        print(f"Evaluating ONNX model: {args.onnx}")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            metrics_onnx = evaluate_onnx(args.onnx, loader, args.max_clusters)
            onnx_time = time.time() - start_time
            
            results['onnx'] = {
                'metrics': metrics_onnx,
                'time': onnx_time,
                'accuracy': metrics_onnx.accuracy,
                'group_accuracy': metrics_onnx.get_group_accuracy()
            }
            
            # Plot ONNX results
            plot_all_metrics(metrics_onnx, args.output_dir, prefix='onnx_')
            
            print(f"\nONNX Evaluation completed in {onnx_time:.2f}s")
            print(f"Accuracy: {metrics_onnx.accuracy:.4f}")
            print(f"Group Accuracy: {metrics_onnx.get_group_accuracy():.4f}")
            
        except Exception as e:
            print(f"Error evaluating ONNX model: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== Comparison =====
    if args.compare and 'pytorch' in results and 'onnx' in results:
        print(f"\n{'='*60}")
        print("PyTorch vs ONNX Comparison")
        print(f"{'='*60}")
        
        pt_metrics = results['pytorch']['metrics']
        onnx_metrics = results['onnx']['metrics']
        
        # Compare predictions
        pt_preds = np.array(pt_metrics.all_preds)
        onnx_preds = np.array(onnx_metrics.all_preds)
        
        agreement = (pt_preds == onnx_preds).mean()
        
        print(f"Prediction Agreement: {agreement:.6f}")
        print(f"\nAccuracy Difference: {abs(pt_metrics.accuracy - onnx_metrics.accuracy):.6f}")
        print(f"Group Accuracy Difference: {abs(pt_metrics.get_group_accuracy() - onnx_metrics.get_group_accuracy()):.6f}")
        print(f"\nSpeed Comparison:")
        print(f"  PyTorch: {results['pytorch']['time']:.2f}s")
        print(f"  ONNX:    {results['onnx']['time']:.2f}s")
        print(f"  Speedup: {results['pytorch']['time']/results['onnx']['time']:.2f}x")
        
        # Plot comparison
        plt.figure(figsize=(8, 6))
        disagreement_mask = pt_preds != onnx_preds
        if disagreement_mask.any():
            plt.hist([pt_preds[disagreement_mask], onnx_preds[disagreement_mask]], 
                     label=['PyTorch', 'ONNX'], alpha=0.7, bins=range(num_classes+1))
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title(f'Disagreements ({disagreement_mask.sum()} samples)')
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, 'comparison_disagreements.png'), dpi=150)
            plt.close()
    
    # ===== Summary =====
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"Plots saved to: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
