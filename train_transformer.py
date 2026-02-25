"""
# Train the Set Transformer model (8 classes, EM vs HAD group loss)
python train_transformer.py --input ticl_dataset.h5 --batch-size 32 --epochs 50

# Adjust hyperparameters
python train_transformer.py --input ticl_dataset.h5 --hidden-dim 128 --lr 0.0005 --heads 8

# With mixed precision (faster on GPU)
python train_transformer.py --input ticl_dataset.h5 --amp

# With torch.compile (PyTorch 2.0+, free ~30% speedup)
python train_transformer.py --input ticl_dataset.h5 --compile

# Train using only clusters
python train_transformer.py --input ticl_dataset.h5 --input-mode clusters

# Train using only trackster features
python train_transformer.py --input ticl_dataset.h5 --input-mode trackster

# Train using both (default)
python train_transformer.py --input ticl_dataset.h5 --input-mode both

# Resume training from checkpoint
python train_transformer.py --input ticl_dataset.h5 --epochs 30 --resume ./output/best_set_transformer.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from tqdm import tqdm
import argparse
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# 1. LABEL MAPPING  full 8 classes, all PIDs kept

class_labels = {
    22:   0,             # photon
    11:   1,  -11:  1,   # electron
    13:   2,  -13:  2,   # muon            
    111:  3,             # pi0          
    211:  4,  -211: 4,   # charged hadron
    321:  4,  -321: 4,   # charged hadron
    310:  5,   130:  5,   # nuetral hadron
    -1:   6,             # unknown        
    0:    7,             # noise/empty    
}

num_classes = 8

EM_CLASS_INDICES  = [0, 1]
HAD_CLASS_INDICES = [4, 5]


# 2. DATASET
class TracksterDataset(Dataset):
    def __init__(self, h5_file, split='train',
                 train_ratio=0.7, val_ratio=0.15, seed=42,
                 exclude_pid=None):

        self.h5_file = h5_file
        self.split   = split
        self._handle = None

        with h5py.File(h5_file, 'r') as f:
            total        = int(f.attrs['num_tracksters'])
            num_clusters = f['num_clusters'][:]
            true_pids    = f['true_pid'][:]

        # Keep only tracksters with at least one cluster
        valid = np.where(num_clusters > 0)[0]

        # Exclude specific PID from training split if requested
        if exclude_pid is not None and split == 'train':
            pid_mask = np.abs(true_pids[valid]) != abs(exclude_pid)
            valid    = valid[pid_mask]
            print(f"Excluded PID {exclude_pid} from training. Remaining: {len(valid):,}")

        # Count EM vs HAD for info
        pids_valid  = np.abs(true_pids[valid])
        em_pids     = {22, 11}
        had_pids    = {211, 321, 310, 130}
        em_count    = int(np.sum([p in em_pids  for p in pids_valid]))
        had_count   = int(np.sum([p in had_pids for p in pids_valid]))
        other_count = len(valid) - em_count - had_count

        print(f"Total: {total:,}  |  valid: {len(valid):,}  "
              f"(EM={em_count:,}, HAD={had_count:,}, other={other_count:,})")

        rng = np.random.RandomState(seed)
        rng.shuffle(valid)

        n       = len(valid)
        n_train = int(train_ratio * n)
        n_val   = int(val_ratio   * n)

        if split == 'train':
            self.indices = valid[:n_train]
        elif split == 'val':
            self.indices = valid[n_train:n_train + n_val]
        else:
            self.indices = valid[n_train + n_val:]

        print(f" {split}: {len(self.indices):,} tracksters")

    def _get_handle(self):
        if self._handle is None:
            self._handle = h5py.File(self.h5_file, 'r', swmr=True)
        return self._handle

    def __del__(self):
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        f = self._get_handle()

        trackster  = torch.from_numpy(f['features'][i].astype(np.float32))
        true_pid   = int(f['true_pid'][i])
        label      = torch.tensor(class_labels.get(true_pid, 6), dtype=torch.long)
        n_clusters = int(f['num_clusters'][i])
        raw        = f['clusters'][i]

        if n_clusters > 0:
            clusters = torch.from_numpy(raw.reshape(n_clusters, 4).astype(np.float32))
        else:
            clusters = torch.zeros((0, 4), dtype=torch.float32)

        return clusters, label, trackster


def worker_init_fn(worker_id):
    import gc
    for obj in gc.get_objects():
        if isinstance(obj, TracksterDataset):
            obj._handle = None


def collate_sets(batch):
    cluster_sets = [item[0] for item in batch]
    labels       = torch.stack([item[1] for item in batch])
    tracksters   = torch.stack([item[2] for item in batch])
    return cluster_sets, labels, tracksters


class GroupCollapsedLoss(nn.Module):

    def __init__(self, em_indices=EM_CLASS_INDICES, had_indices=HAD_CLASS_INDICES):
        super().__init__()
        self.em_idx  = em_indices
        self.had_idx = had_indices
        self.ce      = nn.CrossEntropyLoss()

    def _group_logits(self, pred):
        em_score  = pred[:, self.em_idx].sum(dim=1, keepdim=True)
        had_score = pred[:, self.had_idx].sum(dim=1, keepdim=True)
        return torch.cat([em_score, had_score], dim=1)   # [B, 2]

    def _group_target(self, target):
        is_em  = sum(target == i for i in self.em_idx).bool()
        is_had = sum(target == i for i in self.had_idx).bool()
        valid  = is_em | is_had                          # exclude "other"
        group_t = torch.zeros_like(target)
        group_t[is_had] = 1
        return group_t, valid

    def forward(self, pred, target):
        group_t, valid = self._group_target(target)

        # If no EM or HAD samples in this batch, return zero loss
        if valid.sum() == 0:
            return pred.sum() * 0.0

        group_pred = self._group_logits(pred[valid])     # [valid_B, 2]
        group_t    = group_t[valid]                      # [valid_B]

        return self.ce(group_pred, group_t)

    def get_group_accuracy(self, pred, target):
        group_t, valid = self._group_target(target)
        if valid.sum() == 0:
            return 0.0
        gp = self._group_logits(pred[valid]).argmax(1)
        return (gp == group_t[valid]).float().mean().item()

    def get_em_had_confusion(self, pred, target):
        group_t, valid = self._group_target(target)
        if valid.sum() == 0:
            return np.zeros((2, 2), dtype=int)
        gp = self._group_logits(pred[valid]).argmax(1)
        return confusion_matrix(
            group_t[valid].cpu().numpy(),
            gp.cpu().numpy(), labels=[0, 1])


# 4. MODEL
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn  = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        n = self.norm1(x)
        x2, _ = self.attn(n, n, n, key_padding_mask=key_padding_mask)
        x = x + x2
        x = x + self.ff(self.norm2(x))
        return x


class SetTransformer(nn.Module):

    def __init__(self,
                 node_features=4,
                 hidden_dim=128,
                 num_heads=8,
                 num_layers=3,
                 num_classes=8,
                 dropout=0.1,
                 input_mode='both'):  

        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_mode = input_mode  

        # ALWAYS create the cluster encoder and transformer blocks
        self.encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Transformer blocks (always created)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # ALWAYS create trackster encoder (but we may not use it)
        self.trackster_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        # Calculate pool dimension based on input_mode
        if input_mode == 'clusters':
            pool_input_dim = hidden_dim * 2  # only cluster features
        elif input_mode == 'trackster':
            pool_input_dim = hidden_dim // 2  # only trackster features
        else:  # 'both'
            pool_input_dim = hidden_dim * 2 + hidden_dim // 2

        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(pool_input_dim),
            nn.Linear(pool_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, cluster_sets, trackster_features=None):
        device = cluster_sets[0].device if cluster_sets else trackster_features.device
        parts = []
        
        # Only process clusters if we're in a mode that uses them
        if self.input_mode in ['clusters', 'both'] and cluster_sets:
            max_clusters = max(c.size(0) for c in cluster_sets)

            # Pad to uniform length within batch
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

            x = torch.stack(padded)  # [B, N, 4]
            mask = torch.stack(masks).to(device)  # [B, N]

            # Encode clusters
            x = self.encoder(x)  # [B, N, H]

            # Transformer blocks
            pad_mask = ~mask.bool()  # True = ignore (HF convention)
            for block in self.blocks:
                x = block(x, key_padding_mask=pad_mask)

            # Mean + max pooling over real clusters only
            exp_mask = mask.unsqueeze(-1)  # [B, N, 1]
            x_masked = x * exp_mask
            x_mean = x_masked.sum(1) / exp_mask.sum(1).clamp(min=1)
            x_max = (x_masked + (1 - exp_mask) * -1e9).max(1)[0]
            cluster_features = torch.cat([x_mean, x_max], dim=1)  # [B, H*2]
            parts.append(cluster_features)
        
        # Only process trackster features if we're in a mode that uses them
        if self.input_mode in ['trackster', 'both'] and trackster_features is not None:
            trackster_features_encoded = self.trackster_encoder(trackster_features)
            parts.append(trackster_features_encoded)
        
        # Concatenate all features
        if len(parts) == 0:
            raise RuntimeError(f"No features extracted! input_mode={self.input_mode}")
        
        x_pool = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.classifier(x_pool)
    
# 5. TRAINING
def train_epoch(model, loader, optimizer, criterion, device, scaler=None, grad_clip=1.0):
    model.train()
    total_loss, total, group_correct = 0, 0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for cluster_sets, labels, tracksters in pbar:
        if not cluster_sets:
            continue

        labels       = labels.to(device)
        tracksters   = tracksters.to(device)
        cluster_sets = [c.to(device) for c in cluster_sets]

        optimizer.zero_grad(set_to_none=True)

        amp_device = 'cuda' if 'cuda' in str(device) else 'cpu'
        with torch.autocast(device_type=amp_device, enabled=scaler is not None):
            out  = model(cluster_sets, tracksters)
            loss = criterion(out, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss   += loss.item()
        group_correct += criterion.get_group_accuracy(out, labels) * labels.size(0)
        total        += labels.size(0)

        pbar.set_postfix(loss=f'{loss.item():.4f}',
                         group_acc=f'{group_correct/max(total,1):.3f}')

    return total_loss / len(loader), group_correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, scaler=None):
    model.eval()
    total_loss, total, group_correct = 0, 0, 0
    all_preds, all_labels = [], []

    for cluster_sets, labels, tracksters in loader:
        labels       = labels.to(device)
        tracksters   = tracksters.to(device)
        cluster_sets = [c.to(device) for c in cluster_sets]

        amp_device = 'cuda' if 'cuda' in str(device) else 'cpu'
        with torch.autocast(device_type=amp_device, enabled=scaler is not None):
            out  = model(cluster_sets, tracksters)
            loss = criterion(out, labels)

        total_loss   += loss.item()
        group_correct += criterion.get_group_accuracy(out, labels) * labels.size(0)
        total        += labels.size(0)

        # Store raw 8-class predictions for confusion matrix
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / len(loader),
            group_correct / max(total, 1),
            all_preds, all_labels)


# 6. MAIN
def export_to_onnx(model, args, device):
    print("\n" + "=" * 60)
    print("Exporting model to ONNX")
    print("=" * 60)
    
    # Create mode-specific wrapper classes
    if args.input_mode == 'trackster':
        class SetTransformerONNXTrackster(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                print("ONNX wrapper initialized for trackster-only mode")

            def forward(self, trackster_features):
                # Only process trackster features
                features = self.m.trackster_encoder(trackster_features)
                return self.m.classifier(features)
        wrapper_class = SetTransformerONNXTrackster
        
    elif args.input_mode == 'clusters':
        class SetTransformerONNXClusters(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                print("ONNX wrapper initialized for clusters-only mode")

            def forward(self, x, mask):
                # Only process clusters
                z = self.m.encoder(x)
                for block in self.m.blocks:
                    z = block(z, key_padding_mask=~mask.bool())
                exp = mask.unsqueeze(-1)
                zmean = (z * exp).sum(1) / exp.sum(1).clamp(min=1)
                zmax = (z * exp + (1 - exp) * -1e9).max(1)[0]
                features = torch.cat([zmean, zmax], dim=1)
                return self.m.classifier(features)
        wrapper_class = SetTransformerONNXClusters
        
    else:  # 'both'
        class SetTransformerONNXBoth(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                print("ONNX wrapper initialized for both-mode")

            def forward(self, x, mask, trackster_features):
                # Process clusters
                z = self.m.encoder(x)
                for block in self.m.blocks:
                    z = block(z, key_padding_mask=~mask.bool())
                exp = mask.unsqueeze(-1)
                zmean = (z * exp).sum(1) / exp.sum(1).clamp(min=1)
                zmax = (z * exp + (1 - exp) * -1e9).max(1)[0]
                cluster_features = torch.cat([zmean, zmax], dim=1)
                
                # Process trackster
                track_features = self.m.trackster_encoder(trackster_features)
                
                # Combine
                features = torch.cat([cluster_features, track_features], dim=1)
                return self.m.classifier(features)
        wrapper_class = SetTransformerONNXBoth

    # Prepare model for export
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    onnx_model = wrapper_class(raw).eval().cpu()
    
    # Create dummy inputs based on mode
    dyn = {'output': {0: 'batch'}}
    
    print(f"\nCreating dummy inputs for mode: {args.input_mode}")
    
    if args.input_mode == 'trackster':
        input_names = ['trackster_features']
        dummy_trackster = torch.randn(1, 3)
        dummy_args = (dummy_trackster,)
        dyn['trackster_features'] = {0: 'batch'}
        print(f"  Added trackster_features: shape {dummy_trackster.shape}")
        
    elif args.input_mode == 'clusters':
        input_names = ['clusters', 'mask']
        dummy_clusters = torch.randn(1, 200, 4)
        dummy_mask = torch.ones(1, 200)
        dummy_args = (dummy_clusters, dummy_mask)
        dyn['clusters'] = {0: 'batch'}
        dyn['mask'] = {0: 'batch'}
        print(f"  Added clusters: shape {dummy_clusters.shape}")
        print(f"  Added mask: shape {dummy_mask.shape}")
        
    else:  # 'both'
        input_names = ['clusters', 'mask', 'trackster_features']
        dummy_clusters = torch.randn(1, 200, 4)
        dummy_mask = torch.ones(1, 200)
        dummy_trackster = torch.randn(1, 3)
        dummy_args = (dummy_clusters, dummy_mask, dummy_trackster)
        dyn['clusters'] = {0: 'batch'}
        dyn['mask'] = {0: 'batch'}
        dyn['trackster_features'] = {0: 'batch'}
        print(f"  Added clusters: shape {dummy_clusters.shape}")
        print(f"  Added mask: shape {dummy_mask.shape}")
        print(f"  Added trackster_features: shape {dummy_trackster.shape}")

    # Test the ONNX wrapper with dummy args before exporting
    print("\nTesting ONNX wrapper with dummy args...")
    with torch.no_grad():
        test_output = onnx_model(*dummy_args)
        print(f"Test forward pass successful! Output shape: {test_output.shape}")
    
    # Export
    onnx_path = os.path.join(args.output_dir, f'set_transformer_{args.input_mode}.onnx')
    print(f"\nExporting with inputs: {input_names}")
    print(f"Dummy args shapes: {[a.shape for a in dummy_args]}")
    
    torch.onnx.export(onnx_model, dummy_args, onnx_path,
                      input_names=input_names, output_names=['output'],
                      dynamic_axes=dyn, opset_version=14)
    print(f"ONNX saved to {onnx_path}")
    return onnx_path

def main():
    parser = argparse.ArgumentParser(
        description='Set Transformer 8 classes, EM vs HAD group loss')
    parser.add_argument('--input',       '-i', type=str, required=True,
                        help='Input HDF5 dataset file')
    parser.add_argument('--output-dir',  '-o', type=str, default='./output',
                        help='Output directory for models and plots')
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--epochs',      type=int,   default=120)
    parser.add_argument('--hidden-dim',  type=int,   default=128)
    parser.add_argument('--heads',       type=int,   default=8)
    parser.add_argument('--layers',      type=int,   default=3)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--patience',    type=int,   default=15)
    parser.add_argument('--num-workers', type=int,   default=4)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--input_mode',  type=str,
                        choices=['clusters', 'trackster', 'both'], default='both',
                        help='Input features to use')
    parser.add_argument('--amp',     action='store_true',
                        help='Automatic mixed precision')
    parser.add_argument('--compile', action='store_true',
                        help='torch.compile (PyTorch 2.0+)')
    parser.add_argument('--exclude-pid', type=int, default=None,
                        help='Exclude a specific PID from training data (e.g. 111 to remove pi0)')
    parser.add_argument('--resume',  '-r', type=str, default=None,
                        help='Checkpoint to resume from')
    parser.add_argument('--export',  '-e', type=str, default=None,
                        help='Export model to ONNX from checkpoint file (no training)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export-only mode
    if args.export:
        print("=" * 60)
        print("EXPORT MODE: Loading model from checkpoint and exporting to ONNX")
        print("=" * 60)
        
        device = torch.device(args.device)
        
        # Create model with same architecture
        model = SetTransformer(
            hidden_dim=args.hidden_dim,
            num_heads=args.heads,
            num_layers=args.layers,
            num_classes=num_classes,
            input_mode=args.input_mode,
        ).to(device)
        
        # Load checkpoint
        if os.path.isfile(args.export):
            print(f"Loading checkpoint from {args.export}")
            checkpoint = torch.load(args.export, map_location=device, weights_only=False)
            
            # Handle model state dict
            state_dict = checkpoint['model_state_dict']
            if hasattr(model, '_orig_mod'):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print(f"Model loaded successfully")
            
            # Export to ONNX
            export_to_onnx(model, args, device)
        else:
            print(f"Checkpoint {args.export} not found!")
            return
        
        return  # Exit after export
    
    # Regular training mode (rest of your existing code)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    # ---- Datasets ----
    print("=" * 60)
    print("Loading dataset splits")
    train_ds = TracksterDataset(args.input, split='train', seed=args.seed, exclude_pid=args.exclude_pid)
    val_ds   = TracksterDataset(args.input, split='val',   seed=args.seed)
    test_ds  = TracksterDataset(args.input, split='test',  seed=args.seed)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, collate_fn=collate_sets,
                          pin_memory=(device.type == 'cuda'),
                          persistent_workers=(args.num_workers > 0),
                          worker_init_fn=worker_init_fn,
                          prefetch_factor=2 if args.num_workers > 0 else None)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)
    test_loader  = make_loader(test_ds,  shuffle=False)

    # ---- Model ----
    model = SetTransformer(
        hidden_dim=args.hidden_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        num_classes=num_classes,
        input_mode=args.input_mode,
    ).to(device)

    if args.compile:
        print("Compiling with torch.compile")
        model = torch.compile(model)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input mode: {args.input_mode}  |  Outputs: 8 classes  |  Loss: EM vs HAD group")

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.6, patience=5, min_lr=1e-6)

    scaler    = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None
    criterion = GroupCollapsedLoss().to(device)

    # ---- Resume ----
    start_epoch      = 0
    best_val_gacc    = 0.0
    patience_counter = 0
    history = dict(train_loss=[], val_loss=[], train_gacc=[], val_gacc=[])

    if args.resume and os.path.isfile(args.resume):
        print(f"\nResuming from {args.resume}")
        ckpt  = torch.load(args.resume, map_location=device, weights_only=False)
        state = ckpt['model_state_dict']
        if hasattr(model, '_orig_mod'):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict']:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch      = ckpt.get('epoch', 0)
        best_val_gacc    = ckpt.get('best_val_gacc', 0.0)
        patience_counter = ckpt.get('patience_counter', 0)
        if 'history' in ckpt:
            history = ckpt['history']
        print(f"  Resumed at epoch {start_epoch}, best val group_acc={best_val_gacc:.4f}")
    elif args.resume:
        print(f"Checkpoint {args.resume} not found starting fresh.")

    # ---- Training loop ----
    print("\n" + "=" * 60)
    print(f"Training  |  input_mode={args.input_mode}  |  loss=GroupCollapsedLoss")
    print("=" * 60)
    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        train_loss, train_gacc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_gacc, _, _ = evaluate(
            model, val_loader, criterion, device, scaler)

        scheduler.step(val_gacc)
        current_lr = optimizer.param_groups[0]['lr']
        elapsed    = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_gacc'].append(train_gacc)
        history['val_gacc'].append(val_gacc)

        print(f"\nEpoch {epoch+1:3d}  ({elapsed:.1f}s)  LR={current_lr:.2e}")
        print(f"  Train  loss={train_loss:.4f}  group_acc={train_gacc:.4f}")
        print(f"  Val    loss={val_loss:.4f}  group_acc={val_gacc:.4f}")

        if val_gacc > best_val_gacc:
            best_val_gacc    = val_gacc
            patience_counter = 0
            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict':    scaler.state_dict() if scaler else None,
                'best_val_gacc':        best_val_gacc,
                'patience_counter':     patience_counter,
                'args':                 args,
                'history':              history,
            }, os.path.join(args.output_dir, 'best_set_transformer.pt'))
            print(f" Best model saved (group_acc={val_gacc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nTraining done in {(time.time()-start_time)/60:.1f} min")

    # ---- Test evaluation ----
    ckpt_path = os.path.join(args.output_dir, 'best_set_transformer.pt')
    if os.path.isfile(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt['model_state_dict']
        if hasattr(model, '_orig_mod'):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model.load_state_dict(state)

    test_loss, test_gacc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, scaler)

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Loss:           {test_loss:.4f}")
    print(f"  Group Accuracy: {test_gacc:.4f}")

    # EM vs HAD confusion from 8-class predictions
    def to_group(arr):
        g = np.full_like(arr, 2)
        for i in EM_CLASS_INDICES:
            g[arr == i] = 0
        for i in HAD_CLASS_INDICES:
            g[arr == i] = 1
        return g

    pred_groups = to_group(np.array(test_preds))
    label_groups = to_group(np.array(test_labels))
    mask = label_groups != 2   # only EM and HAD
    cm = confusion_matrix(label_groups[mask], pred_groups[mask], labels=[0, 1])

    if cm.sum() > 0:
        print(f"  EM  precision={cm[0,0]/max(cm[:,0].sum(),1):.4f}  "
              f"recall={cm[0,0]/max(cm[0].sum(),1):.4f}")
        print(f"  HAD precision={cm[1,1]/max(cm[:,1].sum(),1):.4f}  "
              f"recall={cm[1,1]/max(cm[1].sum(),1):.4f}")
    print("=" * 60)

    # ---- Plots with percentages ----
    plt.figure(figsize=(6, 5))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    # Convert to percentages
    annot = np.empty_like(cm).astype(str)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f'{cm_percent[i, j]:.1f}%'

    sns.heatmap(cm_percent, annot=annot, fmt='', cmap='Blues',
                xticklabels=['EM', 'HAD'], yticklabels=['EM', 'HAD'],
                vmin=0, vmax=100, cbar_kws={'label': 'Percentage (%)'})
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'EM vs HAD Confusion  group_acc={test_gacc:.4f}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_em_had.png'), dpi=150)
    plt.close()

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Loss'); axes[0].legend()
    axes[1].plot(history['train_gacc'], label='Train')
    axes[1].plot(history['val_gacc'],   label='Val')
    axes[1].set_title('EM vs HAD Group Accuracy'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Plots saved to {args.output_dir}/")

    # ---- Export to ONNX after training ----
    export_to_onnx(model, args, device)


if __name__ == "__main__":
    main()
