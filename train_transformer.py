"""
# Train the Set Transformer model
python train_transformer.py --input ticl_dataset.h5 --batch-size 32 --epochs 50

# Adjust hyperparameters
python train_transformer.py --input ticl_dataset.h5 --hidden-dim 128 --lr 0.0005 --heads 8

# With mixed precision (faster on GPU)
python train_transformer.py --input ticl_dataset.h5 --amp

# With torch.compile (PyTorch 2.0+, free ~30% speedup)
python train_transformer.py --input ticl_dataset.h5 --compile
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
    22: 0, 111: 0,# photon, pion0
    11: 1, -11: 1,# electron/positron
    13: 2, -13: 2,# muon
    1011: 3,
    211: 4, -211: 4,# charged hadron
    321: 4, -321: 4, # charged hadron
    310: 5, 130: 5, # neutral hadron
    -1: 6,# unknown
    0: 7# noise/empty
}

num_classes = 8


# 2. DATASET one HDF5 

class TracksterDataset(Dataset):
    """
    Reads tracksters from a flat-columnar HDF5 file.
    """

    def __init__(self, h5_file, split='train',
                 train_ratio=0.7, val_ratio=0.15,
                 seed=42):

        self.h5_file  = h5_file
        self.split    = split
        self._handle  = None   # per-process file handle, opened lazily

        with h5py.File(h5_file, 'r') as f:
            total = int(f.attrs['num_tracksters'])

            # Keep only tracksters that have at least one cluster
            num_clusters = f['num_clusters'][:]
            valid = np.where(num_clusters > 0)[0]

            print(f"Total tracksters: {total:,}  |  valid (>0 clusters): {len(valid):,}")

        # Reproducible shuffle then split
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

        print(f"  {split} split: {len(self.indices):,} tracksters")

    # HDF5 handle management: open once per worker, reuse for all samples

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

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i  = int(self.indices[idx])
        f  = self._get_handle()

        trackster    = torch.from_numpy(f['features'][i].astype(np.float32))
        true_pid     = int(f['true_pid'][i])
        label        = torch.tensor(class_labels.get(true_pid, 6), dtype=torch.long)
        n_clusters   = int(f['num_clusters'][i])
        clusters_raw = f['clusters'][i]           # flat float32 array of length N*4

        if n_clusters > 0:
            clusters = torch.from_numpy(
                clusters_raw.reshape(n_clusters, 4).astype(np.float32))
        else:
            clusters = torch.zeros((0, 4), dtype=torch.float32)

        return clusters, label, trackster


def worker_init_fn(worker_id):
    # Each worker has its own copy of the dataset object (forked).
    # Resetting _handle ensures the child process opens a fresh file descriptor
    # rather than sharing the parent's (which can cause corruption).
    import gc
    for obj in gc.get_objects():
        if isinstance(obj, TracksterDataset):
            obj._handle = None


def collate_sets(batch):
    cluster_sets = [item[0] for item in batch]
    labels       = torch.stack([item[1] for item in batch])
    tracksters   = torch.stack([item[2] for item in batch])
    return cluster_sets, labels, tracksters


# 3. SET TRANSFORMER fixed transformer blocks (proper Pre-LN)

class TransformerBlock(nn.Module):
    """
    Standard Pre-LayerNorm transformer block:
        x = x + Attn(LN(x))
        x = x + FF(LN(x))
    """

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn  = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        # Pre-LN attention
        x2, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=key_padding_mask)
        x = x + x2
        # Pre-LN feedforward
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
                 use_trackster_features=True):

        super().__init__()
        self.hidden_dim             = hidden_dim
        self.use_trackster_features = use_trackster_features

        # Per-cluster encoder
        self.encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Trackster-level encoder
        if use_trackster_features:
            self.trackster_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
            )
            pool_input_dim = hidden_dim * 2 + hidden_dim // 2
        else:
            pool_input_dim = hidden_dim * 2

        # Transformer blocks (now with proper Pre-LN)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

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
        device       = cluster_sets[0].device
        max_clusters = max(c.size(0) for c in cluster_sets)

        # Pad to uniform length within batch
        padded, masks = [], []
        for c in cluster_sets:
            n = c.size(0)
            if n < max_clusters:
                pad  = torch.zeros(max_clusters - n, c.size(1), device=device)
                c    = torch.cat([c, pad], dim=0)
                mask = torch.cat([torch.ones(n), torch.zeros(max_clusters - n)])
            else:
                mask = torch.ones(max_clusters)
            padded.append(c)
            masks.append(mask)

        x    = torch.stack(padded)                    # [B, N, 4]
        mask = torch.stack(masks).to(device)          # [B, N]

        # Encode clusters
        x = self.encoder(x)                           # [B, N, H]

        # Transformer blocks
        pad_mask = ~mask.bool()                       # True = ignore (HF convention)
        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)

        # Mean + max pooling over real clusters only
        exp_mask = mask.unsqueeze(-1)                 # [B, N, 1]
        x_masked = x * exp_mask
        x_mean   = x_masked.sum(1) / exp_mask.sum(1).clamp(min=1)
        x_max    = (x_masked + (1 - exp_mask) * -1e9).max(1)[0]
        x_pool   = torch.cat([x_mean, x_max], dim=1) # [B, H*2]

        # Optionally concatenate trackster features
        if self.use_trackster_features and trackster_features is not None:
            x_pool = torch.cat([x_pool, self.trackster_encoder(trackster_features)], dim=1)

        return self.classifier(x_pool)


# 4. HIERARCHICAL LOSS  
class HierarchicalLoss(nn.Module):

    def __init__(self,
                 em_classes=[0, 1],
                 had_classes=[4, 5],
                 within_group_penalty=0.5,
                 cross_group_penalty=5.0,
                 other_penalty=2.0):
        super().__init__()
        self.register_buffer('em_classes',  torch.tensor(em_classes))
        self.register_buffer('had_classes', torch.tensor(had_classes))
        self.within_group_penalty = within_group_penalty
        self.cross_group_penalty  = cross_group_penalty
        self.other_penalty        = other_penalty
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def _get_group(self, ids):
        g = torch.full_like(ids, 2)
        for c in self.em_classes:
            g[ids == c] = 0
        for c in self.had_classes:
            g[ids == c] = 1
        return g

    def forward(self, pred, target):
        base        = self.ce(pred, target)
        pred_cls    = pred.argmax(1)
        tg, pg      = self._get_group(target), self._get_group(pred_cls)
        same        = tg == pg
        cross_em_had = ((tg == 0) & (pg == 1)) | ((tg == 1) & (pg == 0))
        other       = ~(same | cross_em_had)
        penalties   = torch.ones_like(base)
        penalties[same]         = self.within_group_penalty
        penalties[cross_em_had] = self.cross_group_penalty
        penalties[other]        = self.other_penalty
        return (base * penalties).mean()

    def get_group_accuracy(self, pred, target):
        tg = self._get_group(target)
        pg = self._get_group(pred.argmax(1))
        return (tg == pg).float().mean().item()

    def get_em_vs_had_confusion(self, pred, target):
        pred_cls = pred.argmax(1) if pred.dim() > 1 else pred
        tg, pg   = self._get_group(target), self._get_group(pred_cls)
        mask     = tg != 2
        if mask.sum() == 0:
            return np.zeros((2, 2), dtype=int)
        return confusion_matrix(
            tg[mask].cpu().numpy(), pg[mask].cpu().numpy(), labels=[0, 1])


# 5. TRAINING mixed precision + gradient scaler

def train_epoch(model, loader, optimizer, criterion, device,
                scaler=None, grad_clip=1.0):
    model.train()
    total_loss, correct, total, group_correct = 0, 0, 0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for cluster_sets, labels, tracksters in pbar:
        if not cluster_sets:
            continue

        labels      = labels.to(device)
        tracksters  = tracksters.to(device)
        cluster_sets = [c.to(device) for c in cluster_sets]

        optimizer.zero_grad(set_to_none=True)

        # ---- forward (with optional AMP) ----
        with torch.autocast(device_type=device.type if hasattr(device, 'type')
                            else 'cuda' if 'cuda' in str(device) else 'cpu',
                            enabled=scaler is not None):
            out  = model(cluster_sets, tracksters)
            loss = criterion(out, labels)

        # ---- backward ----
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
        pred          = out.argmax(1)
        correct      += (pred == labels).sum().item()
        total        += labels.size(0)
        group_correct += criterion.get_group_accuracy(out, labels) * labels.size(0)

        pbar.set_postfix(loss=f'{loss.item():.4f}',
                         acc=f'{correct/max(total,1):.3f}')

    return total_loss / len(loader), correct / max(total, 1), group_correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, scaler=None):
    model.eval()
    total_loss, correct, total, group_correct = 0, 0, 0, 0
    all_preds, all_labels = [], []

    for cluster_sets, labels, tracksters in loader:
        labels      = labels.to(device)
        tracksters  = tracksters.to(device)
        cluster_sets = [c.to(device) for c in cluster_sets]

        with torch.autocast(device_type=device.type if hasattr(device, 'type')
                            else 'cuda' if 'cuda' in str(device) else 'cpu',
                            enabled=scaler is not None):
            out  = model(cluster_sets, tracksters)
            loss = criterion(out, labels)

        total_loss   += loss.item()
        pred          = out.argmax(1)
        correct      += (pred == labels).sum().item()
        total        += labels.size(0)
        group_correct += criterion.get_group_accuracy(out, labels) * labels.size(0)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / len(loader),
            correct / max(total, 1),
            group_correct / max(total, 1),
            all_preds, all_labels)


# 6. MAIN

def main():
    parser = argparse.ArgumentParser(description='Train Set Transformer on TICL cluster data')
    parser.add_argument('--input',      '-i', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, default='./output')
    parser.add_argument('--batch-size',   type=int,   default=32)
    parser.add_argument('--epochs',       type=int,   default=60)
    parser.add_argument('--hidden-dim',   type=int,   default=128)
    parser.add_argument('--heads',        type=int,   default=8)
    parser.add_argument('--layers',       type=int,   default=3)
    parser.add_argument('--lr',           type=float, default=0.001)
    parser.add_argument('--patience',     type=int,   default=15)
    parser.add_argument('--num-workers',  type=int,   default=4)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--use-trackster', action='store_true',
                        help='Use trackster-level features in model')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision (faster on GPU, ~2x memory saving)')
    parser.add_argument('--compile', action='store_true',
                        help='torch.compile the model (PyTorch 2.0+, free ~30% speedup)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    # ---- Datasets (each split gets its own dataset object) ----
    print("=" * 60)
    print("Loading dataset splits...")
    train_dataset = TracksterDataset(args.input, split='train', seed=args.seed)
    val_dataset   = TracksterDataset(args.input, split='val',   seed=args.seed)
    test_dataset  = TracksterDataset(args.input, split='test',  seed=args.seed)

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            collate_fn=collate_sets,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(args.num_workers > 0),
            worker_init_fn=worker_init_fn,   # reset HDF5 handle per worker
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader   = make_loader(val_dataset,   shuffle=False)
    test_loader  = make_loader(test_dataset,  shuffle=False)

    # ---- Model ----
    model = SetTransformer(
        hidden_dim=args.hidden_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        num_classes=num_classes,
        use_trackster_features=args.use_trackster,
    ).to(device)

    if args.compile:
        print("Compiling model with torch.compile …")
        model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {total_params:,}")

    # ---- Optimizer: cosine schedule with linear warmup ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Warmup for 5% of total steps, then cosine decay
    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.6, patience=5, min_lr=1e-6
    )
    # ---- AMP scaler ----
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None
    if scaler:
        print("Mixed precision (AMP) enabled.")

    criterion = HierarchicalLoss().to(device)

    # ---- Training loop ----
    best_val_group_acc = 0
    patience_counter   = 0
    history = dict(train_loss=[], val_loss=[],
                   train_acc=[], val_acc=[],
                   train_gacc=[], val_gacc=[])

    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)
    start_time = time.time()

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_acc, train_gacc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler)

        val_loss, val_acc, val_gacc, _, _ = evaluate(
            model, val_loader, criterion, device, scaler)

        current_lr = optimizer.param_groups[0]['lr']

        elapsed    = time.time() - t0

        scheduler.step(val_gacc)

        for k, v in zip(history, [train_loss, val_loss, train_acc, val_acc,
                                   train_gacc, val_gacc]):
            history[k].append(v)

        print(f"\nEpoch {epoch+1:3d}/{args.epochs}  ({elapsed:.1f}s)  LR={current_lr:.2e}")
        print(f"  Train  loss={train_loss:.4f}  class_acc={train_acc:.4f}  group_acc={train_gacc:.4f}")
        print(f"  Val    loss={val_loss:.4f}  class_acc={val_acc:.4f}  group_acc={val_gacc:.4f}")

        if val_gacc > best_val_group_acc:
            best_val_group_acc = val_gacc
            torch.save({
                'epoch':             epoch,
                'model_state_dict':  model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_group_acc':     val_gacc,
                'val_acc':           val_acc,
                'args':              args,
            }, os.path.join(args.output_dir, 'best_set_transformer.pt'))
            print(f"New best model saved (group_acc={val_gacc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")

    # ---- Test evaluation ----
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_set_transformer.pt'))
    # Handle compiled model (state dict keys may have _orig_mod prefix)
    state = checkpoint['model_state_dict']
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # Strip torch.compile prefix if needed
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model.load_state_dict(state)

    test_loss, test_acc, test_gacc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, scaler)

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Loss:           {test_loss:.4f}")
    print(f"  Class Accuracy: {test_acc:.4f}")
    print(f"  Group Accuracy: {test_gacc:.4f}")
    print("=" * 60)

    # ---- Plots ----
    # EM vs HAD confusion
    em_had_cm = criterion.get_em_vs_had_confusion(
        torch.tensor(test_preds), torch.tensor(test_labels))

    plt.figure(figsize=(8, 6))
    sns.heatmap(em_had_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['EM', 'HAD'], yticklabels=['EM', 'HAD'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('EM vs HAD Confusion Matrix', fontsize=14, fontweight='bold')
    if em_had_cm.sum() > 0:
        acc = (em_had_cm[0,0] + em_had_cm[1,1]) / em_had_cm.sum()
        plt.gca().text(0.5, -0.15, f'Accuracy: {acc:.4f}',
                       transform=plt.gca().transAxes, ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'em_vs_had_confusion.png'), dpi=150)
    plt.close()

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Loss'); axes[0].legend()
    axes[1].plot(history['train_acc'],  label='Train')
    axes[1].plot(history['val_acc'],    label='Val')
    axes[1].set_title('Class Accuracy'); axes[1].legend()
    axes[2].plot(history['train_gacc'], label='Train')
    axes[2].plot(history['val_gacc'],   label='Val')
    axes[2].set_title('Group Accuracy (EM/HAD/Other)'); axes[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Plots saved to {args.output_dir}/")

    # ---- ONNX export ----
    print("\nExporting to ONNX")

    class SetTransformerONNX(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, mask, trackster_features=None):
            # x: [B, N, 4], mask: [B, N]  (1=real, 0=pad)
            x = self.m.encoder(x)
            pad_mask = ~mask.bool()
            for block in self.m.blocks:
                x = block(x, key_padding_mask=pad_mask)
            exp = mask.unsqueeze(-1)
            x_mean = (x * exp).sum(1) / exp.sum(1).clamp(min=1)
            x_max  = (x * exp + (1 - exp) * -1e9).max(1)[0]
            x_pool = torch.cat([x_mean, x_max], 1)
            if self.m.use_trackster_features and trackster_features is not None:
                x_pool = torch.cat([x_pool, self.m.trackster_encoder(trackster_features)], 1)
            return self.m.classifier(x_pool)

    # Unwrap compiled model for ONNX
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    onnx_model = SetTransformerONNX(raw_model).eval().cpu()

    dummy_x    = torch.randn(1, 200, 4)
    dummy_mask = torch.ones(1, 200)
    dummy_ts   = torch.randn(1, 3) if args.use_trackster else None

    dynamic = {'clusters': {0: 'batch'}, 'mask': {0: 'batch'}, 'output': {0: 'batch'}}
    inputs  = ['clusters', 'mask'] + (['trackster_features'] if args.use_trackster else [])
    args_onnx = (dummy_x, dummy_mask, dummy_ts) if args.use_trackster else (dummy_x, dummy_mask)

    torch.onnx.export(
        onnx_model, args_onnx,
        os.path.join(args.output_dir, 'set_transformer.onnx'),
        input_names=inputs, output_names=['output'],
        dynamic_axes=dynamic, opset_version=14)
    print(f"ONNX model saved to {args.output_dir}/set_transformer.onnx")


if __name__ == "__main__":
    main()
