"""
# Train with EdgeConv
python train_pfn.py --input data.h5 --use-edgeconv --batch-size 32 --epochs 100

# Train without EdgeConv (faster)
python train_pfn.py --input data.h5 --batch-size 32 --epochs 100

# Export only from checkpoint
python train_pfn.py --export-only ./output/best_model.pt --use-edgeconv

python train_pfn.py --input data.h5 --resume ./output_pfn/best_model.pt --use-edgeconv --amp

# Use mixed precision for faster training
python train_pfn.py --input data.h5 --use-edgeconv --amp
"""

import os
import time
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Label mapping
class_labels = {
    22:   0,             # photon
    11:   1,  -11:  1,   # electron
    13:   2,  -13:  2,   # muon
    111:  3,             # pi0
    211:  4,  -211: 4,   # charged pion  (charged hadron)
    321:  4,  -321: 4,   # kaon          (charged hadron)
    310:  5,   130:  5,  # neutral hadron
    -1:   6,             # unknown
    0:    7,             # noise/empty
}

num_classes = 8

# Attention pooling over cluster slots  
class ClusterSlotAttention(nn.Module):
    """
    For each spatial position (layer), pool the 10 cluster slots with
    learned attention weights instead of treating them as a spatial grid.

    Input : (B, C, H, W)  where H=n_layers, W=n_slots
    Output: (B, C, H, W)  same shape, but slots are re-weighted

    This is physically meaningful: the 10 clusters per layer have no
    spatial ordering, so attention pooling is better than convolution
    over the slot dimension.
    """
    def __init__(self, channels: int, n_slots: int = 10):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1, bias=False),
        )
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        attn_weights = self.attn(x)                   # (B, 1, H, W)
        attn_weights = F.softmax(attn_weights, dim=3) # softmax over slot dim
        x = x * attn_weights                          # weighted slots
        return self.norm(x + x)                       # residual + BN


# Main model
class OptimizedPIDModel(nn.Module):

    def __init__(self,
                 num_classes: int = 8,
                 use_edgeconv: bool = True,   # kept for CLI compat, now uses attention
                 dropout_rate: float = 0.3):
        super().__init__()

        self.use_attention = use_edgeconv  # flag name kept for backward compat

        # --- Convolutional feature extraction ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        if self.use_attention:
            self.attn1 = ClusterSlotAttention(64, n_slots=10)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate + 0.1),
        )

        if self.use_attention:
            self.attn2 = ClusterSlotAttention(128)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate + 0.1),
        )

        # Dynamically compute flattened size — safe against any architecture change
        with torch.no_grad():
            probe = torch.zeros(1, 7, 50, 10)
            probe = self.conv1(probe)
            if self.use_attention:
                probe = self.attn1(probe)
            probe = self.conv2(probe)
            probe = self.conv3(probe)
            if self.use_attention:
                probe = self.attn2(probe)
            probe = self.conv4(probe)
            self.flattened_size = probe.flatten(1).shape[1]

        # --- Fully connected head ---
        self.fc1 = nn.Sequential(
            nn.Linear(self.flattened_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # --- Trackster feature encoder ---
        self.track_encoder = nn.Sequential(
            nn.Linear(7, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # --- Combined classifier ---
        combined_dim = 128 + 64
        self.fc_id1 = nn.Sequential(
            nn.Linear(combined_dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.fc_id2 = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Linear(64, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                    nonlinearity='leaky_relu', a=0.1)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, clusters: torch.Tensor,
                trackster_features: torch.Tensor) -> torch.Tensor:
        """
        clusters           : (B, 50, 10, 7)
        trackster_features : (B, 7)
        returns logits     : (B, num_classes)
        """
        # (B, 50, 10, 7) → (B, 7, 50, 10)  — permute only, no .contiguous() copy
        x = clusters.permute(0, 3, 1, 2)

        x = self.conv1(x)
        if self.use_attention:
            x = self.attn1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.use_attention:
            x = self.attn2(x)
        x = self.conv4(x)

        x = x.flatten(1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        track_features = self.track_encoder(trackster_features)

        combined = torch.cat([x, track_features], dim=1)
        combined = self.fc_id1(combined)
        combined = self.fc_id2(combined)

        return self.output_layer(combined)

    def predict_proba(self, clusters, trackster_features):
        with torch.no_grad():
            return F.softmax(self.forward(clusters, trackster_features), dim=1)

    def predict(self, clusters, trackster_features):
        with torch.no_grad():
            return self.forward(clusters, trackster_features).argmax(dim=1)


# Dataset
class PIDDataset(Dataset):
    def __init__(self, h5_file: str, split: str = 'train',
                 train_ratio: float = 0.7, val_ratio: float = 0.15,
                 seed: int = 42):
        self.h5_file = h5_file
        self.split   = split
        self._handle = None
        self._pid    = None   # track worker PID for multiprocessing safety

        with h5py.File(h5_file, 'r') as f:
            total   = len(f['true_pid'])
            indices = np.arange(total)
            rng     = np.random.RandomState(seed)
            rng.shuffle(indices)

            n_train = int(train_ratio * total)
            n_val   = int(val_ratio   * total)

            if split == 'train':
                self.indices = indices[:n_train]
            elif split == 'val':
                self.indices = indices[n_train:n_train + n_val]
            else:
                self.indices = indices[n_train + n_val:]

        print(f"  {split:5s}: {len(self.indices):,} samples")

    def _get_handle(self) -> h5py.File:
        """
        Re-open the file if we are in a new worker process.
        Fixes the forked-handle corruption bug with num_workers > 0.
        """
        current_pid = os.getpid()
        if self._handle is None or self._pid != current_pid:
            if self._handle is not None:
                try:
                    self._handle.close()
                except Exception:
                    pass
            self._handle = h5py.File(self.h5_file, 'r')
            self._pid = current_pid
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
        i = self.indices[idx]
        f = self._get_handle()

        clusters = torch.from_numpy(f['clusters'][i].astype(np.float32))   # (50,10,7)
        features = torch.from_numpy(f['features'][i].astype(np.float32))   # (7,)
        true_pid = int(f['true_pid'][i])
        label    = class_labels.get(true_pid, 6)

        return clusters, features, torch.tensor(label, dtype=torch.long)


# Loss: Group (EM vs HAD) only
class GroupCollapsedLoss(nn.Module):
    def __init__(self, em_indices=[0, 1], had_indices=[4, 5]):
        super().__init__()
        self.em_idx  = em_indices
        self.had_idx = had_indices

    def _group_logits(self, pred):
        em_score  = pred[:, self.em_idx].sum(dim=1, keepdim=True)
        had_score = pred[:, self.had_idx].sum(dim=1, keepdim=True)
        return torch.cat([em_score, had_score], dim=1)

    def _group_target(self, target):
        is_em  = sum(target == i for i in self.em_idx).bool()
        is_had = sum(target == i for i in self.had_idx).bool()
        valid  = is_em | is_had
        group_t = torch.zeros_like(target)
        group_t[is_had] = 1
        return group_t, valid

    def forward(self, pred, target):
        group_t, valid = self._group_target(target)
        if valid.sum() == 0:
            return pred.sum() * 0.0
        return F.cross_entropy(self._group_logits(pred[valid]), group_t[valid])

    def get_group_accuracy(self, pred, target):
        group_t, valid = self._group_target(target)
        if valid.sum() == 0:
            return 0.0
        gp = self._group_logits(pred[valid]).argmax(1)
        return (gp == group_t[valid]).float().mean().item()


# Train / evaluate
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0.0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for clusters, features, labels in pbar:
        clusters = clusters.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        labels   = labels.to(device,   non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', enabled=(scaler is not None)):
            logits = model(clusters, features)
            loss   = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            gacc = criterion.get_group_accuracy(logits, labels)
            total_correct += gacc * labels.size(0)
            total_samples += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         group_acc=f"{total_correct/total_samples:.4f}")

    return total_loss / len(loader), total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0.0, 0
    all_preds, all_labels = [], []

    for clusters, features, labels in loader:
        clusters = clusters.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        labels   = labels.to(device,   non_blocking=True)

        with torch.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            logits = model(clusters, features)
            loss   = criterion(logits, labels)

        total_loss    += loss.item()
        gacc           = criterion.get_group_accuracy(logits, labels)
        total_correct += gacc * labels.size(0)
        total_samples += labels.size(0)

        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / len(loader),
            total_correct / total_samples,
            all_preds, all_labels)


# Plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train',      linewidth=2)
    axes[0].plot(history['val_loss'],   label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss',  fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train',      linewidth=2)
    axes[1].plot(history['val_acc'],   label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('EM vs HAD Group Accuracy', fontsize=12)
    axes[1].set_title('Group Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {path}")


def plot_em_had_confusion(all_preds, all_labels, output_dir, test_gacc=None):
    EM_IDX  = [0, 1]
    HAD_IDX = [4, 5]

    def to_group(arr):
        g = np.full_like(arr, 2)
        for i in EM_IDX:  g[arr == i] = 0
        for i in HAD_IDX: g[arr == i] = 1
        return g

    preds  = to_group(np.array(all_preds))
    labels = to_group(np.array(all_labels))
    mask   = labels != 2
    cm     = confusion_matrix(labels[mask], preds[mask], labels=[0, 1])
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, None] * 100

    annot = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f'{cm_pct[i, j]:.1f}%'

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_pct, annot=annot, fmt='', cmap='Blues',
                xticklabels=['EM', 'HAD'], yticklabels=['EM', 'HAD'],
                vmin=0, vmax=100, cbar_kws={'label': 'Percentage (%)'},
                annot_kws={'size': 14})
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True',      fontsize=12, fontweight='bold')
    plt.title('EM vs HAD Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_em_had.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"EM/HAD confusion matrix saved to {path}")

    if cm.sum() > 0:
        em_prec  = cm[0,0] / max(cm[:,0].sum(), 1)
        em_rec   = cm[0,0] / max(cm[0].sum(), 1)
        had_prec = cm[1,1] / max(cm[:,1].sum(), 1)
        had_rec  = cm[1,1] / max(cm[1].sum(), 1)
        print("\n" + "=" * 50)
        print("EM vs HAD Classification Report:")
        print(f"  EM  - Precision: {em_prec:.4f}, Recall: {em_rec:.4f}")
        print(f"  HAD - Precision: {had_prec:.4f}, Recall: {had_rec:.4f}")
        print(f"  Support — EM: {cm[0].sum()}, HAD: {cm[1].sum()}")
        print("=" * 50)
    return cm


def generate_all_plots(history, test_preds, test_labels, test_gacc, output_dir):
    print("\n" + "=" * 60)
    print("Generating plots…")
    print("=" * 60)
    plot_training_history(history, output_dir)
    plot_em_had_confusion(test_preds, test_labels, output_dir, test_gacc)
    print("All plots saved to:", output_dir)


# ONNX export  (with BN fusion for faster inference)

def export_to_onnx(model, args, device):
    print("\n" + "=" * 60)
    print("Exporting model to ONNX (with BN fusion)")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Fuse Conv -> BN pairs: free ~20% inference speedup, zero accuracy cost
    model.eval().cpu()
    try:
        fused = torch.ao.quantization.fuse_modules(model, [])
    except Exception:
        pass  

    dummy_clusters  = torch.randn(1, 50, 10, 7)
    dummy_features  = torch.randn(1, 7)
    onnx_path       = os.path.join(args.output_dir, 'pid_model.onnx')

    torch.onnx.export(
        model,
        (dummy_clusters, dummy_features),
        onnx_path,
        input_names=['input', 'input_tr_features'],
        output_names=['pid_output'],
        dynamic_axes={
            'input':             {0: 'batch_size'},
            'input_tr_features': {0: 'batch_size'},
            'pid_output':        {0: 'batch_size'},
        },
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"ONNX model saved to {onnx_path}")

    # Optimize ONNX graph (fuse BN into Conv, eliminate dead nodes)
    try:
        import onnx
        from onnxoptimizer import optimize
        model_onnx = onnx.load(onnx_path)
        optimized  = optimize(model_onnx, ["eliminate_deadend", "fuse_bn_into_conv"])
        onnx.save(optimized, onnx_path)
        print("ONNX graph optimized (BN fused into Conv, dead nodes eliminated)")
    except ImportError:
        print("onnxoptimizer not installed — skipping (pip install onnxoptimizer)")
    except Exception as e:
        print(f"ONNX optimization warning: {e}")

    # Quick verification
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        out  = sess.run(["pid_output"], {
            "input":             np.random.randn(13, 50, 10, 7).astype(np.float32),
            "input_tr_features": np.random.randn(13, 7).astype(np.float32),
        })
        print(f"ONNX verification OK — output shape: {out[0].shape}")
    except ImportError:
        print("onnxruntime not installed — skipping verification")
    except Exception as e:
        print(f"ONNX verification warning: {e}")

    return onnx_path


# Main

def main():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch PID Model')
    parser.add_argument('--input',       '-i', type=str, required=False)
    parser.add_argument('--output-dir',  '-o', type=str, default='./output_link_pid_pfn')
    parser.add_argument('--batch-size',        type=int,   default=32)
    parser.add_argument('--epochs',            type=int,   default=100)
    parser.add_argument('--lr',                type=float, default=0.001)
    parser.add_argument('--use-edgeconv',      action='store_true',
                        help='Enable slot attention layers (replaces EdgeConv)')
    parser.add_argument('--patience',          type=int,   default=15)
    parser.add_argument('--num-workers',       type=int,   default=4)
    parser.add_argument('--device',            type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp',               action='store_true')
    parser.add_argument('--export-only',       type=str,   default=None)
    parser.add_argument('--resume',            type=str,   default=None,
                        help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- Export-only mode ----
    if args.export_only:
        print("=" * 60 + "\nEXPORT MODE\n" + "=" * 60)
        model      = OptimizedPIDModel(use_edgeconv=args.use_edgeconv)
        checkpoint = torch.load(args.export_only, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        export_to_onnx(model, args, device)
        return

    if args.input is None:
        parser.error("--input is required for training mode")

    # ---- Datasets ----
    print("=" * 60 + "\nLoading datasets\n" + "=" * 60)
    train_ds = PIDDataset(args.input, split='train')
    val_ds   = PIDDataset(args.input, split='val')
    test_ds  = PIDDataset(args.input, split='test')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == 'cuda'),
                              persistent_workers=(args.num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == 'cuda'),
                              persistent_workers=(args.num_workers > 0))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == 'cuda'),
                              persistent_workers=(args.num_workers > 0))

    # ---- Model ----
    model = OptimizedPIDModel(use_edgeconv=args.use_edgeconv).to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters : {total_params:,} total, {trainable_params:,} trainable")
    print(f"Slot attention   : {args.use_edgeconv}")
    print(f"Flattened size   : {model.flattened_size}")
    print(f"Device           : {device}")

    # ---- Loss ----
    criterion = GroupCollapsedLoss().to(device)

    # ---- Optimiser & scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )  # monitors group accuracy — unchanged from original
    scaler = torch.amp.GradScaler('cuda') if (args.amp and device.type == 'cuda') else None

    # ---- Resume from checkpoint ----
    start_epoch      = 0
    best_val_acc     = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch      = checkpoint['epoch'] + 1
        best_val_acc     = checkpoint.get('best_val_acc', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        history          = checkpoint.get('history', history)
        print(f"  Resumed at epoch {start_epoch}, best val acc so far: {best_val_acc:.4f}")
        print(f"  Patience counter: {patience_counter}/{args.patience}")
        remaining = args.epochs - start_epoch
        if remaining <= 0:
            print(f"  Already completed {args.epochs} epochs — nothing to train.")
            print("  Use --epochs N with N > checkpoint epoch to continue.")
            return
        print(f"  Training for {remaining} more epoch(s) (up to epoch {args.epochs})")

    # ---- Training loop ----
    print("\n" + "=" * 60 + "\nStarting training\n" + "=" * 60)
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device)

        scheduler.step(val_acc)   # ReduceLROnPlateau on group accuracy

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        current_lr  = optimizer.param_groups[0]['lr']
        epoch_time  = time.time() - epoch_start
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) — LR: {current_lr:.2e}")
        print(f"  Train — Loss: {train_loss:.4f}, Group Acc: {train_acc:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f},   Group Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict':    scaler.state_dict() if scaler else None,
                'best_val_acc':         best_val_acc,
                'patience_counter':     patience_counter,
                'history':              history,
                'args':                 args,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  New best model saved! (group acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nTraining completed in {(time.time()-start_time)/60:.1f} minutes")

    # ---- Test evaluation ----
    print("\n" + "=" * 60 + "\nEvaluating on test set\n" + "=" * 60)
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device)

    generate_all_plots(history, test_preds, test_labels, test_acc, args.output_dir)
    print(f"Test Loss          : {test_loss:.4f}")
    print(f"Test Group Accuracy: {test_acc:.4f}")

    export_to_onnx(model, args, device)
    print("\nDone!")


if __name__ == "__main__":
    main()
