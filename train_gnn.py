"""
# Train the model
python train_gnn.py --input ticl_simplified.h5 --batch-size 32 --epochs 50 --model gnn

# Try different architectures
python train_gnn.py --input ticl_simplified.h5 --model sage
python train_gnn.py --input ticl_simplified.h5 --model gat

# Adjust hyperparameters
python train_gnn.py --input ticl_simplified.h5 --hidden-dim 128 --lr 0.0005 --heads 4
python train_gnn.py --input ticl_simplified.h5 --input-mode clusters --group-loss
python train_gnn.py --input ticl_simplified.h5 --amp --compile
python train_gnn.py --input ticl_simplified.h5 --resume ./output_gnn/best_gnn_both.pt
python train_gnn.py --input ticl_simplified.h5 --export ./output_gnn/best_gnn_both.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import numpy as np
import h5py
from tqdm import tqdm
import argparse
import random
import os
import time
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# ONNX-traceable pooling helpers
# Replaces global_mean_pool / global_max_pool which use PyG scatter and
# cannot be traced by the legacy TorchScript ONNX exporter.
# Strategy: pad node features to a dense [B, max_nodes, H] tensor using the
# batch index vector, then mean/max over the node dimension with masking.
# This uses only standard tensor ops that ONNX handles correctly.
# ---------------------------------------------------------------------------

def _dense_pool(x, batch):
    """
    Returns (x_mean [B, H], x_max [B, H]) using dense padding — ONNX-safe.
    x:     [N_total, H]
    batch: [N_total]  integer graph index for each node
    """
    B      = int(batch.max().item()) + 1
    H      = x.size(1)
    counts = batch.bincount(minlength=B)          # [B]
    max_n  = int(counts.max().item())

    # Build dense tensor [B, max_n, H] and a boolean mask [B, max_n]
    dense = x.new_zeros(B, max_n, H)
    mask  = x.new_zeros(B, max_n, dtype=torch.bool)

    # Scatter nodes into their per-graph slot
    # slot[i] = how many nodes graph batch[i] has seen so far
    slot = x.new_zeros(B, dtype=torch.long)
    for node_idx in range(x.size(0)):
        g = int(batch[node_idx].item())
        s = int(slot[g].item())
        dense[g, s] = x[node_idx]
        mask[g, s]  = True
        slot[g]    += 1

    # Mean pool: sum over valid nodes / count
    float_mask = mask.float().unsqueeze(-1)           # [B, max_n, 1]
    x_mean = (dense * float_mask).sum(1) / float_mask.sum(1).clamp(min=1)

    # Max pool: set padding to -inf before taking max
    neg_inf = torch.full_like(dense, float('-inf'))
    x_max   = torch.where(mask.unsqueeze(-1), dense, neg_inf).max(1)[0]

    return x_mean, x_max


def _dense_pool_onnx(x, batch, B, max_n):
    """
    ONNX-export version: B and max_n must be static integers known at trace time.
    Called only from the ONNX wrapper with fixed dummy sizes.
    """
    H     = x.size(1)
    dense = x.new_zeros(B, max_n, H)
    mask  = x.new_zeros(B, max_n, dtype=torch.bool)
    slot  = x.new_zeros(B, dtype=torch.long)
    for node_idx in range(x.size(0)):
        g = int(batch[node_idx].item())
        s = int(slot[g].item())
        dense[g, s] = x[node_idx]
        mask[g, s]  = True
        slot[g]    += 1
    float_mask = mask.float().unsqueeze(-1)
    x_mean = (dense * float_mask).sum(1) / float_mask.sum(1).clamp(min=1)
    neg_inf = torch.full_like(dense, float('-inf'))
    x_max   = torch.where(mask.unsqueeze(-1), dense, neg_inf).max(1)[0]
    return x_mean, x_max


# 1. LABEL MAPPING

class_labels = {
    22:   0,              # photon
    11:   1,  -11:  1,   # electron/positron
    13:   2,  -13:  2,   # muon
    111:  3,              # pion0
    211:  4,  -211: 4,
    321:  4,  -321: 4,   # charged hadron
    310:  5,  130:  5,   # neutral hadron
    -1:   6,              # unknown
    0:    7,              # noise/empty
}

num_classes       = 8
EM_CLASS_INDICES  = [0, 1]
HAD_CLASS_INDICES = [4, 5]


# 2. DATASET
class TracksterGraphDataset(torch.utils.data.Dataset):
    """
    Builds graphs from HDF5 file.
    """

    def __init__(self, h5_file, split='train', train_ratio=0.7, val_ratio=0.15,
                 seed=42, exclude_pid=None):

        self.h5_file = h5_file
        self.split   = split
        self._handle = None

        with h5py.File(h5_file, 'r') as f:
            total        = int(f.attrs['num_tracksters'])
            num_clusters = f['num_clusters'][:] if 'num_clusters' in f else np.ones(total)
            true_pids    = f['true_pid'][:]     if 'true_pid'    in f else None

        valid = np.where(num_clusters > 0)[0]

        if exclude_pid is not None and split == 'train' and true_pids is not None:
            pid_mask = np.abs(true_pids[valid]) != abs(exclude_pid)
            valid    = valid[pid_mask]
            print(f"Excluded PID {exclude_pid}. Remaining: {len(valid):,}")

        if true_pids is not None:
            pids_v    = np.abs(true_pids[valid])
            em_pids   = {22, 111, 11}
            had_pids  = {211, 321, 310, 130}
            em_c      = int(np.sum([p in em_pids  for p in pids_v]))
            had_c     = int(np.sum([p in had_pids for p in pids_v]))
            print(f"Total: {total:,}  |  valid: {len(valid):,}  "
                  f"(EM={em_c:,}, HAD={had_c:,}, other={len(valid)-em_c-had_c:,})")

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

        print(f"  {split}: {len(self.indices):,} tracksters")

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

        # Clusters
        raw = f['clusters'][i] if 'clusters' in f else np.array([])
        if len(raw) > 0 and raw.ndim == 1:
            raw = raw.reshape(-1, 4)
        if len(raw) == 0:
            return None

        # Trackster features stored as [1, 3] so PyG cat [B, 3]
        if 'features' in f:
            ts_feat = torch.from_numpy(
                f['features'][i].astype(np.float32)).unsqueeze(0)   # [1, 3]
        else:
            ts_feat = torch.zeros(1, 3, dtype=torch.float32)

        true_pid    = int(f['true_pid'][i])    if 'true_pid'    in f else -1
        true_energy = float(f['true_energy'][i]) if 'true_energy' in f else 0.0

        x          = torch.tensor(raw, dtype=torch.float)
        edge_index = torch.tensor(self._build_edges(raw), dtype=torch.long)
        y          = torch.tensor([class_labels.get(true_pid, 6)], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.trackster_features = ts_feat                           # [1, 3]
        data.true_energy        = torch.tensor([true_energy], dtype=torch.float)
        return data

    @staticmethod
    def _build_edges(clusters):
        """Within-layer + consecutive-layer fully-connected edges."""
        if len(clusters) == 0:
            return np.empty((2, 0), dtype=np.int64)

        layers       = clusters[:, 3].astype(int)
        unique_layers = np.unique(layers)
        node_idx     = np.arange(len(clusters))
        edges        = []

        for k, layer in enumerate(unique_layers):
            in_layer = node_idx[layers == layer]
            # within-layer
            for u in in_layer:
                for v in in_layer:
                    if u != v:
                        edges.append([u, v])
            # to next layer
            if k < len(unique_layers) - 1:
                in_next = node_idx[layers == unique_layers[k + 1]]
                for u in in_layer:
                    for v in in_next:
                        edges.append([u, v])

        if edges:
            return np.array(edges, dtype=np.int64).T
        return np.empty((2, 0), dtype=np.int64)


def worker_init_fn(worker_id):
    import gc
    for obj in gc.get_objects():
        if isinstance(obj, TracksterGraphDataset):
            obj._handle = None


def collate_graphs(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return Batch.from_data_list(batch)


# 3. GROUP-COLLAPSED LOSS  

class GroupCollapsedLoss(nn.Module):

    def __init__(self, em_indices=EM_CLASS_INDICES, had_indices=HAD_CLASS_INDICES):
        super().__init__()
        self.em_idx  = em_indices
        self.had_idx = had_indices
        self.ce      = nn.CrossEntropyLoss()

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
        return self.ce(self._group_logits(pred[valid]), group_t[valid])

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
        return confusion_matrix(group_t[valid].cpu().numpy(),
                                gp.cpu().numpy(), labels=[0, 1])


# 4. MODELS

class EnhancedGNN(nn.Module):

    def __init__(self, node_features=4, hidden_dim=128, num_layers=3, num_heads=4,
                 num_classes=8, dropout=0.1, input_mode='both',
                 use_layer_norm=True, use_residual=True):
        super().__init__()
        self.input_mode  = input_mode
        self.use_residual = use_residual

        if input_mode in ('clusters', 'both'):
            self.node_encoder = nn.Sequential(
                nn.Linear(node_features, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.GELU(), nn.Dropout(dropout))
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(
                    GATConv(hidden_dim, hidden_dim // num_heads,
                            heads=num_heads, concat=True, dropout=dropout)
                    if num_heads > 1 else GCNConv(hidden_dim, hidden_dim))
                self.norms.append(nn.LayerNorm(hidden_dim))
            cluster_out = hidden_dim * 2
        else:
            self.node_encoder = None
            self.convs = None
            self.norms = None
            cluster_out = 0

        if input_mode in ('trackster', 'both'):
            self.trackster_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 2))
            ts_out = hidden_dim // 2
        else:
            self.trackster_encoder = None
            ts_out = 0

        self.dropout    = nn.Dropout(dropout)
        pool_dim        = cluster_out + ts_out
        self.classifier = nn.Sequential(
            nn.LayerNorm(pool_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(pool_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # trackster_features: [B, 3]  (PyG cat of per-graph [1,3])
        ts = data.trackster_features if hasattr(data, 'trackster_features') else None
        parts = []

        if self.node_encoder is not None:
            x = self.node_encoder(x)
            for conv, norm in zip(self.convs, self.norms):
                x_new = self.dropout(nn.GELU()(norm(conv(x, edge_index))))
                x = x + x_new if self.use_residual else x_new
            _mean, _max = _dense_pool(x, batch)
            parts.append(torch.cat([_mean, _max], dim=1))

        if self.trackster_encoder is not None and ts is not None:
            parts.append(self.trackster_encoder(ts))

        return self.classifier(torch.cat(parts, dim=1))


class EnhancedGraphSAGE(nn.Module):

    def __init__(self, node_features=4, hidden_dim=128, num_layers=3,
                 num_classes=8, dropout=0.1, input_mode='both', use_layer_norm=True):
        super().__init__()
        self.input_mode = input_mode

        if input_mode in ('clusters', 'both'):
            self.node_encoder = nn.Linear(node_features, hidden_dim)
            self.convs = nn.ModuleList(
                [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
            ) if use_layer_norm else None
            cluster_out = hidden_dim * 2
        else:
            self.node_encoder = None
            self.convs = None
            self.norms = None
            cluster_out = 0

        if input_mode in ('trackster', 'both'):
            self.trackster_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
                nn.ReLU(), nn.Linear(hidden_dim // 2, hidden_dim // 2))
            ts_out = hidden_dim // 2
        else:
            self.trackster_encoder = None
            ts_out = 0

        self.dropout    = nn.Dropout(dropout)
        pool_dim        = cluster_out + ts_out
        self.classifier = nn.Sequential(
            nn.LayerNorm(pool_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(pool_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        ts = data.trackster_features if hasattr(data, 'trackster_features') else None
        parts = []

        if self.node_encoder is not None:
            x = F.relu(self.node_encoder(x))
            for i, conv in enumerate(self.convs):
                x_new = conv(x, edge_index)
                if self.norms:
                    x_new = self.norms[i](x_new)
                x_new = self.dropout(F.relu(x_new))
                x = x + x_new   # residual
            _mean, _max = _dense_pool(x, batch)
            parts.append(torch.cat([_mean, _max], dim=1))

        if self.trackster_encoder is not None and ts is not None:
            parts.append(self.trackster_encoder(ts))

        return self.classifier(torch.cat(parts, dim=1))


class EnhancedGAT(nn.Module):

    def __init__(self, node_features=4, hidden_dim=128, num_layers=3,
                 num_heads=4, num_classes=8, dropout=0.1, input_mode='both'):
        super().__init__()
        self.input_mode = input_mode

        if input_mode in ('clusters', 'both'):
            self.node_encoder = nn.Linear(node_features, hidden_dim)
            self.convs = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim // num_heads,
                        heads=num_heads, dropout=dropout)
                for _ in range(num_layers)])
            cluster_out = hidden_dim * 2
        else:
            self.node_encoder = None
            self.convs = None
            cluster_out = 0

        if input_mode in ('trackster', 'both'):
            self.trackster_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(), nn.Linear(hidden_dim // 2, hidden_dim // 2))
            ts_out = hidden_dim // 2
        else:
            self.trackster_encoder = None
            ts_out = 0

        self.dropout    = nn.Dropout(dropout)
        pool_dim        = cluster_out + ts_out
        self.classifier = nn.Sequential(
            nn.LayerNorm(pool_dim), nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        ts = data.trackster_features if hasattr(data, 'trackster_features') else None
        parts = []

        if self.node_encoder is not None:
            x = F.relu(self.node_encoder(x))
            for conv in self.convs:
                x = self.dropout(F.relu(conv(x, edge_index)))
            _mean, _max = _dense_pool(x, batch)
            parts.append(torch.cat([_mean, _max], dim=1))

        if self.trackster_encoder is not None and ts is not None:
            parts.append(self.trackster_encoder(ts))

        return self.classifier(torch.cat(parts, dim=1))


def build_model(args):
    kwargs = dict(hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                  num_classes=num_classes, dropout=args.dropout,
                  input_mode=args.input_mode)
    if args.model == 'gnn':
        return EnhancedGNN(num_heads=args.num_heads, **kwargs)
    elif args.model == 'sage':
        return EnhancedGraphSAGE(**kwargs)
    elif args.model == 'gat':
        return EnhancedGAT(num_heads=args.num_heads, **kwargs)
    raise ValueError(f"Unknown model: {args.model}")


# 5. TRAINING
def train_epoch(model, loader, optimizer, criterion, device, scaler=None, grad_clip=1.0):
    model.train()
    total_loss, total, metric_sum = 0, 0, 0
    use_group = isinstance(criterion, GroupCollapsedLoss)

    pbar = tqdm(loader, desc="Training", leave=False)
    for data in pbar:
        if data is None:
            continue
        data = data.to(device)
        labels = data.y.squeeze()

        optimizer.zero_grad(set_to_none=True)
        amp_device = 'cuda' if 'cuda' in str(device) else 'cpu'
        with torch.autocast(device_type=amp_device, enabled=scaler is not None):
            out  = model(data)
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

        total_loss += loss.item()
        n = labels.size(0)
        total += n
        if use_group:
            metric_sum += criterion.get_group_accuracy(out, labels) * n
        else:
            metric_sum += (out.argmax(1) == labels).sum().item()

        pbar.set_postfix(loss=f'{loss.item():.4f}',
                         metric=f'{metric_sum/max(total,1):.3f}')

    return total_loss / len(loader), metric_sum / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, scaler=None):
    model.eval()
    total_loss, total, metric_sum = 0, 0, 0
    all_preds, all_labels = [], []
    use_group = isinstance(criterion, GroupCollapsedLoss)

    for data in loader:
        if data is None:
            continue
        data   = data.to(device)
        labels = data.y.squeeze()

        amp_device = 'cuda' if 'cuda' in str(device) else 'cpu'
        with torch.autocast(device_type=amp_device, enabled=scaler is not None):
            out  = model(data)
            loss = criterion(out, labels)

        total_loss += loss.item()
        n = labels.size(0)
        total += n
        if use_group:
            metric_sum += criterion.get_group_accuracy(out, labels) * n
        else:
            metric_sum += (out.argmax(1) == labels).sum().item()

        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), metric_sum / max(total, 1), all_preds, all_labels


# 6. ONNX EXPORT
def export_to_onnx(model, args, device, output_dir):
    print("\nExporting to ONNX")
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model

    _ONNX_B    = 1
    _ONNX_MAXN = 10

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def _encode(self, x, edge_index, batch):
            if hasattr(self.m, 'node_encoder') and self.m.node_encoder is not None:
                x = self.m.node_encoder(x)
            if hasattr(self.m, 'convs') and self.m.convs is not None:
                norms    = getattr(self.m, 'norms', None)
                residual = getattr(self.m, 'use_residual', True)
                dropout  = getattr(self.m, 'dropout', nn.Identity())
                act      = getattr(self.m, 'act', nn.GELU())
                for i, conv in enumerate(self.m.convs):
                    x_new = conv(x, edge_index)
                    if norms is not None:
                        x_new = norms[i](x_new)
                    x_new = act(x_new)
                    x_new = dropout(x_new)
                    if residual:
                        x = x + x_new
                    else:
                        x = x_new
            return x

        def forward(self, x, edge_index, batch, trackster_features=None):
            parts = []
            if self.m.input_mode in ('clusters', 'both'):
                x_enc        = self._encode(x, edge_index, batch)
                x_mean, x_max = _dense_pool_onnx(x_enc, batch, _ONNX_B, _ONNX_MAXN)
                parts.append(torch.cat([x_mean, x_max], dim=1))
            if (self.m.input_mode in ('trackster', 'both')
                    and trackster_features is not None
                    and hasattr(self.m, 'trackster_encoder')
                    and self.m.trackster_encoder is not None):
                parts.append(self.m.trackster_encoder(trackster_features))
            return self.m.classifier(torch.cat(parts, dim=1))

    onnx_model = Wrapper(raw).eval().cpu()
    ins, dummy, dyn = [], [], {'output': {0: 'batch'}}

    if args.input_mode in ('clusters', 'both'):
        ins   += ['x', 'edge_index', 'batch']
        dummy += [torch.randn(10, 4), torch.randint(0, 10, (2, 20)),
                  torch.zeros(10, dtype=torch.long)]
        dyn.update({'x': {0: 'nodes'}, 'edge_index': {1: 'edges'}, 'batch': {0: 'nodes'}})

    if args.input_mode in ('trackster', 'both'):
        ins   += ['trackster_features']
        dummy += [torch.randn(1, 3)]
        dyn['trackster_features'] = {0: 'batch'}

    onnx_path = os.path.join(output_dir, f'gnn_{args.model}_{args.input_mode}.onnx')
    torch.onnx.export(onnx_model, tuple(dummy), onnx_path,
                      input_names=ins, output_names=['output'],
                      dynamic_axes=dyn, opset_version=14)
    print(f"ONNX saved to {onnx_path}")


# 7. MAIN

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced GNN for TICL cluster graphs EM vs HAD')
    parser.add_argument('--input',       '-i', type=str, required=True)
    parser.add_argument('--output-dir',  '-o', type=str, default='./output_gnn')
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--hidden-dim',  type=int,   default=128)
    parser.add_argument('--num-layers',  type=int,   default=3)
    parser.add_argument('--num-heads',   type=int,   default=4)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--patience',    type=int,   default=10)
    parser.add_argument('--model',       type=str,   default='gnn',
                        choices=['gnn', 'sage', 'gat'])
    parser.add_argument('--input-mode',  type=str,   default='both',
                        choices=['clusters', 'trackster', 'both'])
    parser.add_argument('--group-loss',  action='store_true',
                        help='Use EM vs HAD group-collapsed loss')
    parser.add_argument('--exclude-pid', type=int,   default=None,
                        help='Exclude a specific PID from training')
    parser.add_argument('--amp',     action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--device',  type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--resume',  type=str, default=None)
    parser.add_argument('--export',  type=str, default=None,
                        help='Export checkpoint to ONNX (skips training)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    # ---- Export-only mode ----
    if args.export:
        model = build_model(args).to(device)
        ckpt  = torch.load(args.export, map_location=device, weights_only=False)
        state = ckpt['model_state_dict']
        if hasattr(model, '_orig_mod'):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        export_to_onnx(model, args, device, args.output_dir)
        return

    # ---- Datasets ----
    print("=" * 60)
    print("Loading dataset splits")
    train_ds = TracksterGraphDataset(args.input, split='train',
                                     seed=args.seed, exclude_pid=args.exclude_pid)
    val_ds   = TracksterGraphDataset(args.input, split='val',   seed=args.seed)
    test_ds  = TracksterGraphDataset(args.input, split='test',  seed=args.seed)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=4, collate_fn=collate_graphs,
                          pin_memory=(device.type == 'cuda'),
                          persistent_workers=True,
                          worker_init_fn=worker_init_fn)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)
    test_loader  = make_loader(test_ds,  shuffle=False)

    # ---- Model ----
    model = build_model(args).to(device)
    if args.compile:
        print("Compiling with torch.compile")
        model = torch.compile(model)
    print(f"Model: {args.model.upper()}  |  Parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f"Input mode: {args.input_mode}  |  "
          f"Loss: {'GroupCollapsedLoss' if args.group_loss else 'CrossEntropyLoss'}")

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.6, patience=5, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None

    if args.group_loss:
        criterion = GroupCollapsedLoss().to(device)
    else:
        w = torch.ones(num_classes)
        for c in EM_CLASS_INDICES + HAD_CLASS_INDICES:
            w[c] = 2.0
        criterion = nn.CrossEntropyLoss(weight=w.to(device))

    # ---- Resume ----
    start_epoch      = 0
    best_val_metric  = 0.0
    patience_counter = 0
    history = dict(train_loss=[], val_loss=[], train_metric=[], val_metric=[])

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
        best_val_metric  = ckpt.get('best_val_metric', 0.0)
        patience_counter = ckpt.get('patience_counter', 0)
        if 'history' in ckpt:
            history = ckpt['history']
        print(f"  Resumed at epoch {start_epoch}, best={best_val_metric:.4f}")
    elif args.resume:
        print(f"Checkpoint {args.resume} not found  starting fresh.")

    # ---- Training loop ----
    print("\n" + "=" * 60)
    print(f"Training  |  model={args.model}  |  input_mode={args.input_mode}")
    print("=" * 60)
    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        train_loss, train_metric = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_metric, _, _ = evaluate(
            model, val_loader, criterion, device, scaler)

        scheduler.step(val_metric)
        current_lr = optimizer.param_groups[0]['lr']
        elapsed    = time.time() - t0
        mname      = "group_acc" if args.group_loss else "acc"

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metric'].append(train_metric)
        history['val_metric'].append(val_metric)

        print(f"\nEpoch {epoch+1:3d}  ({elapsed:.1f}s)  LR={current_lr:.2e}")
        print(f"  Train  loss={train_loss:.4f}  {mname}={train_metric:.4f}")
        print(f"  Val    loss={val_loss:.4f}  {mname}={val_metric:.4f}")

        if val_metric > best_val_metric:
            best_val_metric  = val_metric
            patience_counter = 0
            ckpt_name = f'best_{args.model}_{args.input_mode}.pt'
            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict':    scaler.state_dict() if scaler else None,
                'best_val_metric':      best_val_metric,
                'patience_counter':     patience_counter,
                'args':                 args,
                'history':              history,
            }, os.path.join(args.output_dir, ckpt_name))
            print(f"  Best model saved ({mname}={val_metric:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nTraining done in {(time.time()-start_time)/60:.1f} min")

    # ---- Test ----
    ckpt_name = f'best_{args.model}_{args.input_mode}.pt'
    ckpt_path = os.path.join(args.output_dir, ckpt_name)
    if os.path.isfile(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt['model_state_dict']
        if hasattr(model, '_orig_mod'):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model.load_state_dict(state)

    test_loss, test_metric, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, scaler)

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Loss:   {test_loss:.4f}")
    print(f"  {mname}: {test_metric:.4f}")

    def to_group(arr):
        g = np.full_like(arr, 2)
        for i in EM_CLASS_INDICES:  g[arr == i] = 0
        for i in HAD_CLASS_INDICES: g[arr == i] = 1
        return g

    pg = to_group(np.array(test_preds))
    lg = to_group(np.array(test_labels))
    mask = lg != 2
    cm = confusion_matrix(lg[mask], pg[mask], labels=[0, 1])
    if cm.sum() > 0:
        print(f"  EM  precision={cm[0,0]/max(cm[:,0].sum(),1):.4f}  "
              f"recall={cm[0,0]/max(cm[0].sum(),1):.4f}")
        print(f"  HAD precision={cm[1,1]/max(cm[:,1].sum(),1):.4f}  "
              f"recall={cm[1,1]/max(cm[1].sum(),1):.4f}")
    print("=" * 60)

    class_names = ['photon','electron','muon','pion0',
                   'charged_hadron','neutral_hadron','unknown','noise']
    print("\n8-Class Report:")
    print(classification_report(test_labels, test_preds,
                                labels=list(range(8)),
                                target_names=class_names, zero_division=0))

    # Confusion matrix plot
    if cm.sum() > 0:
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annot  = np.array([[f'{v:.1f}%' for v in row] for row in cm_pct])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_pct, annot=annot, fmt='', cmap='Blues',
                    xticklabels=['EM','HAD'], yticklabels=['EM','HAD'],
                    vmin=0, vmax=100)
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.title(f'{args.model.upper()} EM vs HAD  {mname}={test_metric:.4f}',
                  fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir,
                    f'confusion_em_had_{args.model}.png'), dpi=150)
        plt.close()

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history['train_metric'], label='Train')
    axes[1].plot(history['val_metric'],   label='Val')
    axes[1].set_title(mname); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f'training_curves_{args.model}.png'), dpi=150)
    plt.close()
    print(f"Plots saved to {args.output_dir}/")

    export_to_onnx(model, args, device, args.output_dir)


if __name__ == "__main__":
    main()
