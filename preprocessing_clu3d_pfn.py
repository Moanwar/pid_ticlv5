#!/usr/bin/env python3

"""
# Single ROOT file
python preprocessing_clu3d.py --input data/sample.root

# Single text file with list of ROOT files
python preprocessing_clu3d.py --input file_list.txt

# Multiple text files
python preprocessing_clu3d.py --input list1.txt list2.txt list3.txt

# Directory containing ROOT files
python preprocessing_clu3d.py --input /path/to/root/files/

# Mix of sources
python preprocessing_clu3d.py --input single.root list.txt /data/dir/

# With options
python preprocessing_clu3d.py --input file_list.txt --output mydata.h5 --num-workers 8 --max-files 50

# Just merge with no shuffle
python preprocessing_clu3d.py --input dummy --output ticl_clu3d_data.h5 --merge-only --no-shuffle --partial-dir ticl_clu3d_data_partials/

# Merge and shuffle
python preprocessing_clu3d.py --input dummy --output ticl_clu3d_data.h5 --merge-only --partial-dir ticl_clu3d_data_partials/

# Only merge already-processed partial files (skip processing, useful after a crash)
python preprocessing_clu3d.py --input file_list.txt --output mydata.h5 --merge-only

Output shapes per trackster:
  clusters  : (50, 10, 7)  float32
                axis-0 = layer slot  (layer_id - 1, range 0-49)
                axis-1 = up to 10 clusters per layer, sorted by descending energy
                axis-2 = [energy, |eta|, phi, x, y, |z|, num_hits]
                zero-padded wherever fewer than 10 clusters exist in a layer
  features  : (7,)         float32
                [raw_energy, raw_em_energy, x, y, |z|, |eta|, phi]
  true_pid  : ()           int32
  true_energy: ()          float32
"""

import os
import os.path as osp
import argparse
import numpy as np
import uproot
import awkward as ak
import h5py
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SCORE_RECO2SIM = 0.6   
MAX_SCORE_SIM2RECO = 0.9   

N_LAYERS      = 50   # layer slots  (layer_id 1..50  index 0..49)
MAX_CLU_LAYER = 10   # max clusters kept per layer (top-N by energy)
N_CLU_FEAT    = 7    # [energy, |eta|, phi, x, y, |z|, num_hits]
N_TS_FEAT     = 7    # [raw_energy, raw_em_energy, x, y, |z|, |eta|, phi]

# HDF5 write chunk (number of tracksters per chunk)
H5_CHUNK_SIZE = 10_000

# Branches to read
TRACKSTER_FEATURES = [
    "raw_energy", "raw_em_energy",
    "barycenter_x", "barycenter_y", "barycenter_z",
    "barycenter_eta", "barycenter_phi",
    "vertices_indexes",
]
CLUSTER_FEATURES = [
    "energy",
    "position_eta", "position_phi",
    "position_x", "position_y", "position_z",
    "cluster_number_of_hits",
    "cluster_layer_id",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_branch_with_highest_cycle(file, branch_name):
    try:
        all_keys = file.keys()
        matching = [k for k in all_keys if k.startswith(branch_name)]
        if not matching:
            return None
        best = max(matching, key=lambda k: int(k.split(";")[1]))
        return file[best]
    except Exception:
        return None


def _open_h5(path, mode, **kwargs):
    """Open HDF5 with locking disabled — required for EOS/NFS/GPFS/AFS."""
    try:
        return h5py.File(path, mode, locking=False, **kwargs)
    except TypeError:
        return h5py.File(path, mode, **kwargs)


# ---------------------------------------------------------------------------
# Cluster image builder  (fully vectorised, no Python loops per trackster)
# ---------------------------------------------------------------------------

def build_cluster_image(v_idxs, ev_clu_feat):
    """
    Build a (50, 10, 7) float32 image for ONE trackster.

    Parameters
    ----------
    v_idxs     : 1-D int32 array  valid cluster indices for this trackster
    ev_clu_feat: (n_clusters, 7) float32 pre-built per-event cluster feature
                 matrix with columns [energy, |eta|, phi, x, y, |z|, num_hits]
                 (column 0 is energy, used for top-10 selection)

    Returns
    -------
    image : (50, 10, 7) float32, zero-padded
    """
    image = np.zeros((N_LAYERS, MAX_CLU_LAYER, N_CLU_FEAT), dtype=np.float32)

    if len(v_idxs) == 0:
        return image

    # Gather features for the clusters belonging to this trackster
    # Shape: (n_clu, 7)
    clu_feat = ev_clu_feat[v_idxs]          # [energy, |eta|, phi, x, y, |z|, num_hits]
    # layer_id column is stored separately e pass layer ids alongside
    # Caller guarantees ev_clu_layer[v_idxs] gives 0-based layer indices
    return image  # placeholder see build_cluster_image_with_layers below


def build_cluster_image_with_layers(v_idxs, ev_clu_feat, ev_clu_layer0):
    """
    Build a (50, 10, 7) float32 image for ONE trackster.

    Parameters
    ----------
    v_idxs        : 1-D int32 array valid cluster indices (already bounds-checked)
    ev_clu_feat   : (n_clusters, 7) float32 columns:
                    [energy, |eta|, phi, x, y, |z|, num_hits]
    ev_clu_layer0 : (n_clusters,)   int32 0-based layer index (layer_id - 1)

    Returns
    -------
    image : (50, 10, 7) float32
    """
    image = np.zeros((N_LAYERS, MAX_CLU_LAYER, N_CLU_FEAT), dtype=np.float32)

    if len(v_idxs) == 0:
        return image

    clu_feat  = ev_clu_feat[v_idxs]       # (n_clu, 7)
    clu_layer = ev_clu_layer0[v_idxs]     # (n_clu,)

    # Clamp layer indices to valid range (defensive)
    valid_mask = (clu_layer >= 0) & (clu_layer < N_LAYERS)
    clu_feat   = clu_feat[valid_mask]
    clu_layer  = clu_layer[valid_mask]

    if len(clu_layer) == 0:
        return image

    # Sort ALL clusters by layer first, then by descending energy within layer.
    # energy is column 0.
    sort_key = clu_layer * 1_000_000 - clu_feat[:, 0]   # ascending layer, descending energy
    order    = np.argsort(sort_key, kind='stable')
    clu_feat  = clu_feat[order]
    clu_layer = clu_layer[order]

    # Find layer boundaries with np.searchsorted (vectorised, no Python loop)
    unique_layers, layer_starts = np.unique(clu_layer, return_index=True)
    layer_ends = np.append(layer_starts[1:], len(clu_layer))

    for i in range(len(unique_layers)):
        lay   = unique_layers[i]
        start = layer_starts[i]
        end   = layer_ends[i]
        # Top-10 by energy (already sorted descending within this layer slice)
        count = min(end - start, MAX_CLU_LAYER)
        image[lay, :count, :] = clu_feat[start:start + count]

    return image


# ---------------------------------------------------------------------------
# HDF5 writers
# ---------------------------------------------------------------------------

def write_records_to_h5(records, h5_path):
    """
    Write a list of record dicts to a flat-columnar HDF5 file.

    Each record must have:
        'features'    : (7,)        float32  trackster features
        'true_pid'    : int
        'true_energy' : float
        'cluster_img' : (50,10,7)   float32  cluster image
    """
    n = len(records)
    with h5py.File(h5_path, 'w') as f:
        ds_feat = f.create_dataset(
            'features',
            shape=(n, N_TS_FEAT),
            dtype='f4',
            chunks=(min(n, H5_CHUNK_SIZE), N_TS_FEAT),
        )
        ds_pid  = f.create_dataset('true_pid',     shape=(n,), dtype='i4')
        ds_en   = f.create_dataset('true_energy',  shape=(n,), dtype='f4')
        ds_cimg = f.create_dataset(
            'clusters',
            shape=(n, N_LAYERS, MAX_CLU_LAYER, N_CLU_FEAT),
            dtype='f4',
            chunks=(min(n, H5_CHUNK_SIZE), N_LAYERS, MAX_CLU_LAYER, N_CLU_FEAT),
        )

        f.attrs['num_tracksters']     = n
        f.attrs['trackster_features'] = [
            'raw_energy', 'raw_em_energy', 'x', 'y', 'abs_z', 'abs_eta', 'phi'
        ]
        f.attrs['cluster_features']   = [
            'energy', 'abs_eta', 'phi', 'x', 'y', 'abs_z', 'num_hits'
        ]
        f.attrs['cluster_shape'] = [N_LAYERS, MAX_CLU_LAYER, N_CLU_FEAT]

        for start in range(0, n, H5_CHUNK_SIZE):
            end   = min(start + H5_CHUNK_SIZE, n)
            batch = records[start:end]
            b     = end - start

            feat_buf = np.empty((b, N_TS_FEAT),                      dtype=np.float32)
            pid_buf  = np.empty(b,                                    dtype=np.int32)
            en_buf   = np.empty(b,                                    dtype=np.float32)
            cimg_buf = np.zeros((b, N_LAYERS, MAX_CLU_LAYER, N_CLU_FEAT), dtype=np.float32)

            for j, rec in enumerate(batch):
                feat_buf[j]  = rec['features']
                pid_buf[j]   = rec['true_pid']
                en_buf[j]    = rec['true_energy']
                cimg_buf[j]  = rec['cluster_img']

            ds_feat[start:end] = feat_buf
            ds_pid [start:end] = pid_buf
            ds_en  [start:end] = en_buf
            ds_cimg[start:end] = cimg_buf


# ---------------------------------------------------------------------------
# Core per-file processor
# ---------------------------------------------------------------------------

def process_root_file(file_path, partial_dir):
    """
    Process one ROOT file and write results to partial_dir/<name>.h5
    Returns (n_tracksters, partial_h5_path) or (0, None).
    """
    basename   = osp.splitext(osp.basename(file_path))[0]
    partial_h5 = osp.join(partial_dir, f"{basename}.h5")

    # Crash-recovery: skip if already done
    if osp.exists(partial_h5):
        try:
            with h5py.File(partial_h5, 'r') as f:
                n = int(f.attrs.get('num_tracksters', len(f['features'])))
            return n, partial_h5
        except Exception:
            pass  # corrupted reprocess

    try:
        file = uproot.open(file_path)

        tracksters_tree   = load_branch_with_highest_cycle(file, 'ticlDumper/ticlTrackstersCLUE3DHigh')
        simcandidate_tree = load_branch_with_highest_cycle(file, 'ticlDumper/simTICLCandidate')
        associations_tree = load_branch_with_highest_cycle(file, 'ticlDumper/associations')
        clusters_tree     = load_branch_with_highest_cycle(file, 'ticlDumper/clusters')

        if any(t is None for t in [tracksters_tree, simcandidate_tree,
                                    associations_tree, clusters_tree]):
            return 0, None

        tracksters = tracksters_tree.arrays(TRACKSTER_FEATURES, library="ak")
        clusters   = clusters_tree.arrays(CLUSTER_FEATURES,     library="ak")

        assoc_branches = [
            'ticlTrackstersCLUE3DHigh_simToReco_CP_sharedE',
            'ticlTrackstersCLUE3DHigh_recoToSim_CP',
            'ticlTrackstersCLUE3DHigh_simToReco_CP_score',
            'ticlTrackstersCLUE3DHigh_recoToSim_CP_score',
            'ticlTrackstersCLUE3DHigh_simToReco_CP',
            'ticlTrackstersCLUE3DHigh_recoToSim_CP_sharedE',
        ]
        # deduplicate while preserving order
        seen_b = set()
        assoc_branches = [b for b in assoc_branches
                          if not (b in seen_b or seen_b.add(b))]
        assoc = associations_tree.arrays(assoc_branches, library="ak")

        sim_candidates = simcandidate_tree.arrays(
            ["simTICLCandidate_regressed_energy", "simTICLCandidate_pdgId"],
            library="ak",
        )

        valid_records = []

        for event_idx in range(len(tracksters)):
            try:
                sim_to_reco_sharedE = assoc['ticlTrackstersCLUE3DHigh_simToReco_CP_sharedE'][event_idx]
                if sim_to_reco_sharedE is None or len(sim_to_reco_sharedE) == 0:
                    continue

                reco_to_sim_scores = assoc['ticlTrackstersCLUE3DHigh_recoToSim_CP_score'][event_idx]
                sim_to_reco_scores = assoc['ticlTrackstersCLUE3DHigh_simToReco_CP_score'][event_idx]
                sim_to_reco_index  = assoc['ticlTrackstersCLUE3DHigh_simToReco_CP'][event_idx]

                # ---- Build per-event cluster feature matrix ONCE ----
                # Columns: [energy, |eta|, phi, x, y, |z|, num_hits]
                ev_clu_energy  = np.asarray(clusters['energy'][event_idx],              dtype=np.float32)
                ev_clu_eta     = np.abs(np.asarray(clusters['position_eta'][event_idx], dtype=np.float32))
                ev_clu_phi     = np.asarray(clusters['position_phi'][event_idx],         dtype=np.float32)
                ev_clu_x       = np.asarray(clusters['position_x'][event_idx],           dtype=np.float32)
                ev_clu_y       = np.asarray(clusters['position_y'][event_idx],           dtype=np.float32)
                ev_clu_z       = np.abs(np.asarray(clusters['position_z'][event_idx],   dtype=np.float32))
                ev_clu_nhits   = np.asarray(clusters['cluster_number_of_hits'][event_idx], dtype=np.float32)

                # 0-based layer index:  layer_id (1-indexed) - 1
                ev_clu_layer0  = np.asarray(clusters['cluster_layer_id'][event_idx], dtype=np.int32) - 1

                n_clusters = len(ev_clu_energy)

                # Stack into (n_clusters, 7) matches N_CLU_FEAT order
                ev_clu_feat = np.stack([
                    ev_clu_energy,
                    ev_clu_eta,
                    ev_clu_phi,
                    ev_clu_x,
                    ev_clu_y,
                    ev_clu_z,
                    ev_clu_nhits,
                ], axis=1)  # (n_clusters, 7)

                # ---- Trackster-level arrays (ONCE per event) ----
                ev_ts_raw_en    = np.asarray(tracksters['raw_energy'][event_idx],      dtype=np.float32)
                ev_ts_raw_em_en = np.asarray(tracksters['raw_em_energy'][event_idx],   dtype=np.float32)
                ev_ts_x         = np.asarray(tracksters['barycenter_x'][event_idx],    dtype=np.float32)
                ev_ts_y         = np.asarray(tracksters['barycenter_y'][event_idx],    dtype=np.float32)
                ev_ts_z         = np.abs(np.asarray(tracksters['barycenter_z'][event_idx], dtype=np.float32))
                ev_ts_eta       = np.abs(np.asarray(tracksters['barycenter_eta'][event_idx], dtype=np.float32))
                ev_ts_phi       = np.asarray(tracksters['barycenter_phi'][event_idx],   dtype=np.float32)

                # (n_tracksters, 7): [raw_energy, raw_em_energy, x, y, |z|, |eta|, phi]
                ev_ts_feat = np.stack([
                    ev_ts_raw_en,
                    ev_ts_raw_em_en,
                    ev_ts_x,
                    ev_ts_y,
                    ev_ts_z,
                    ev_ts_eta,
                    ev_ts_phi,
                ], axis=1)  # (n_tracksters, 7)

                ev_true_en  = np.asarray(
                    sim_candidates['simTICLCandidate_regressed_energy'][event_idx], dtype=np.float32)
                ev_true_pid = np.abs(np.asarray(
                    sim_candidates['simTICLCandidate_pdgId'][event_idx],            dtype=np.int32))

                # ---- Loop over sim particles ----
                for sim_idx in range(len(sim_to_reco_sharedE)):
                    shared_e_arr   = sim_to_reco_sharedE[sim_idx]
                    trackster_idxs = sim_to_reco_index[sim_idx]
                    if len(shared_e_arr) == 0:
                        continue

                    shared_e_np  = np.asarray(shared_e_arr,   dtype=np.float32)
                    ts_idx_np    = np.asarray(trackster_idxs, dtype=np.int32)
                    sim_scores_r = np.asarray(sim_to_reco_scores[sim_idx], dtype=np.float32)

                    true_en  = float(ev_true_en[sim_idx])
                    true_pid = int(ev_true_pid[sim_idx])

                    for local_idx in range(len(ts_idx_np)):
                        if shared_e_np[local_idx] <= 0:
                            continue

                        trackster_idx = int(ts_idx_np[local_idx])

                        # Score filtering
                        # Lower score = better match (scores are sorted ascending,
                        # best match is first entry).
                        # sim2reco: always apply skip if best score is too high
                        sim_score = (float(sim_scores_r[local_idx])
                                     if local_idx < len(sim_scores_r) else 1.0)
                        if sim_score > MAX_SCORE_SIM2RECO:
                            continue

                        # reco2sim: only apply if this trackster has any sim match at all
                        rts = reco_to_sim_scores[trackster_idx]
                        if len(rts) > 0 and float(rts[0]) > MAX_SCORE_RECO2SIM:
                            continue

                        # ---- Build cluster image (vectorised) ----
                        v_idx_raw = tracksters['vertices_indexes'][event_idx][trackster_idx]
                        if hasattr(v_idx_raw, 'tolist'):
                            v_idx_raw = v_idx_raw.tolist()
                        v_idxs = np.asarray(v_idx_raw, dtype=np.int32)
                        v_idxs = v_idxs[(v_idxs >= 0) & (v_idxs < n_clusters)]

                        cluster_img = build_cluster_image_with_layers(
                            v_idxs, ev_clu_feat, ev_clu_layer0
                        )

                        valid_records.append({
                            'features':    ev_ts_feat[trackster_idx],   # (7,)
                            'true_pid':    true_pid,
                            'true_energy': true_en,
                            'cluster_img': cluster_img,                 # (50,10,7)
                        })

            except Exception:
                continue

        if valid_records:
            write_records_to_h5(valid_records, partial_h5)
            return len(valid_records), partial_h5

        return 0, None

    except Exception as e:
        print(f"  Error processing {osp.basename(file_path)}: {type(e).__name__}: {e}")
        return 0, None


# ---------------------------------------------------------------------------
# Input file collection
# ---------------------------------------------------------------------------

def collect_input_files(input_paths, max_files=-1):
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    all_root_files = []
    for input_path in input_paths:
        if osp.isfile(input_path):
            if input_path.endswith('.root'):
                all_root_files.append(input_path)
            elif input_path.endswith('.txt'):
                try:
                    with open(input_path) as f:
                        files = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                        all_root_files.extend(p for p in files if p.endswith('.root'))
                except Exception:
                    print(f"Warning: Could not read {input_path}, skipping")
            else:
                print(f"Warning: Unknown file type {input_path}, skipping")
        elif osp.isdir(input_path):
            all_root_files.extend(glob(osp.join(input_path, "*.root")))

    seen, unique_files = set(), []
    for f in all_root_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    if max_files > 0:
        unique_files = unique_files[:max_files]

    valid   = [f for f in unique_files if osp.exists(f)]
    skipped = len(unique_files) - len(valid)
    if skipped:
        print(f"Warning: {skipped} file(s) not found on disk, skipped.")
    return valid


# ---------------------------------------------------------------------------
# Merge + shuffle partial h5 files into one big h5
# ---------------------------------------------------------------------------

def merge_partial_h5(partial_paths, output_file, shuffle=True, read_chunk=50_000):
    if not partial_paths:
        print("No partial files to merge.")
        return 0

    # Count total rows
    sizes = []
    for p in tqdm(partial_paths, desc="Scanning partial files"):
        try:
            with _open_h5(p, 'r') as f:
                sizes.append(int(f.attrs.get('num_tracksters', len(f['features']))))
        except Exception as e:
            print(f"  Warning: could not scan {osp.basename(p)}: {e}")
            sizes.append(0)

    total = sum(sizes)
    if total == 0:
        print("No tracksters found in partial files.")
        return 0

    print(f"Merging {total:,} tracksters from {len(partial_paths)} files {output_file}")

    tmp_dir  = os.environ.get('TMPDIR', '/tmp')
    tmp_file = osp.join(tmp_dir, f"merge_{os.getpid()}.h5")
    print(f"  (writing to local temp: {tmp_file})")

    CLU_SHAPE = (N_LAYERS, MAX_CLU_LAYER, N_CLU_FEAT)

    try:
        with h5py.File(tmp_file, 'w') as out:
            ds_feat = out.create_dataset(
                'features',
                shape=(total, N_TS_FEAT),
                dtype='f4',
                chunks=(min(total, H5_CHUNK_SIZE), N_TS_FEAT),
            )
            ds_pid  = out.create_dataset('true_pid',    shape=(total,), dtype='i4')
            ds_en   = out.create_dataset('true_energy', shape=(total,), dtype='f4')
            ds_cimg = out.create_dataset(
                'clusters',
                shape=(total, *CLU_SHAPE),
                dtype='f4',
                chunks=(min(total, H5_CHUNK_SIZE), *CLU_SHAPE),
            )

            out.attrs['num_tracksters']     = total
            out.attrs['trackster_features'] = [
                'raw_energy', 'raw_em_energy', 'x', 'y', 'abs_z', 'abs_eta', 'phi'
            ]
            out.attrs['cluster_features']   = [
                'energy', 'abs_eta', 'phi', 'x', 'y', 'abs_z', 'num_hits'
            ]
            out.attrs['cluster_shape'] = list(CLU_SHAPE)

            # --- Pass 1: stream-copy partials local temp ---
            write_ptr = 0
            for p, n in tqdm(zip(partial_paths, sizes), total=len(partial_paths),
                             desc="Merging"):
                if n == 0:
                    continue
                try:
                    with _open_h5(p, 'r') as src:
                        for start in range(0, n, read_chunk):
                            end = min(start + read_chunk, n)
                            b   = end - start
                            dst = write_ptr + start
                            ds_feat[dst:dst+b] = src['features'][start:end]
                            ds_pid [dst:dst+b] = src['true_pid'][start:end]
                            ds_en  [dst:dst+b] = src['true_energy'][start:end]
                            ds_cimg[dst:dst+b] = src['clusters'][start:end]
                    write_ptr += n
                except Exception as e:
                    print(f"  Warning: could not merge {osp.basename(p)}: {e}")

            # Store shuffle index regardless
            rng     = np.random.default_rng()
            indices = rng.permutation(total).astype(np.int64)
            out.create_dataset('shuffle_indices', data=indices, dtype='i8')

        # --- Pass 2: physical shuffle (optional) ---
        if shuffle:
            print("Shuffling (sequential two-pass rewrite)")
            tmp2_file = osp.join(tmp_dir, f"merge_{os.getpid()}_shuffled.h5")

            # write_pos[src_row] = dst_row
            write_pos = np.empty(total, dtype=np.int64)
            write_pos[indices] = np.arange(total, dtype=np.int64)

            try:
                with h5py.File(tmp_file, 'r') as src_f, \
                     h5py.File(tmp2_file, 'w') as dst_f:

                    ds2_feat = dst_f.create_dataset(
                        'features', shape=(total, N_TS_FEAT), dtype='f4',
                        chunks=(min(total, H5_CHUNK_SIZE), N_TS_FEAT))
                    ds2_pid  = dst_f.create_dataset('true_pid',    shape=(total,), dtype='i4')
                    ds2_en   = dst_f.create_dataset('true_energy', shape=(total,), dtype='f4')
                    ds2_cimg = dst_f.create_dataset(
                        'clusters', shape=(total, *CLU_SHAPE), dtype='f4',
                        chunks=(min(total, H5_CHUNK_SIZE), *CLU_SHAPE))
                    dst_f.create_dataset('shuffle_indices', data=indices, dtype='i8')
                    dst_f.attrs['num_tracksters']     = total
                    dst_f.attrs['trackster_features'] = [
                        'raw_energy', 'raw_em_energy', 'x', 'y', 'abs_z', 'abs_eta', 'phi'
                    ]
                    dst_f.attrs['cluster_features']   = [
                        'energy', 'abs_eta', 'phi', 'x', 'y', 'abs_z', 'num_hits'
                    ]
                    dst_f.attrs['cluster_shape'] = list(CLU_SHAPE)

                    src_feat = src_f['features']
                    src_pid  = src_f['true_pid']
                    src_en   = src_f['true_energy']
                    src_cimg = src_f['clusters']

                    for start in tqdm(range(0, total, read_chunk), desc="Shuffling"):
                        end = min(start + read_chunk, total)

                        feat_buf = src_feat[start:end]
                        pid_buf  = src_pid [start:end]
                        en_buf   = src_en  [start:end]
                        cimg_buf = src_cimg[start:end]

                        dst_idx  = write_pos[start:end]
                        sort_d   = np.argsort(dst_idx)
                        dst_sort = dst_idx[sort_d]

                        ds2_feat[dst_sort] = feat_buf[sort_d]
                        ds2_pid [dst_sort] = pid_buf [sort_d]
                        ds2_en  [dst_sort] = en_buf  [sort_d]
                        ds2_cimg[dst_sort] = cimg_buf[sort_d]

                os.replace(tmp2_file, tmp_file)

            except Exception:
                if osp.exists(tmp2_file):
                    os.remove(tmp2_file)
                raise

        print(f"Copying local temp {output_file}")
        shutil.copy2(tmp_file, output_file)
        print(f"Output written to {output_file}")

    finally:
        for f in [tmp_file,
                  osp.join(tmp_dir, f"merge_{os.getpid()}_shuffled.h5")]:
            if osp.exists(f):
                os.remove(f)

    return total


# ---------------------------------------------------------------------------
# Parallel processing
# ---------------------------------------------------------------------------

def process_files_parallel(input_files, partial_dir, num_workers=1):
    os.makedirs(partial_dir, exist_ok=True)
    print(f"\nProcessing {len(input_files)} files with {num_workers} worker(s)")
    print(f"Partial files {partial_dir}")

    partial_h5_paths = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(process_root_file, f, partial_dir): f
            for f in input_files
        }
        with tqdm(total=len(input_files), desc="Processing files") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    n, h5_path = future.result()
                    if h5_path:
                        partial_h5_paths.append(h5_path)
                    pbar.set_postfix({"valid": n,
                                      "file": osp.basename(file_path)[:25]})
                except Exception as e:
                    print(f"\nError on {file_path}: {e}")
                pbar.update(1)

    return partial_h5_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Optimised preprocessing for TICL PID data cluster images (50 x 10 x 7)'
    )
    parser.add_argument('--input', '-i', type=str, nargs='+', required=True,
                        help='Input: .root file(s), .txt file(s) with file lists, or directory(ies)')
    parser.add_argument('--output', '-o', type=str, default='ticl_clu3d_data_pfn.h5',
                        help='Output HDF5 file (default: ticl_clu3d_data_pfn.h5)')
    parser.add_argument('--partial-dir', type=str, default=None,
                        help='Directory for per-file partial HDF5 files '
                             '(default: <output_stem>_partials/)')
    parser.add_argument('--max-files', type=int, default=-1,
                        help='Maximum number of ROOT files to process (default: all)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel worker processes (default: 1)')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Disable global shuffle of merged output')
    parser.add_argument('--merge-only', action='store_true',
                        help='Skip ROOT processing; only merge existing partial files')
    parser.add_argument('--keep-partials', action='store_true',
                        help='Keep partial files after successful merge')
    args = parser.parse_args()

    if args.partial_dir is None:
        base = osp.splitext(args.output)[0]
        args.partial_dir = base + '_partials'

    # ---- Processing phase ----
    if not args.merge_only:
        print("Collecting input files")
        input_files = collect_input_files(args.input, args.max_files)
        if not input_files:
            print("No valid input files found!")
            return
        partial_paths = process_files_parallel(input_files, args.partial_dir, args.num_workers)
    else:
        partial_paths = sorted(glob(osp.join(args.partial_dir, "*.h5")))
        print(f"--merge-only: found {len(partial_paths)} partial files in {args.partial_dir}")

    if not partial_paths:
        print("No partial files produced. Exiting.")
        return

    # Clean up any stale temp files from a previous crashed merge
    output_dir = osp.dirname(osp.abspath(args.output))
    for stale in glob(osp.join(output_dir, ".tmp_merge_*.h5")):
        print(f"Removing stale temp file: {stale}")
        os.remove(stale)

    # ---- Merge phase ----
    total = merge_partial_h5(partial_paths, args.output, shuffle=not args.no_shuffle)

    print(f"\nDone. {total:,} tracksters written to {args.output}")
    print(f"  Dataset shapes:")
    print(f"    clusters   : ({total}, {N_LAYERS}, {MAX_CLU_LAYER}, {N_CLU_FEAT})")
    print(f"    features   : ({total}, {N_TS_FEAT})")
    print(f"    true_pid   : ({total},)")
    print(f"    true_energy: ({total},)")

    if not args.keep_partials and not args.merge_only:
        ans = input("\nDelete partial files to free disk space? [y/N] ").strip().lower()
        if ans == 'y':
            shutil.rmtree(args.partial_dir)
            print("Partial files deleted.")
        else:
            print(f"Partial files kept in: {args.partial_dir}")


if __name__ == "__main__":
    main()
