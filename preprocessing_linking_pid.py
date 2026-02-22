#!/usr/bin/env python3

"""
# Single ROOT file
python preprocessing_linking_pid.py --input data/sample.root

# Single text file with list of ROOT files
python preprocessing_linking_pid.py --input file_list.txt

# Multiple text files
python preprocessing_linking_pid.py --input list1.txt list2.txt list3.txt

# Directory containing ROOT files
python preprocessing_linking_pid.py --input /path/to/root/files/

# Mix of sources
python preprocessing_linking_pid.py --input single.root list.txt /data/dir/

# With options
python preprocessing_linking_pid.py --input file_list.txt --output mydata.h5 --num-workers 8 --max-files 50

# Only merge already-processed partial files (skip processing, useful after a crash)
python preprocessing_linking_pid.py --input file_list.txt --output mydata.h5 --merge-only
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
import random
import shutil

# Selection thresholds
MAX_SCORE_RECO2SIM = 0.6
MAX_SCORE_SIM2RECO = 0.9

# Feature definitions
TRACKSTER_FEATURES = ["barycenter_eta", "barycenter_phi", "raw_energy", "vertices_indexes"]
CLUSTER_FEATURES   = ["position_eta", "position_phi", "energy", "cluster_layer_id"]

# How many tracksters to buffer per HDF5 write chunk
H5_CHUNK_SIZE = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_branch_with_highest_cycle(file, branch_name):
    try:
        all_keys     = file.keys()
        matching     = [k for k in all_keys if k.startswith(branch_name)]
        if not matching:
            return None
        best = max(matching, key=lambda k: int(k.split(";")[1]))
        return file[best]
    except Exception:
        return None


def _vlen_f32():
    return h5py.special_dtype(vlen=np.dtype('float32'))


def write_records_to_h5(records, h5_path):
    """Write a list of record dicts to a flat-columnar HDF5 file."""
    n = len(records)
    with h5py.File(h5_path, 'w') as f:
        ds_feat = f.create_dataset('features',     shape=(n, 3), dtype='f4',
                                   chunks=(min(n, H5_CHUNK_SIZE), 3))
        ds_pid  = f.create_dataset('true_pid',     shape=(n,),   dtype='i4')
        ds_en   = f.create_dataset('true_energy',  shape=(n,),   dtype='f4')
        ds_nclu = f.create_dataset('num_clusters', shape=(n,),   dtype='i4')
        ds_clus = f.create_dataset('clusters',     shape=(n,),   dtype=_vlen_f32())

        f.attrs['num_tracksters']     = n
        f.attrs['trackster_features'] = ['eta', 'phi', 'energy']
        f.attrs['cluster_features']   = ['eta', 'phi', 'energy', 'layer']

        for start in range(0, n, H5_CHUNK_SIZE):
            end   = min(start + H5_CHUNK_SIZE, n)
            batch = records[start:end]
            b     = end - start

            feat_buf = np.empty((b, 3), dtype=np.float32)
            pid_buf  = np.empty(b,      dtype=np.int32)
            en_buf   = np.empty(b,      dtype=np.float32)
            nclu_buf = np.empty(b,      dtype=np.int32)
            clus_buf = np.empty(b,      dtype=object)

            for j, rec in enumerate(batch):
                feat_buf[j] = rec['features']
                pid_buf[j]  = rec['true_pid']
                en_buf[j]   = rec['true_energy']
                clu         = rec['clusters']
                nclu_buf[j] = len(clu)
                clus_buf[j] = clu.ravel() if len(clu) else np.array([], dtype=np.float32)

            ds_feat[start:end] = feat_buf
            ds_pid [start:end] = pid_buf
            ds_en  [start:end] = en_buf
            ds_nclu[start:end] = nclu_buf
            ds_clus[start:end] = clus_buf


# ---------------------------------------------------------------------------
# Core per-file processor
# ---------------------------------------------------------------------------

def process_root_file(file_path, partial_dir):
    """
    Process one ROOT file and write results immediately to partial_dir/<name>.h5
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

        tracksters_tree   = load_branch_with_highest_cycle(file, 'ticlDumper/ticlCandidate')
        simcandidate_tree = load_branch_with_highest_cycle(file, 'ticlDumper/simTICLCandidate')
        associations_tree = load_branch_with_highest_cycle(file, 'ticlDumper/associations')
        clusters_tree     = load_branch_with_highest_cycle(file, 'ticlDumper/clusters')

        if any(t is None for t in [tracksters_tree, simcandidate_tree,
                                    associations_tree, clusters_tree]):
            return 0, None

        tracksters = tracksters_tree.arrays(TRACKSTER_FEATURES, library="ak")
        clusters   = clusters_tree.arrays(CLUSTER_FEATURES,     library="ak")

        assoc_branches = list(dict.fromkeys([
            'ticlCandidate_simToReco_CP_sharedE',
            'ticlCandidate_recoToSim_CP',
            'ticlCandidate_simToReco_CP_score',
            'ticlCandidate_recoToSim_CP_score',
            'ticlCandidate_simToReco_CP',
            'ticlCandidate_recoToSim_CP_sharedE',
        ]))
        assoc = associations_tree.arrays(assoc_branches, library="ak")

        sim_candidates = simcandidate_tree.arrays(
            ["simTICLCandidate_regressed_energy", "simTICLCandidate_pdgId"], library="ak")

        valid_records = []

        for event_idx in range(len(tracksters)):
            try:
                sim_to_reco_sharedE = assoc['ticlCandidate_simToReco_CP_sharedE'][event_idx]
                if sim_to_reco_sharedE is None or len(sim_to_reco_sharedE) == 0:
                    continue

                reco_to_sim_scores = assoc['ticlCandidate_recoToSim_CP_score'][event_idx]
                sim_to_reco_scores = assoc['ticlCandidate_simToReco_CP_score'][event_idx]
                sim_to_reco_index  = assoc['ticlCandidate_simToReco_CP'][event_idx]

                # Convert cluster arrays to numpy ONCE per event
                ev_clu_eta   = np.abs(np.asarray(clusters['position_eta'][event_idx],    dtype=np.float32))
                ev_clu_phi   = np.asarray(clusters['position_phi'][event_idx],            dtype=np.float32)
                ev_clu_en    = np.asarray(clusters['energy'][event_idx],                  dtype=np.float32)
                ev_clu_layer = np.asarray(clusters['cluster_layer_id'][event_idx],        dtype=np.float32)
                n_clusters   = len(ev_clu_en)

                ev_ts_eta    = np.abs(np.asarray(tracksters['barycenter_eta'][event_idx],  dtype=np.float32))
                ev_ts_phi    = np.asarray(tracksters['barycenter_phi'][event_idx],          dtype=np.float32)
                ev_ts_energy = np.asarray(tracksters['raw_energy'][event_idx],              dtype=np.float32)

                ev_true_en  = np.asarray(sim_candidates['simTICLCandidate_regressed_energy'][event_idx], dtype=np.float32)
                ev_true_pid = np.abs(np.asarray(sim_candidates['simTICLCandidate_pdgId'][event_idx],     dtype=np.int32))

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

                        reco_score = 1.0
                        rts = reco_to_sim_scores[trackster_idx]
                        if len(rts) > 0:
                            reco_score = float(rts[0])

                        sim_score = float(sim_scores_r[local_idx]) if local_idx < len(sim_scores_r) else 1.0

                        if reco_score > MAX_SCORE_RECO2SIM or sim_score > MAX_SCORE_SIM2RECO:
                            continue

                        # Build cluster array (vectorised)
                        v_idx_raw = tracksters['vertices_indexes'][event_idx][trackster_idx]
                        if hasattr(v_idx_raw, 'tolist'):
                            v_idx_raw = v_idx_raw.tolist()
                        v_idxs = np.asarray(v_idx_raw, dtype=np.int32)
                        v_idxs = v_idxs[(v_idxs >= 0) & (v_idxs < n_clusters)]

                        if len(v_idxs) > 0:
                            clu   = np.stack([ev_clu_eta[v_idxs],
                                              ev_clu_phi[v_idxs],
                                              ev_clu_en[v_idxs],
                                              ev_clu_layer[v_idxs]], axis=1)
                            order = np.argsort(clu[:, 3] * 1e9 - clu[:, 2])
                            clu   = clu[order]
                        else:
                            clu = np.empty((0, 4), dtype=np.float32)

                        valid_records.append({
                            'features':    np.array([ev_ts_eta[trackster_idx],
                                                     ev_ts_phi[trackster_idx],
                                                     ev_ts_energy[trackster_idx]], dtype=np.float32),
                            'true_pid':    true_pid,
                            'true_energy': true_en,
                            'clusters':    clu,
                        })

            except Exception:
                continue

        # Write to disk immediately frees all RAM for this file
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

    valid = [f for f in unique_files if osp.exists(f)]
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
            with h5py.File(p, 'r') as f:
                sizes.append(int(f.attrs.get('num_tracksters', len(f['features']))))
        except Exception:
            sizes.append(0)

    total = sum(sizes)
    if total == 0:
        print("No tracksters found in partial files.")
        return 0

    print(f"Merging {total:,} tracksters from {len(partial_paths)} files {output_file}")

    with h5py.File(output_file, 'w') as out:
        ds_feat = out.create_dataset('features',     shape=(total, 3), dtype='f4',
                                     chunks=(min(total, H5_CHUNK_SIZE), 3))
        ds_pid  = out.create_dataset('true_pid',     shape=(total,),   dtype='i4')
        ds_en   = out.create_dataset('true_energy',  shape=(total,),   dtype='f4')
        ds_nclu = out.create_dataset('num_clusters', shape=(total,),   dtype='i4')
        ds_clus = out.create_dataset('clusters',     shape=(total,),   dtype=_vlen_f32())

        out.attrs['num_tracksters']     = total
        out.attrs['trackster_features'] = ['eta', 'phi', 'energy']
        out.attrs['cluster_features']   = ['eta', 'phi', 'energy', 'layer']

        # --- Pass 1: stream-copy in file order ---
        write_ptr = 0
        for p, n in tqdm(zip(partial_paths, sizes), total=len(partial_paths),
                         desc="Merging"):
            if n == 0:
                continue
            try:
                with h5py.File(p, 'r') as src:
                    for start in range(0, n, read_chunk):
                        end = min(start + read_chunk, n)
                        b   = end - start
                        dst = write_ptr + start
                        ds_feat[dst:dst+b] = src['features'][start:end]
                        ds_pid [dst:dst+b] = src['true_pid'][start:end]
                        ds_en  [dst:dst+b] = src['true_energy'][start:end]
                        ds_nclu[dst:dst+b] = src['num_clusters'][start:end]
                        ds_clus[dst:dst+b] = src['clusters'][start:end]
                write_ptr += n
            except Exception as e:
                print(f"  Warning: could not merge {p}: {e}")

        # --- Pass 2: global in-place shuffle via index permutation ---
        if shuffle:
            print("Shuffling (in-place swap)")
            rng     = np.random.default_rng()
            indices = rng.permutation(total)  # shape [total]

            for start in tqdm(range(0, total, read_chunk), desc="Shuffling"):
                end     = min(start + read_chunk, total)
                src_idx = indices[start:end]   # where to READ from

                # Sort for sequential HDF5 reads (much faster)
                sort_order     = np.argsort(src_idx)
                sorted_src_idx = src_idx[sort_order]

                tmp_feat = ds_feat[sorted_src_idx]
                tmp_pid  = ds_pid [sorted_src_idx]
                tmp_en   = ds_en  [sorted_src_idx]
                tmp_nclu = ds_nclu[sorted_src_idx]
                tmp_clus = ds_clus[sorted_src_idx]

                # Un-sort to restore permutation order before writing
                unsort             = np.argsort(sort_order)
                ds_feat[start:end] = tmp_feat[unsort]
                ds_pid [start:end] = tmp_pid [unsort]
                ds_en  [start:end] = tmp_en  [unsort]
                ds_nclu[start:end] = tmp_nclu[unsort]
                ds_clus[start:end] = tmp_clus[unsort]

    return total


# ---------------------------------------------------------------------------
# Parallel processing
# ---------------------------------------------------------------------------

def process_files_parallel(input_files, partial_dir, num_workers=1):
    os.makedirs(partial_dir, exist_ok=True)
    print(f"\n Processing {len(input_files)} files with {num_workers} worker(s)")
    print(f"Partial files {partial_dir}")

    partial_h5_paths = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_root_file, f, partial_dir): f
                          for f in input_files}
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
    parser = argparse.ArgumentParser(description='Optimised preprocessing for TICL PID data')
    parser.add_argument('--input', '-i', type=str, nargs='+', required=True,
                        help='Input: .root file(s), .txt file(s) with file lists, or directory(ies)')
    parser.add_argument('--output', '-o', type=str, default='ticl_linking_data_pid.h5',
                        help='Output HDF5 file (default: ticl_linking_data_pid.h5)')
    parser.add_argument('--partial-dir', type=str, default=None,
                        help='Directory for per-file partial HDF5 files '
                             '(default: <output_stem>_partials/)')
    parser.add_argument('--max-files', type=int, default=-1,
                        help='Maximum number of ROOT files to process (default: all)')
    parser.add_argument('--num-workers', type=int, default=1,
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

    # ---- Merge phase ----
    total = merge_partial_h5(partial_paths, args.output, shuffle=not args.no_shuffle)

    print(f"\nDone. {total:,} tracksters written to {args.output}")

    if not args.keep_partials and not args.merge_only:
        ans = input("\nDelete partial files to free disk space? [y/N] ").strip().lower()
        if ans == 'y':
            shutil.rmtree(args.partial_dir)
            print("Partial files deleted.")
        else:
            print(f"Partial files kept in: {args.partial_dir}")


if __name__ == "__main__":
    main()
