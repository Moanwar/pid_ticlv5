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

#Just merge with no shuffle
python preprocessing_clu3d.py --input dummy --output ticl_clu3d_data.h5 --merge-only --no-shuffle --partial-dir ticl_clu3d_data_partials/

#merge and shuffle
python preprocessing_clu3d.py --input dummy --output ticl_clu3d_data.h5 --merge-only --partial-dir ticl_clu3d_data_partials/

# Only merge already-processed partial files (skip processing, useful after a crash)
python preprocessing_clu3d.py --input file_list.txt --output mydata.h5 --merge-only
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

        tracksters_tree   = load_branch_with_highest_cycle(file, 'ticlDumper/ticlTrackstersCLUE3DHigh')
        simcandidate_tree = load_branch_with_highest_cycle(file, 'ticlDumper/simTICLCandidate')
        associations_tree = load_branch_with_highest_cycle(file, 'ticlDumper/associations')
        clusters_tree     = load_branch_with_highest_cycle(file, 'ticlDumper/clusters')

        if any(t is None for t in [tracksters_tree, simcandidate_tree,
                                    associations_tree, clusters_tree]):
            return 0, None

        tracksters = tracksters_tree.arrays(TRACKSTER_FEATURES, library="ak")
        clusters   = clusters_tree.arrays(CLUSTER_FEATURES,     library="ak")

        assoc_branches = list(dict.fromkeys([
            'ticlTrackstersCLUE3DHigh_simToReco_CP_sharedE',
            'ticlTrackstersCLUE3DHigh_recoToSim_CP',
            'ticlTrackstersCLUE3DHigh_simToReco_CP_score',
            'ticlTrackstersCLUE3DHigh_recoToSim_CP_score',
            'ticlTrackstersCLUE3DHigh_simToReco_CP',
            'ticlTrackstersCLUE3DHigh_recoToSim_CP_sharedE',
        ]))
        assoc = associations_tree.arrays(assoc_branches, library="ak")

        sim_candidates = simcandidate_tree.arrays(
            ["simTICLCandidate_regressed_energy", "simTICLCandidate_pdgId"], library="ak")

        valid_records = []

        for event_idx in range(len(tracksters)):
            try:
                sim_to_reco_sharedE = assoc['ticlTrackstersCLUE3DHigh_simToReco_CP_sharedE'][event_idx]
                if sim_to_reco_sharedE is None or len(sim_to_reco_sharedE) == 0:
                    continue

                reco_to_sim_scores = assoc['ticlTrackstersCLUE3DHigh_recoToSim_CP_score'][event_idx]
                sim_to_reco_scores = assoc['ticlTrackstersCLUE3DHigh_simToReco_CP_score'][event_idx]
                sim_to_reco_index  = assoc['ticlTrackstersCLUE3DHigh_simToReco_CP'][event_idx]

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

def _open_h5(path, mode, **kwargs):
    """
    Open HDF5 with locking disabled — required for EOS/NFS/GPFS/AFS filesystems
    where POSIX advisory locks are broken or unavailable (errno 11).
    Falls back gracefully for older h5py that doesn't support the keyword.
    """
    try:
        return h5py.File(path, mode, locking=False, **kwargs)
    except TypeError:
        return h5py.File(path, mode, **kwargs)


def merge_partial_h5(partial_paths, output_file, shuffle=True, read_chunk=50_000):
    if not partial_paths:
        print("No partial files to merge.")
        return 0

    # Count total rows open partials read-only with locking disabled
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

    print(f"Merging {total:,} tracksters from {len(partial_paths)} files → {output_file}")

    # Write to LOCAL disk to avoid EOS locking entirely
    tmp_dir  = os.environ.get('TMPDIR', '/tmp')
    tmp_file = osp.join(tmp_dir, f"merge_{os.getpid()}.h5")
    print(f"  (writing to local temp: {tmp_file})")

    try:
        with h5py.File(tmp_file, 'w') as out:
            ds_feat = out.create_dataset('features',     shape=(total, 3), dtype='f4',
                                         chunks=(min(total, H5_CHUNK_SIZE), 3))
            ds_pid  = out.create_dataset('true_pid',     shape=(total,),   dtype='i4')
            ds_en   = out.create_dataset('true_energy',  shape=(total,),   dtype='f4')
            ds_nclu = out.create_dataset('num_clusters', shape=(total,),   dtype='i4')
            ds_clus = out.create_dataset('clusters',     shape=(total,),   dtype=_vlen_f32())

            out.attrs['num_tracksters']     = total
            out.attrs['trackster_features'] = ['eta', 'phi', 'energy']
            out.attrs['cluster_features']   = ['eta', 'phi', 'energy', 'layer']

            # --- Pass 1: stream-copy partials → local temp ---
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
                            ds_nclu[dst:dst+b] = src['num_clusters'][start:end]
                            ds_clus[dst:dst+b] = src['clusters'][start:end]
                    write_ptr += n
                except Exception as e:
                    print(f"  Warning: could not merge {osp.basename(p)}: {e}")

            # --- Pass 2: write shuffle index (instant, used by dataloader) ---
            rng     = np.random.default_rng()
            indices = rng.permutation(total).astype(np.int64)
            out.create_dataset('shuffle_indices', data=indices, dtype='i8')

        if shuffle:
            # Physical shuffle via a second sequential pass:
            #   READ tmp_file sequentially (chunk by chunk, ascending row order)
            #   WRITE each chunk to its shuffled destination in tmp2_file
            # This avoids random-access reads on vlen cluster data entirely.
            print("Shuffling (sequential two-pass rewrite)")
            tmp2_file = osp.join(tmp_dir, f"merge_{os.getpid()}_shuffled.h5")

            # Pre-compute: for each source row i, which destination row does it go to?
            # indices[dst] = src  =>  write_pos[src] = dst
            write_pos = np.empty(total, dtype=np.int64)
            write_pos[indices] = np.arange(total, dtype=np.int64)

            try:
                with h5py.File(tmp_file, 'r') as src_f, h5py.File(tmp2_file, 'w') as dst_f:
                    ds2_feat = dst_f.create_dataset('features',     shape=(total, 3), dtype='f4',
                                                    chunks=(min(total, H5_CHUNK_SIZE), 3))
                    ds2_pid  = dst_f.create_dataset('true_pid',     shape=(total,),   dtype='i4')
                    ds2_en   = dst_f.create_dataset('true_energy',  shape=(total,),   dtype='f4')
                    ds2_nclu = dst_f.create_dataset('num_clusters', shape=(total,),   dtype='i4')
                    ds2_clus = dst_f.create_dataset('clusters',     shape=(total,),   dtype=_vlen_f32())
                    dst_f.create_dataset('shuffle_indices', data=indices, dtype='i8')
                    dst_f.attrs['num_tracksters']     = total
                    dst_f.attrs['trackster_features'] = ['eta', 'phi', 'energy']
                    dst_f.attrs['cluster_features']   = ['eta', 'phi', 'energy', 'layer']

                    src_feat = src_f['features']
                    src_pid  = src_f['true_pid']
                    src_en   = src_f['true_energy']
                    src_nclu = src_f['num_clusters']
                    src_clus = src_f['clusters']

                    for start in tqdm(range(0, total, read_chunk), desc="Shuffling"):
                        end     = min(start + read_chunk, total)

                        # Sequential read from source (fast)
                        feat_buf = src_feat[start:end]
                        pid_buf  = src_pid [start:end]
                        en_buf   = src_en  [start:end]
                        nclu_buf = src_nclu[start:end]
                        clus_buf = src_clus[start:end]

                        # Destination indices for this chunk
                        dst_idx = write_pos[start:end]

                        # Sort destinations to make writes as sequential as possible
                        sort_d   = np.argsort(dst_idx)
                        dst_sort = dst_idx[sort_d]

                        ds2_feat[dst_sort] = feat_buf[sort_d]
                        ds2_pid [dst_sort] = pid_buf [sort_d]
                        ds2_en  [dst_sort] = en_buf  [sort_d]
                        ds2_nclu[dst_sort] = nclu_buf[sort_d]
                        ds2_clus[dst_sort] = clus_buf[sort_d]

                os.replace(tmp2_file, tmp_file)

            except Exception:
                if osp.exists(tmp2_file):
                    os.remove(tmp2_file)
                raise

        print(f"Copying local temp {output_file} ")
        shutil.copy2(tmp_file, output_file)
        print(f"Output written to {output_file}")

    finally:
        # Always clean up local temp files
        for f in [tmp_file, osp.join(tmp_dir, f"merge_{os.getpid()}_shuffled.h5")]:
            if osp.exists(f):
                os.remove(f)

    return total


# ---------------------------------------------------------------------------
# Parallel processing
# ---------------------------------------------------------------------------

def process_files_parallel(input_files, partial_dir, num_workers=1):
    os.makedirs(partial_dir, exist_ok=True)
    print(f"\nProcessing {len(input_files)} files with {num_workers} worker(s)…")
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
    parser.add_argument('--output', '-o', type=str, default='ticl_clu3d_data.h5',
                        help='Output HDF5 file (default: ticl_clu3d_data.h5)')
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
        print("Collecting input files…")
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
    # Clean up any stale temp files from a previous crashed merge
    output_dir = osp.dirname(osp.abspath(args.output))
    for stale in glob(osp.join(output_dir, ".tmp_merge_*.h5")):
        print(f"Removing stale temp file: {stale}")
        os.remove(stale)

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
