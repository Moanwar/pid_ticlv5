#!/usr/bin/env python3

"""
# Single ROOT file
python preprocess_simple.py --input data/sample.root

# Single text file with list of ROOT files
python preprocess_simple.py --input file_list.txt

# Multiple text files
python preprocess_simple.py --input list1.txt list2.txt list3.txt

# Directory containing ROOT files
python preprocess_simple.py --input /path/to/root/files/

# Mix of sources
python preprocess_simple.py --input single.root list.txt /data/dir/

# With options
python preprocess_simple.py --input file_list.txt --output mydata.h5 --num-workers 8 --max-files 50
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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import traceback

# Selection thresholds
MAX_SCORE_RECO2SIM = 0.6
MAX_SCORE_SIM2RECO = 0.9

# Simple feature definitions
TRACKSTER_FEATURES = ["barycenter_eta", "barycenter_phi", "raw_energy", "vertices_indexes"]
CLUSTER_FEATURES = ["position_eta", "position_phi", "energy", "cluster_layer_id"]
def load_branch_with_highest_cycle(file, branch_name):
    try:
        all_keys = file.keys()
        matching_keys = [key for key in all_keys if key.startswith(branch_name)]
        if not matching_keys:
            return None
        highest_cycle_key = max(matching_keys, key=lambda key: int(key.split(";")[1]))
        return file[highest_cycle_key]
    except:
        return None

class SimpleTrackster:
    """Simple container for a trackster and its associated clusters"""
    __slots__ = ['eta', 'phi', 'energy', 'true_pid', 'true_energy', 'clusters']
    
    def __init__(self, eta, phi, energy, true_pid, true_energy, clusters):
        self.eta = float(eta)
        self.phi = float(phi)
        self.energy = float(energy)
        self.true_pid = int(true_pid)
        self.true_energy = float(true_energy)
        # clusters: list of [eta, phi, energy, layer]
        self.clusters = np.array(clusters, dtype=np.float32) if clusters else np.array([])
    
    def __len__(self):
        return len(self.clusters)
    
    def to_dict(self):
        return {
            'trackster': [self.eta, self.phi, self.energy],
            'true_pid': self.true_pid,
            'true_energy': self.true_energy,
            'clusters': self.clusters
        }
def process_root_file(file_path):
    try:
        #open ROOT file
        file = uproot.open(file_path)
        
        #load branches - if any are missing, skip this file
        tracksters_tree = load_branch_with_highest_cycle(file, 'ticlDumper/ticlCandidate')
        simcandidate_tree = load_branch_with_highest_cycle(file, 'ticlDumper/simTICLCandidate')
        associations_tree = load_branch_with_highest_cycle(file, 'ticlDumper/associations')
        clusters_tree = load_branch_with_highest_cycle(file, 'ticlDumper/clusters')
        
        #skip if any required tree is missing
        if any(t is None for t in [tracksters_tree, simcandidate_tree, associations_tree, clusters_tree]):
            print(f"  Skipping {osp.basename(file_path)}: missing required trees")
            return []
        
        #load data
        tracksters = tracksters_tree.arrays(TRACKSTER_FEATURES, library="ak")
        clusters = clusters_tree.arrays(CLUSTER_FEATURES, library="ak")
        
        #load association data
        assoc = associations_tree.arrays([
            'ticlCandidate_simToReco_CP_sharedE',
            'ticlCandidate_recoToSim_CP',
            'ticlCandidate_simToReco_CP_score',
            'ticlCandidate_recoToSim_CP_score',
            'ticlCandidate_simToReco_CP',
            'ticlCandidate_recoToSim_CP_sharedE',
        ], library="ak")
        
        #load simulation data
        sim_features = ["simTICLCandidate_regressed_energy", "simTICLCandidate_pdgId"]
        sim_candidates = simcandidate_tree.arrays(sim_features, library="ak")
        
        num_events = len(tracksters)
        valid_tracksters = []
        
        #process each event
        for event_idx in range(num_events):
            try:
                #get association data
                sim_to_reco_sharedE = assoc['ticlCandidate_simToReco_CP_sharedE'][event_idx]
                
                if sim_to_reco_sharedE is None or len(sim_to_reco_sharedE) == 0:
                    continue
                
                reco_to_sim_scores = assoc['ticlCandidate_recoToSim_CP_score'][event_idx]
                sim_to_reco_scores = assoc['ticlCandidate_simToReco_CP_score'][event_idx]
                sim_to_reco_index  = assoc['ticlCandidate_simToReco_CP'][event_idx]
                
                #iterate over simulated particles
                for sim_idx in range(len(sim_to_reco_sharedE)):
                    if len(sim_to_reco_sharedE[sim_idx]) == 0:
                        continue
                    
                    for idx, (trackster_idx, shared_energy) in enumerate(
                            zip(sim_to_reco_index[sim_idx], sim_to_reco_sharedE[sim_idx])):
                        
                        if shared_energy <= 0:
                            continue
                        
                        #apply selection criteria
                        reco_score = 1.0
                        if trackster_idx < len(reco_to_sim_scores) and len(reco_to_sim_scores[trackster_idx]) > 0:
                            reco_score = reco_to_sim_scores[trackster_idx][0]
                        
                        sim_score = 1.0
                        if sim_idx < len(sim_to_reco_scores) and idx < len(sim_to_reco_scores[sim_idx]):
                            sim_score = sim_to_reco_scores[sim_idx][idx]
                        
                        if reco_score > MAX_SCORE_RECO2SIM or sim_score > MAX_SCORE_SIM2RECO:
                            continue
                        
                        #get true values
                        true_energy = float(sim_candidates['simTICLCandidate_regressed_energy'][event_idx][sim_idx])
                        true_pid = abs(int(sim_candidates['simTICLCandidate_pdgId'][event_idx][sim_idx]))
                        
                        #get trackster data (eta, phi, energy)
                        ts_eta = abs(float(tracksters['barycenter_eta'][event_idx][trackster_idx]))
                        ts_phi = float(tracksters['barycenter_phi'][event_idx][trackster_idx])
                        ts_energy = float(tracksters['raw_energy'][event_idx][trackster_idx])
                        
                        #get associated layer clusters
                        vertex_indices = tracksters['vertices_indexes'][event_idx][trackster_idx]
                        
                        #handle different formats
                        if hasattr(vertex_indices, 'tolist'):
                            vertex_indices = vertex_indices.tolist()
                        
                        if not isinstance(vertex_indices, (list, tuple, np.ndarray)):
                            vertex_indices = [vertex_indices] if vertex_indices >= 0 else []
                        
                        #first, collect all clusters for this trackster
                        raw_clusters = []
                        for v_idx in vertex_indices:
                            if v_idx >= 0 and v_idx < len(clusters['energy'][event_idx]):
                                try:
                                    cluster_features = [
                                        abs(float(clusters['position_eta'][event_idx][v_idx])),
                                        float(clusters['position_phi'][event_idx][v_idx]),
                                        float(clusters['energy'][event_idx][v_idx]),
                                        float(clusters['cluster_layer_id'][event_idx][v_idx])
                                    ]
                                    raw_clusters.append(cluster_features)
                                except:
                                    continue
                        
                        #order clusters
                        #first by layer (ascending), then by energy (descending) within each layer
                        if raw_clusters:
                            clusters_array = np.array(raw_clusters)
                            
                            layers = np.unique(clusters_array[:, 3]) 
                            layers.sort()  # ascending order
                            
                            ordered_clusters = []
                            for layer in layers:
                                layer_mask = clusters_array[:, 3] == layer
                                layer_clusters = clusters_array[layer_mask]
                                
                                sorted_indices = np.argsort(layer_clusters[:, 2])[::-1]
                                layer_clusters_sorted = layer_clusters[sorted_indices]
                                
                                ordered_clusters.extend(layer_clusters_sorted.tolist())
                                
                            cluster_list = ordered_clusters
                        else:
                            cluster_list = []
    
                        #create trackster object
                        trackster = SimpleTrackster(
                            eta=ts_eta,
                            phi=ts_phi,
                            energy=ts_energy,
                            true_pid=true_pid,
                            true_energy=true_energy,
                            clusters=cluster_list
                        )
                        
                        valid_tracksters.append(trackster)
            
            except Exception as e:
                continue
        
        return valid_tracksters
        
    except Exception as e:
        print(f"  Error processing {osp.basename(file_path)}: {type(e).__name__}")
        return []  
    
def collect_input_files(input_paths, max_files=-1):
    all_root_files = []
    
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    
    for input_path in input_paths:
        if osp.isfile(input_path):
            if input_path.endswith('.root'):
                all_root_files.append(input_path)
            elif input_path.endswith('.txt'):
                try:
                    with open(input_path, 'r') as f:
                        files = [line.strip() for line in f 
                                if line.strip() and not line.startswith('#')]
                        all_root_files.extend([f for f in files if f.endswith('.root')])
                except:
                    print(f"Warning: Could not read {input_path}, skipping")
            else:
                print(f"Warning: Unknown file type {input_path}, skipping")
        elif osp.isdir(input_path):
            root_files = glob(osp.join(input_path, "*.root"))
            all_root_files.extend(root_files)
    
    seen = set()
    unique_files = []
    for f in all_root_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    
    if max_files > 0:
        unique_files = unique_files[:max_files]
    
    valid_files = []
    for f in unique_files:
        if osp.exists(f):
            valid_files.append(f)
        else:
            print(f"Warning: File not found, skipping: {f}")
    
    return valid_files

def save_tracksters_to_h5(tracksters_list, output_file, shuffle=True):
    if not tracksters_list:
        print("No tracksters to save!")
        return 0
    
    if shuffle:
        print("Shuffling tracksters...")
        random.shuffle(tracksters_list)
    
    print(f"Saving {len(tracksters_list)} tracksters to {output_file}...")
    
    with h5py.File(output_file, 'w') as f:
        for i, ts in enumerate(tqdm(tracksters_list, desc="Writing to HDF5")):
            grp = f.create_group(f"trackster_{i:08d}")
            
            #trackster features [eta, phi, energy]
            grp.create_dataset('features', data=[ts.eta, ts.phi, ts.energy], dtype='f4')
            
            #true values
            grp.create_dataset('true_pid', data=ts.true_pid, dtype='i4')
            grp.create_dataset('true_energy', data=ts.true_energy, dtype='f4')
            
            #associated clusters [N x 4] - each: [eta, phi, energy, layer]
            if len(ts.clusters) > 0:
                grp.create_dataset('clusters', data=ts.clusters, dtype='f4')
            
            #metadata
            grp.attrs['num_clusters'] = len(ts.clusters)
        
        #global metadata
        f.attrs['num_tracksters'] = len(tracksters_list)
        f.attrs['trackster_features'] = ['eta', 'phi', 'energy']
        f.attrs['cluster_features'] = ['eta', 'phi', 'energy', 'layer']
    
    return len(tracksters_list)

def process_files_parallel(input_files, num_workers=4):
    all_tracksters = []
    
    print(f"\nProcessing {len(input_files)} files with {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        #submit all jobs
        future_to_file = {executor.submit(process_root_file, f): f for f in input_files}
        
        #process as they complete
        with tqdm(total=len(input_files), desc="Processing files") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    tracksters = future.result()
                    all_tracksters.extend(tracksters)
                    pbar.set_postfix({"valid": len(tracksters), "file": osp.basename(file_path)[:20]})
                except Exception as e:
                    print(f"\nError processing {file_path}: {e}")
                pbar.update(1)
    
    return all_tracksters

def main():
    parser = argparse.ArgumentParser(description='Simplified preprocessing for TICL PID data')
    parser.add_argument('--input', '-i', type=str, nargs='+', required=True,
                       help='Input: .root file(s), .txt file(s) with file lists, or directory(ies)')
    parser.add_argument('--output', '-o', type=str, default='ticl_simplified.h5',
                       help='Output HDF5 file (default: ticl_simplified.h5)')
    parser.add_argument('--max-files', type=int, default=-1,
                       help='Maximum number of files to process (default: all)')
    parser.add_argument('--num-workers', type=int, default=1,
                       help='Number of worker processes (default: 1)')
    parser.add_argument('--no-shuffle', action='store_true',
                       help='Disable final shuffling of data')
    
    args = parser.parse_args()
    
    #collect all input files
    print("Collecting input files...")
    input_files = collect_input_files(args.input, args.max_files)
    
    if not input_files:
        print("No valid input files found!")
        return
    
    #print(f"Found {len(input_files)} valid ROOT files to process")
    
    # Process files in parallel
    all_tracksters = process_files_parallel(input_files, args.num_workers)
    
    #print(f"\nTotal valid tracksters found: {len(all_tracksters)}")
    
    if len(all_tracksters) == 0:
        print("No valid tracksters found. Exiting.")
        return
    
    # Calculate statistics
    total_clusters = sum(len(ts) for ts in all_tracksters)
    avg_clusters = total_clusters / len(all_tracksters) if all_tracksters else 0
    clusters_per_trackster = [len(ts) for ts in all_tracksters]
    
    #print(f"\nStatistics:")
    #print(f"  Total tracksters: {len(all_tracksters)}")
    #print(f"  Total clusters: {total_clusters}")
    #print(f"  Average clusters per trackster: {avg_clusters:.2f}")
    #print(f"  Min clusters: {min(clusters_per_trackster)}")
    #print(f"  Max clusters: {max(clusters_per_trackster)}")
    #print(f"  Median clusters: {np.median(clusters_per_trackster):.1f}")
    
    # Save to HDF5
    final_count = save_tracksters_to_h5(all_tracksters, args.output, shuffle=not args.no_shuffle)
    
    #print(f"Processing complete!")
    #print(f"  Input files: {len(input_files)}")
    #print(f"  Valid tracksters: {final_count}")
    #print(f"  Output file: {args.output}")

if __name__ == "__main__":
    main()
