"""
# Plot one example for each major particle type
python visualize_graphs.py --input ticl_simplified.h5

# Plot specific PIDs
python visualize_graphs.py --input ticl_simplified.h5 --pid 22 11 13 211

# Plot by particle groups (Photon, Electron, etc.)
python visualize_graphs.py --input ticl_simplified.h5 --by-group

# Plot 3 examples per type
python visualize_graphs.py --input ticl_simplified.h5 --num-examples 3

# Custom output directory
python visualize_graphs.py --input ticl_simplified.h5 --output-dir my_plots
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from collections import defaultdict
import random

# ============================================
# LABEL MAPPING (for reference)
# ============================================

particle_names = {
    22: 'Photon',
    11: 'Electron',
    13: 'Muon', 
    111: 'pion0',
    211: 'Charged Hadron', 321: 'Charged Hadron', 
    310: 'Neutral Hadron', 130: 'Neutral Hadron',
    -1: 'Unknown',
    0: 'Noise'
}

# Group by absolute PID for plotting
particle_groups = {
    'Photon': [22],
    'Electron': [11],
    'Muon': [13],
    'pion0': [111],
    'Charged Hadron': [211, 321],
    'Neutral Hadron': [310, 130],
    'Other': [-1, 0]
}

# Colors for different particle types
type_colors = {
    'Photon': 'gold',
    'Electron': 'blue',
    'Muon': 'red',
    'pion0': 'green',
    'Charged Hadron': 'purple',
    'Neutral Hadron': 'orange',
    'Other': 'gray'
}

def find_tracksters_by_pid(h5_file, target_pid, max_per_type=5):
    """
    Find tracksters with given PID in the HDF5 file.
    Returns list of indices.
    """
    indices = []
    
    with h5py.File(h5_file, 'r') as f:
        num_tracksters = f.attrs['num_tracksters']
        
        for i in range(num_tracksters):
            grp = f[f'trackster_{i:08d}']
            pid = grp['true_pid'][()]
            
            # Check if PID matches (consider absolute value for grouping)
            abs_pid = abs(pid)
            
            # Determine which group this PID belongs to
            for group_name, pid_list in particle_groups.items():
                if abs_pid in pid_list or pid in pid_list:
                    if group_name == target_pid:  # target_pid is actually group name here
                        indices.append(i)
                        break
            
            if len(indices) >= max_per_type:
                break
    
    return indices

def find_trackster_by_abs_pid(h5_file, abs_pid, min_clusters=5):
    """
    Find a trackster with given absolute PID that has at least min_clusters.
    Returns index or None.
    """
    with h5py.File(h5_file, 'r') as f:
        num_tracksters = f.attrs['num_tracksters']
        
        for i in range(num_tracksters):
            grp = f[f'trackster_{i:08d}']
            pid = abs(grp['true_pid'][()])
            
            if pid == abs_pid and 'clusters' in grp:
                clusters = grp['clusters'][:]
                if len(clusters) >= min_clusters:
                    return i
    
    return None

def build_edges_from_clusters(clusters):

    # build edges based on layer structure:
    # within same layer: fully connected
    # between consecutive layers: fully connected

    if len(clusters) == 0:
        return []
    
    layers = clusters[:, 3].astype(int)
    unique_layers = np.unique(layers)
    
    edges = []
    node_indices = np.arange(len(clusters))
    
    for i, layer in enumerate(unique_layers):
        # Nodes in current layer
        nodes_in_layer = node_indices[layers == layer]
        
        # Within-layer connections
        for u in nodes_in_layer:
            for v in nodes_in_layer:
                if u < v:  # Add each edge only once (undirected)
                    edges.append((u, v))
        
        # Connections to next layer
        if i < len(unique_layers) - 1:
            next_layer = unique_layers[i + 1]
            nodes_in_next = node_indices[layers == next_layer]
            
            for u in nodes_in_layer:
                for v in nodes_in_next:
                    edges.append((u, v))
    
    return edges
def plot_trackster_3d(clusters, edges, title, output_file, highlight_energy=True):
    #Create 3D plot with swapped axes:

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    eta = clusters[:, 0]
    phi = clusters[:, 1]
    energy = clusters[:, 2]
    layer = clusters[:, 3]
    
    x_coords = phi
    y_coords = layer
    z_coords = eta
    
    sizes = energy * 30
    sizes = np.clip(sizes, 20, 200)
    
    scatter = ax.scatter(x_coords, y_coords, z_coords, 
                        c=energy, 
                        s=sizes,
                        cmap='hot',
                        alpha=0.9, 
                        edgecolors='black', 
                        linewidth=0.8)
    
    for u, v in edges:
        if abs(layer[u] - layer[v]) > 0.1:  
            ax.plot([x_coords[u], x_coords[v]], 
                    [y_coords[u], y_coords[v]], 
                    [z_coords[u], z_coords[v]], 
                    'gray', alpha=0.3, linewidth=1.0)
    
    ax.zaxis._axinfo['juggled'] = (2, 0, 1)  
    ax.zaxis.set_ticks_position('lower')  
    
    #ax.view_init(elev=15, azim=-60)
    ax.view_init(elev=20, azim=-45)
    #ax.view_init(elev=0, azim=-75)
    #ax.view_init(elev=10, azim=-80)
    #ax.view_init(elev=20, azim=45)
    
    ax.set_xlabel('phi', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Layer', fontsize=12, fontweight='bold', labelpad=10)
    #ax.set_zlabel('eta', fontsize=12, fontweight='bold', labelpad=10)
    ax.text2D(0.0005, 0.5, 'eta', transform=ax.transAxes, 
              fontsize=12, fontweight='bold', va='center', ha='center')


    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label('Energy (GeV)', fontsize=10)
    
    ax.set_xlim(phi.min() - 0.1, phi.max() + 0.1)
    ax.set_ylim(layer.min() - 1, layer.max() + 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"  Saved: {output_file}")


def plot_multiple_tracksters(h5_file, pid_values, num_examples=1, output_dir='plots'):
    #plot multiple tracksters for given PID values.
    
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_file, 'r') as f:
        num_tracksters = f.attrs['num_tracksters']
        print(f"Total tracksters in file: {num_tracksters}")
        
        # Group tracksters by PID
        pid_to_indices = defaultdict(list)
        target_energy = 20.0
        energy_tolerance = 5.0
        max_clusters = 20
        for i in range(num_tracksters):
            grp = f[f'trackster_{i:08d}']
            pid = grp['true_pid'][()]
            abs_pid = abs(pid)
            true_energy = grp['true_energy'][()]
            if true_energy < target_energy :
                continue
            clusters = grp['clusters'][:]
            n_clusters = len(clusters)
            if n_clusters > max_clusters:
                continue
            
            if 'clusters' in grp and len(grp['clusters'][:]) > 0:
                pid_to_indices[abs_pid].append(i)
        
        print(f"Found tracksters for PIDs: {sorted(pid_to_indices.keys())}")
        
        # Plot for requested PID values
        for pid in pid_values:
            if pid not in pid_to_indices:
                print(f"Warning: No tracksters found for PID={pid}")
                continue
            
            indices = pid_to_indices[pid]
            print(f"\nPID {pid} ({particle_names.get(pid, 'Unknown')}): {len(indices)} tracksters")
            
            # Select random examples
            selected = random.sample(indices, min(num_examples, len(indices)))
            
            for j, idx in enumerate(selected):
                grp = f[f'trackster_{idx:08d}']
                clusters = grp['clusters'][:]
                true_energy = grp['true_energy'][()]
                
                # Build edges
                edges = build_edges_from_clusters(clusters)
                
                # Create plot
                title = f"PID={pid} ({particle_names.get(pid, 'Unknown')})\n"
                title += f"True Energy: {true_energy:.2f} GeV | {len(clusters)} clusters"
                
                output_file = os.path.join(output_dir, f"pid_{pid}_example_{j+1}.png")
                
                plot_trackster_3d(clusters, edges, title, output_file)
#Plot one example for each particle group.
def plot_by_particle_group(h5_file, num_examples=1, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_file, 'r') as f:
        num_tracksters = f.attrs['num_tracksters']
        
        # Group tracksters by particle type
        type_to_indices = defaultdict(list)
        target_energy = 20.0
        energy_tolerance = 5.0
        max_clusters = 20
        for i in range(num_tracksters):
            grp = f[f'trackster_{i:08d}']
            pid = grp['true_pid'][()]
            abs_pid = abs(pid)
            if true_energy < target_energy :
                continue
            clusters = grp['clusters'][:]
            n_clusters = len(clusters)
            if n_clusters > max_clusters:
                continue

            # Determine particle type
            for group_name, pid_list in particle_groups.items():
                if abs_pid in pid_list or pid in pid_list:
                    if 'clusters' in grp and len(grp['clusters'][:]) > 0:
                        type_to_indices[group_name].append(i)
                    break
        
        print("\nTracksters by particle type:")
        for group_name, indices in type_to_indices.items():
            if indices:
                print(f"  {group_name}: {len(indices)} tracksters")
        
        # Plot one example for each type that has tracksters
        for group_name, indices in type_to_indices.items():
            if not indices:
                continue
            
            print(f"\nPlotting {group_name}...")
            
            # Select random examples
            selected = random.sample(indices, min(num_examples, len(indices)))
            
            for j, idx in enumerate(selected):
                grp = f[f'trackster_{idx:08d}']
                clusters = grp['clusters'][:]
                pid = grp['true_pid'][()]
                true_energy = grp['true_energy'][()]
                
                # Build edges
                edges = build_edges_from_clusters(clusters)
                
                # Create plot
                title = f"{group_name} (PID={pid})\n"
                title += f"True Energy: {true_energy:.2f} GeV | {len(clusters)} clusters"
                
                output_file = os.path.join(output_dir, f"{group_name.lower().replace(' ', '_')}_example_{j+1}.png")
                
                plot_trackster_3d(clusters, edges, title, output_file, highlight_energy=True)

def main():
    parser = argparse.ArgumentParser(description='Visualize trackster graphs in 3D')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input HDF5 file')
    parser.add_argument('--output-dir', '-o', type=str, default='graph_plots',
                       help='Output directory for plots')
    parser.add_argument('--pid', type=int, nargs='+', default=None,
                       help='Specific PID values to plot (e.g., --pid 22 11 13)')
    parser.add_argument('--num-examples', '-n', type=int, default=1,
                       help='Number of examples per PID/type')
    parser.add_argument('--by-group', action='store_true',
                       help='Plot by particle groups instead of individual PIDs')
    
    args = parser.parse_args()
    
    if args.by_group:
        plot_by_particle_group(args.input, args.num_examples, args.output_dir)
    elif args.pid:
        plot_multiple_tracksters(args.input, args.pid, args.num_examples, args.output_dir)
    else:
        # Default: plot all major particle types
        default_pids = [22, 11, 13, 111, 211, 310]
        plot_multiple_tracksters(args.input, default_pids, args.num_examples, args.output_dir)

if __name__ == "__main__":
    main()
