"""
# Plot one example for each major particle type
python visualize_graphs.py --input ticl_clu3d_data.h5

# Plot specific PIDs
python visualize_graphs.py --input ticl_clu3d_data.h5 --pid 22 11 13 211

# Plot by particle groups (Photon, Electron, etc.)
python visualize_graphs.py --input ticl_clu3d_data.h5 --by-group

# Plot 3 examples per type
python visualize_graphs.py --input ticl_clu3d_data.h5 --num-examples 3

# Custom output directory
python visualize_graphs.py --input ticl_clu3d_data.h5 --output-dir my_plots
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from collections import defaultdict
import random

# LABEL MAPPING (for reference)

particle_names = {
    22: 'Photon',
    11: 'Electron', -11: 'Electron',
    13: 'Muon', -13: 'Muon',
    111: 'Pion0',
    211: 'Charged Hadron', -211: 'Charged Hadron', 
    321: 'Charged Hadron', -321: 'Charged Hadron',
    310: 'Neutral Hadron', 130: 'Neutral Hadron',
    -1: 'Unknown',
    0: 'Noise'
}

# Group by absolute PID for plotting
particle_groups = {
    'Photon': [22],
    'Electron': [11],
    'Muon': [13],
    'Pion0': [111],
    'Charged_Hadron': [211, 321],
    'Neutral_Hadron': [310, 130],
    'Other': [-1, 0]
}

# Colors for different particle types
type_colors = {
    'Photon': 'gold',
    'Electron': 'blue',
    'Muon': 'red',
    'Pion0': 'green',
    'Charged_Hadron': 'purple',
    'Neutral_Hadron': 'orange',
    'Other': 'gray'
}

def get_particle_group(pid):
    """Get group name for a PID"""
    abs_pid = abs(pid)
    for group_name, pid_list in particle_groups.items():
        if abs_pid in pid_list:
            return group_name
    return 'Other'

def find_trackster_indices_by_pid(h5_file, target_pid, max_clusters=20, target_energy=None, energy_tolerance=5.0):
    indices = []
    
    with h5py.File(h5_file, 'r') as f:
        num_tracksters = f.attrs['num_tracksters']
        pids = f['true_pid'][:]
        energies = f['true_energy'][:]
        num_clusters = f['num_clusters'][:]
        
        abs_target = abs(target_pid)
        
        for i in range(num_tracksters):
            pid = abs(pids[i])
            
            # Check PID match
            if pid != abs_target:
                continue
            
            # Check cluster count
            if num_clusters[i] > max_clusters:
                continue
            
            # Check energy if specified
            if target_energy is not None:
                if energies[i] < target_energy:
                    continue
            
            indices.append(i)
            
            # Optional: limit if needed (caller can slice)
    
    return indices

def find_trackster_indices_by_group(h5_file, group_name, max_clusters=20, target_energy=None, energy_tolerance=5.0, max_per_type=10):
    if group_name not in particle_groups:
        return []
    
    pid_list = particle_groups[group_name]
    indices = []
    
    with h5py.File(h5_file, 'r') as f:
        num_tracksters = f.attrs['num_tracksters']
        pids = f['true_pid'][:]
        energies = f['true_energy'][:]
        num_clusters = f['num_clusters'][:]
        
        for i in range(num_tracksters):
            pid = abs(pids[i])
            
            # Check if PID belongs to this group
            if pid not in pid_list:
                continue
            
            # Check cluster count
            if num_clusters[i] > max_clusters:
                continue
            
            # Check energy if specified
            if target_energy is not None:
                if energies[i] < target_energy :
                    continue
            
            indices.append(i)
            
            if len(indices) >= max_per_type:
                break
    
    return indices

def get_trackster_data(h5_file, idx):
    """Get all data for a specific trackster by index"""
    with h5py.File(h5_file, 'r') as f:
        # Get trackster features
        features = f['features'][idx]  # [eta, phi, energy]
        
        # Get labels
        pid = f['true_pid'][idx]
        true_energy = f['true_energy'][idx]
        
        # Get clusters - stored as flat array, need reshaping
        clusters_flat = f['clusters'][idx]
        n_clusters = f['num_clusters'][idx]
        
        if n_clusters > 0:
            clusters = clusters_flat.reshape(-1, 4)  # [N, 4]
        else:
            clusters = np.zeros((0, 4), dtype=np.float32)
    
    return {
        'features': features,
        'pid': pid,
        'true_energy': true_energy,
        'clusters': clusters,
        'n_clusters': n_clusters,
        'index': idx
    }

def build_edges_from_clusters(clusters):
    """
    Build edges based on layer structure:
    - within same layer: fully connected
    - between consecutive layers: fully connected
    """
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
    """
    Create 3D plot with swapped axes:
    X = phi, Y = layer, Z = eta
    """
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
    
    # Plot only inter-layer edges
    for u, v in edges:
        if abs(layer[u] - layer[v]) > 0.1:  
            ax.plot([x_coords[u], x_coords[v]], 
                    [y_coords[u], y_coords[v]], 
                    [z_coords[u], z_coords[v]], 
                    'gray', alpha=0.3, linewidth=1.0)
    
    # Move z-axis to left
    ax.zaxis._axinfo['juggled'] = (2, 0, 1)  
    ax.zaxis.set_ticks_position('lower')  
    
    # Set view angle
    ax.view_init(elev=20, azim=-45)
    
    # Labels
    ax.set_xlabel('φ', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Layer', fontsize=12, fontweight='bold', labelpad=10)
    ax.text2D(0.005, 0.5, 'η', transform=ax.transAxes, 
              fontsize=12, fontweight='bold', va='center', ha='center')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label('Energy (GeV)', fontsize=10)
    
    # Set axis limits
    if len(phi) > 0:
        ax.set_xlim(phi.min() - 0.1, phi.max() + 0.1)
    if len(layer) > 0:
        ax.set_ylim(layer.min() - 1, layer.max() + 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")

def plot_multiple_tracksters(h5_file, pid_values, num_examples=1, output_dir='plots', 
                             max_clusters=20, target_energy=20.0, energy_tolerance=5.0):
    """
    Plot multiple tracksters for given PID values.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_file, 'r') as f:
        num_tracksters = f.attrs['num_tracksters']
        print(f"Total tracksters in file: {num_tracksters:,}")
        
        # Get all PIDs present
        all_pids = np.unique(np.abs(f['true_pid'][:]))
        print(f"PIDs present: {sorted(all_pids)}")
        
        # Plot for requested PID values
        for pid in pid_values:
            print(f"\n{'='*60}")
            print(f"Looking for PID={pid} ({particle_names.get(pid, 'Unknown')})")
            
            # Find indices for this PID
            indices = find_trackster_indices_by_pid(
                h5_file, pid, 
                max_clusters=max_clusters,
                target_energy=target_energy,
                energy_tolerance=energy_tolerance
            )
            
            if not indices:
                print(f"  No tracksters found matching criteria")
                continue
            
            print(f"  Found {len(indices)} tracksters")
            
            # Select random examples
            selected = random.sample(indices, min(num_examples, len(indices)))
            
            for j, idx in enumerate(selected):
                # Get trackster data
                data = get_trackster_data(h5_file, idx)
                clusters = data['clusters']
                true_energy = data['true_energy']
                actual_pid = data['pid']
                
                # Build edges
                edges = build_edges_from_clusters(clusters)
                
                # Create plot
                title = f"PID={actual_pid} ({particle_names.get(abs(actual_pid), 'Unknown')})\n"
                title += f"True Energy: {true_energy:.2f} GeV | {len(clusters)} clusters"
                
                output_file = os.path.join(output_dir, f"pid_{pid}_example_{j+1}.png")
                
                plot_trackster_3d(clusters, edges, title, output_file)

def plot_by_particle_group(h5_file, num_examples=1, output_dir='plots',
                           max_clusters=20, target_energy=20.0, energy_tolerance=5.0):
    """
    Plot one example for each particle group.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_file, 'r') as f:
        num_tracksters = f.attrs['num_tracksters']
        print(f"Total tracksters in file: {num_tracksters:,}")
        
        # Get stats per group
        pids = f['true_pid'][:]
        energies = f['true_energy'][:]
        n_clusters = f['num_clusters'][:]
        
        group_counts = defaultdict(int)
        for pid in pids:
            group = get_particle_group(pid)
            group_counts[group] += 1
        
        print("\nTracksters by particle group:")
        for group, count in sorted(group_counts.items()):
            if count > 0:
                print(f"  {group}: {count:,} tracksters")
        
        # Plot for each group
        for group_name in particle_groups.keys():
            print(f"\n{'='*60}")
            print(f"Plotting {group_name}...")
            
            # Find indices for this group
            indices = find_trackster_indices_by_group(
                h5_file, group_name,
                max_clusters=max_clusters,
                target_energy=target_energy,
                energy_tolerance=energy_tolerance,
                max_per_type=num_examples * 5  # Get more than needed for random selection
            )
            
            if not indices:
                print(f"  No tracksters found for {group_name}")
                continue
            
            print(f"  Found {len(indices)} tracksters matching criteria")
            
            # Select random examples
            selected = random.sample(indices, min(num_examples, len(indices)))
            
            for j, idx in enumerate(selected):
                # Get trackster data
                data = get_trackster_data(h5_file, idx)
                clusters = data['clusters']
                pid = data['pid']
                true_energy = data['true_energy']
                
                # Build edges
                edges = build_edges_from_clusters(clusters)
                
                # Create plot
                title = f"{group_name} (PID={pid})\n"
                title += f"True Energy: {true_energy:.2f} GeV | {len(clusters)} clusters"
                
                output_file = os.path.join(output_dir, f"{group_name.lower()}_example_{j+1}.png")
                
                plot_trackster_3d(clusters, edges, title, output_file)

def main():
    parser = argparse.ArgumentParser(description='Visualize trackster graphs in 3D')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input HDF5 file (columnar format)')
    parser.add_argument('--output-dir', '-o', type=str, default='graph_plots',
                       help='Output directory for plots')
    parser.add_argument('--pid', type=int, nargs='+', default=None,
                       help='Specific PID values to plot (e.g., --pid 22 11 13)')
    parser.add_argument('--num-examples', '-n', type=int, default=1,
                       help='Number of examples per PID/type')
    parser.add_argument('--by-group', action='store_true',
                       help='Plot by particle groups instead of individual PIDs')
    parser.add_argument('--max-clusters', type=int, default=20,
                       help='Max number of clusters per trackster')
    parser.add_argument('--target-energy', type=float, default=20.0,
                       help='Target energy for filtering (GeV)')
    parser.add_argument('--energy-tolerance', type=float, default=5.0,
                       help='Energy tolerance for filtering')
    
    args = parser.parse_args()
    
    if args.by_group:
        plot_by_particle_group(
            args.input, 
            args.num_examples, 
            args.output_dir,
            max_clusters=args.max_clusters,
            target_energy=args.target_energy,
            energy_tolerance=args.energy_tolerance
        )
    elif args.pid:
        plot_multiple_tracksters(
            args.input, 
            args.pid, 
            args.num_examples, 
            args.output_dir,
            max_clusters=args.max_clusters,
            target_energy=args.target_energy,
            energy_tolerance=args.energy_tolerance
        )
    else:
        # Default: plot all major particle types
        default_pids = [22, 11, 13, 111, 211, 310]
        plot_multiple_tracksters(
            args.input, 
            default_pids, 
            args.num_examples, 
            args.output_dir,
            max_clusters=args.max_clusters,
            target_energy=args.target_energy,
            energy_tolerance=args.energy_tolerance
        )

if __name__ == "__main__":
    main()
