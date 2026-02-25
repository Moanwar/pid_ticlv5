
import h5py, numpy as np

f = h5py.File('ticl_clu3d_data_pfn.h5', 'r')

print('=== Shapes ===')
print(f'  clusters   : {f[\"clusters\"].shape}')
print(f'  features   : {f[\"features\"].shape}')
print(f'  true_pid   : {f[\"true_pid\"].shape}')
print(f'  true_energy: {f[\"true_energy\"].shape}')

print()
print('=== Attributes ===')
print(f'  trackster_features: {list(f.attrs[\"trackster_features\"])}')
print(f'  cluster_features  : {list(f.attrs[\"cluster_features\"])}')

print()
print('=== Sanity checks ===')
clu  = f['clusters'][:]
feat = f['features'][:]
pid  = f['true_pid'][:]
en   = f['true_energy'][:]

print(f'  PIDs present       : {np.unique(pid)}')
print(f'  Energy range       : {en.min():.2f} {en.max():.2f} GeV')
print(f'  Trackster eta range: {feat[:,5].min():.3f} {feat[:,5].max():.3f}')
print(f'  Trackster phi range: {feat[:,6].min():.3f} {feat[:,6].max():.3f}')

print()
print('=== Cluster image checks ===')
# Fraction of empty layers per trackster
n_empty_layers = np.sum(clu[:,:,:,0].sum(axis=2) == 0, axis=1)
print(f'  Avg empty layers per trackster: {n_empty_layers.mean():.1f} / 50')
# Check no layer has more than 10 clusters
print(f'  Max clusters in any layer slot : {np.max((clu[:,:,:,0]>0).sum(axis=2))}  (should be <=10)')
# Check energy is non-negative
print(f'  Any negative cluster energy    : {(clu[:,:,:,0] < 0).any()}  (should be False)')
# Check padding is exactly zero
n_nonzero_after_pad = 0
for i in range(min(50, len(clu))):
    for l in range(50):
        row = clu[i,l,:,0]
        nz = np.sum(row > 0)
        if nz < 10 and row[nz:].sum() != 0:
            n_nonzero_after_pad += 1
print(f'  Padding violations (nonzero after last cluster): {n_nonzero_after_pad}  (should be 0)')

print()
print('=== Sample trackster 0 ===')
print(f'  features : {feat[0]}')
print(f'  pid      : {pid[0]}')
print(f'  energy   : {en[0]:.2f} GeV')
occupied = [(l, int((clu[0,l,:,0]>0).sum())) for l in range(50) if clu[0,l,:,0].sum()>0]
print(f'  occupied layers (layer, n_clusters): {occupied}')
f.close()

