import h5py
import numpy as np
import sys

path = sys.argv[1]

with h5py.File(path, 'r') as f:
    print("=" * 60)
    print("FILE:", path)
    print("=" * 60)

    # Global attrs
    print("\n--- Attributes ---")
    for k, v in f.attrs.items():
        print(f"  {k}: {v}")

    # Datasets
    print("\n--- Datasets ---")
    for name in f.keys():
        ds = f[name]
        print(f"  {name:20s}  shape={ds.shape}  dtype={ds.dtype}")

    total = int(f.attrs.get('num_tracksters', 0))
    print(f"\n--- Spot checks (first 5 entries) ---")

    features    = f['features'][:5]
    true_pid    = f['true_pid'][:5]
    true_energy = f['true_energy'][:5]
    num_clusters= f['num_clusters'][:5]
    clusters    = f['clusters'][:5]
    print("clusters ", clusters)
    for i in range(5):
        print(f"\n  [{i}] features={features[i]}  pid={true_pid[i]}"
              f"  energy={true_energy[i]:.3f} "
              f"  n_clusters={num_clusters[i]}")
        clu = clusters[i].reshape(-1, 4) if len(clusters[i]) > 0 else []
        if len(clu):
            print(f"       clusters shape={np.array(clu).shape}  "
                  f"first={clu[0]}")

    print("\n--- Summary statistics ---")
    nc = f['num_clusters'][:]
    en = f['true_energy'][:]

    print(f"  num_clusters:  min={nc.min()}  max={nc.max()}  "
          f"mean={nc.mean():.1f}  median={np.median(nc):.1f}")
    print(f"  true_energy:   min={en.min():.3f}  max={en.max():.3f}  "
          f"mean={en.mean():.3f}")

    # Check for NaN/Inf
    feat = f['features'][:]
    nan_feat = np.isnan(feat).sum()
    inf_feat = np.isinf(feat).sum()
    nan_en   = np.isnan(en).sum()
    zero_en  = (en == 0).sum()

    print(f"\n--- Data quality ---")
    print(f"  NaN in features:    {nan_feat}")
    print(f"  Inf in features:    {inf_feat}")
    print(f"  NaN in true_energy: {nan_en}")
    print(f"  Zero true_energy:   {zero_en}")
    print(f"  Empty clusters:     {(nc == 0).sum()}")

    # PID distribution
    pids, counts = np.unique(f['true_pid'][:], return_counts=True)
    print(f"\n--- PID distribution ---")
    for pid, count in sorted(zip(pids, counts), key=lambda x: -x[1]):
        print(f"  pid={pid:6d}  count={count:8,}  ({100*count/total:.1f}%)")
