import h5py
p="./cache/all/precomp_rollout_mls.h5"
with h5py.File(p,"r") as f:
    g=f["t00001"]
    print("t00001 keys:", list(g.keys()))
    print("mls keys:", list(g["mls"].keys()))
    print("grad_dX shape:", g["mls"]["grad_dX"].shape)
