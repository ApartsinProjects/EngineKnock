"""EXP-012: two ablations the full paper needs.
(A2) k-sensitivity: cond AUC vs k for the kNN-in-speed neighbourhood, both regimes, incl. the
     degenerate endpoints k=1 (noisy) and k=all (conditional mean -> global mean -> ~= baseline;
     an invariant check).
(A3) discrete-bin conditioning vs continuous: bin speed into {4,8,16} bins and subtract the
     per-bin normal mean (discrete conditioning), compared to continuous kNN-in-speed and to the
     unconditional baseline. Isolates 'conditioning at all' from 'CONTINUOUS conditioning', which
     is the paper's novelty claim.
MaFaulDa accel col1, pooled. Reuses mf_*.npz. Output: results/exp012_ablations.json.
"""
import os, json, numpy as np
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "..", "results")
FAULTS = ["horizontal-misalignment", "vertical-misalignment", "imbalance", "underhang", "overhang"]
BANDS = [(12, 20), (20, 28), (28, 36), (36, 44), (44, 52), (52, 61)]


def load(stem):
    d = np.load(os.path.join(OUT, f"mf_{stem}.npz")); return d["T"].astype(np.float32), d["S"].astype(np.float32)


def knn_mean(Xtr, st, sq, k):
    if k >= len(Xtr):
        return np.broadcast_to(Xtr.mean(0), (len(sq), Xtr.shape[1])).copy()
    o = np.argsort(st); ss = st[o]; XX = Xtr[o]; idx = np.searchsorted(ss, sq)
    out = np.empty((len(sq), Xtr.shape[1]), np.float32)
    for i, q in enumerate(sq):
        j = idx[i]; lo = max(0, j - k); hi = min(len(ss), j + k); c = np.arange(lo, hi)
        out[i] = XX[c[np.argsort(np.abs(ss[c] - q))[:k]]].mean(0)
    return out


def bin_mean(Xtr, st, sq, nbins, edges=None):
    """Discrete conditioning: subtract the mean of the training bin the query speed falls in.
    Bins are equal-width over the training speed range; empty-bin queries fall back to global mean."""
    if edges is None:
        edges = np.linspace(st.min(), st.max() + 1e-6, nbins + 1)
    glob = Xtr.mean(0)
    bmean = []
    for b in range(nbins):
        m = (st >= edges[b]) & (st < edges[b + 1])
        bmean.append(Xtr[m].mean(0) if m.sum() >= 3 else glob)
    bmean = np.array(bmean)
    qi = np.clip(np.digitize(sq, edges) - 1, 0, nbins - 1)
    return bmean[qi], edges


def pca(Xtr, k=32):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False); comp = Vt[:min(k, Xtr.shape[0] - 2)]
    return lambda X: np.sqrt((((X - mu) / sd - (((X - mu) / sd) @ comp.T) @ comp) ** 2).mean(1))


def auc(neg, pos):
    a = np.concatenate([neg, pos]); _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); av = (cs - c + cs + 1) / 2.0; r = av[inv]; n1 = len(pos)
    return float((r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (len(neg) * n1))


def bmask(S, b): return (S >= b[0]) & (S < b[1]) if b else np.ones(len(S), bool)


def matched_auc(Tn, Sn, faults, mean_fn, seeds=5):
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed); idx = rng.permutation(len(Tn)); cut = int(0.6 * len(Tn)); tr, te = idx[:cut], idx[cut:]
        if mean_fn is None:
            f = pca(Tn[tr]); neg = f(Tn[te]); pos = np.concatenate([f(Tf) for Tf, Sf in faults])
        else:
            mtr, st = mean_fn(Tn[tr], Sn[tr], Sn[tr]); f = pca(Tn[tr] - mtr)
            neg = f(Tn[te] - mean_fn(Tn[tr], Sn[tr], Sn[te])[0])
            pos = np.concatenate([f(Tf - mean_fn(Tn[tr], Sn[tr], Sf)[0]) for Tf, Sf in faults])
        out.append(auc(neg, pos))
    return float(np.mean(out))


def lobo_auc(Tn, Sn, faults, mean_fn):
    out = []
    for lo, hi in BANDS:
        ten = (Sn >= lo) & (Sn < hi); trn = ~ten
        if ten.sum() < 8 or trn.sum() < 40:
            continue
        Xtr, Str = Tn[trn], Sn[trn]
        if mean_fn is None:
            f = pca(Xtr); neg = f(Tn[ten])
            pos = np.concatenate([f(Tf[bmask(Sf, (lo, hi))]) for Tf, Sf in faults if bmask(Sf, (lo, hi)).sum()])
        else:
            f = pca(Xtr - mean_fn(Xtr, Str, Str)[0]); neg = f(Tn[ten] - mean_fn(Xtr, Str, Sn[ten])[0])
            pos = np.concatenate([f(Tf[bmask(Sf, (lo, hi))] - mean_fn(Xtr, Str, Sf[bmask(Sf, (lo, hi))])[0])
                                  for Tf, Sf in faults if bmask(Sf, (lo, hi)).sum()])
        out.append(auc(neg, pos))
    return float(np.mean(out))


# wrap mean fns to a common (Xtr,Str,Sq)->(mean, extra) signature
def mk_knn(k): return lambda X, s, q: (knn_mean(X, s, q, k), None)
def mk_bin(n): return lambda X, s, q: bin_mean(X, s, q, n)


def main():
    Tn, Sn = load("normal"); faults = [load(s) for s in FAULTS]
    R = {"k_sweep": {}, "discrete_vs_continuous": {}}

    # A2: k sweep
    for k in [1, 5, 10, 25, 50, 100, 250, 10**9]:
        lab = "all" if k > len(Tn) else str(k)
        R["k_sweep"][lab] = dict(matched=matched_auc(Tn, Sn, faults, mk_knn(k)),
                                 lobo=lobo_auc(Tn, Sn, faults, mk_knn(k)))
        print(f"k={lab:4s}: matched={R['k_sweep'][lab]['matched']:.3f}  lobo={R['k_sweep'][lab]['lobo']:.3f}")

    # A3: discrete bins vs continuous vs unconditional
    R["discrete_vs_continuous"]["unconditional"] = dict(matched=matched_auc(Tn, Sn, faults, None), lobo=lobo_auc(Tn, Sn, faults, None))
    for nb in [4, 8, 16]:
        R["discrete_vs_continuous"][f"bins_{nb}"] = dict(matched=matched_auc(Tn, Sn, faults, mk_bin(nb)), lobo=lobo_auc(Tn, Sn, faults, mk_bin(nb)))
    R["discrete_vs_continuous"]["continuous_knn15"] = dict(matched=matched_auc(Tn, Sn, faults, mk_knn(15)), lobo=lobo_auc(Tn, Sn, faults, mk_knn(15)))
    print("\nDiscrete-bin vs continuous conditioning:")
    for k, v in R["discrete_vs_continuous"].items():
        print(f"  {k:16s}: matched={v['matched']:.3f}  lobo={v['lobo']:.3f}")

    json.dump(R, open(os.path.join(OUT, "exp012_ablations.json"), "w"), indent=2)
    print("saved results/exp012_ablations.json")


if __name__ == "__main__":
    main()
