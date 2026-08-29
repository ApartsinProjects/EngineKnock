"""EXP-005c: stress-test the kNN-speed conditional detector before believing 0.961.
Checks: (a) 10 random matched-range split seeds -> mean +- std, not one lucky split;
(b) k sensitivity; (c) bootstrap CI + paired diff vs unconditional; (d) a leakage guard:
shuffle the fault SPEED labels used for detrending -> if AUC stays high, the signal is not
really speed-conditional (would indicate a confound). Reuses cache."""
import os, numpy as np
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "..", "results")


def pca_fit(X, k=32):
    mu = X.mean(0); sd = X.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
    return dict(mu=mu, sd=sd, comp=Vt[:min(k, X.shape[0] - 2)])


def pca_err(m, X):
    Xs = (X - m["mu"]) / m["sd"]
    return np.sqrt(((Xs - (Xs @ m["comp"].T) @ m["comp"]) ** 2).mean(1))


def auc(neg, pos):
    allv = np.concatenate([neg, pos]); _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0; r = avg[inv]; n1 = len(pos)
    return (r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (len(neg) * n1)


def knn_pred(Xtr, str_, sq, k=15):
    order = np.argsort(str_); ss = str_[order]; XX = Xtr[order]
    idx = np.searchsorted(ss, sq); out = np.empty((len(sq), Xtr.shape[1]))
    for i, q in enumerate(sq):
        j = idx[i]; lo = max(0, j - k); hi = min(len(ss), j + k)
        cand = np.arange(lo, hi)
        nn = cand[np.argsort(np.abs(ss[cand] - q))[:k]]
        out[i] = XX[nn].mean(0)
    return out


def cond_scores(Xtr, str_, Xn, sn, Xf, sf, k=15, K=32):
    Rtr = Xtr - knn_pred(Xtr, str_, str_, k)
    m = pca_fit(Rtr, K)
    s_n = pca_err(m, Xn - knn_pred(Xtr, str_, sn, k))
    s_f = pca_err(m, Xf - knn_pred(Xtr, str_, sf, k))
    return s_n, s_f


def main():
    cache = np.load(os.path.join(OUT, "cache_ch1_r16_s256.npz"), allow_pickle=True)["obj"].item()
    Xn, sn = cache["normal"]["time"]; Xf, sf = cache["fault"]["time"]

    print("### (a) 10-seed matched-range (k=15) ###")
    uncond, cond = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed); idx = rng.permutation(len(Xn)); cut = int(0.6 * len(Xn))
        tr, te = idx[:cut], idx[cut:]
        mu = pca_fit(Xn[tr]); uncond.append(auc(pca_err(mu, Xn[te]), pca_err(mu, Xf)))
        s_n, s_f = cond_scores(Xn[tr], sn[tr], Xn[te], sn[te], Xf, sf)
        cond.append(auc(s_n, s_f))
    uncond, cond = np.array(uncond), np.array(cond)
    print(f"  unconditional: {uncond.mean():.3f} +- {uncond.std():.3f}")
    print(f"  kNN-cond     : {cond.mean():.3f} +- {cond.std():.3f}")
    print(f"  paired diff  : {(cond-uncond).mean():+.3f} +- {(cond-uncond).std():.3f}  "
          f"(min per-seed {min(cond-uncond):+.3f})")

    print("\n### (b) k sensitivity (seed 0 matched-range) ###")
    rng = np.random.default_rng(0); idx = rng.permutation(len(Xn)); cut = int(0.6 * len(Xn))
    tr, te = idx[:cut], idx[cut:]
    for k in [5, 10, 15, 25, 40, 60]:
        s_n, s_f = cond_scores(Xn[tr], sn[tr], Xn[te], sn[te], Xf, sf, k=k)
        print(f"  k={k:3d}: AUC={auc(s_n, s_f):.3f}")

    print("\n### (c) bootstrap CI (seed 0, k=15) ###")
    s_n, s_f = cond_scores(Xn[tr], sn[tr], Xn[te], sn[te], Xf, sf, k=15)
    rng = np.random.default_rng(1); v = []
    for _ in range(2000):
        v.append(auc(s_n[rng.integers(0, len(s_n), len(s_n))], s_f[rng.integers(0, len(s_f), len(s_f))]))
    print(f"  kNN-cond AUC 95% CI [{np.percentile(v,2.5):.3f}, {np.percentile(v,97.5):.3f}]")

    print("\n### (d) leakage guard: detrend faults with SHUFFLED speed labels ###")
    rng = np.random.default_rng(2); sf_shuf = sf[rng.permutation(len(sf))]
    Rtr = Xn[tr] - knn_pred(Xn[tr], sn[tr], sn[tr], 15); m = pca_fit(Rtr)
    s_n = pca_err(m, Xn[te] - knn_pred(Xn[tr], sn[tr], sn[te], 15))
    s_f_shuf = pca_err(m, Xf - knn_pred(Xn[tr], sn[tr], sf_shuf, 15))
    print(f"  fault detrended with TRUE speed : AUC={auc(s_n, pca_err(m, Xf - knn_pred(Xn[tr], sn[tr], sf, 15))):.3f}")
    print(f"  fault detrended with SHUF speed : AUC={auc(s_n, s_f_shuf):.3f}  "
          f"(should DROP if the win is genuinely speed-conditional)")


if __name__ == "__main__":
    main()
