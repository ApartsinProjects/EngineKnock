"""EXP-005b: fix the speed-trend extrapolation. EXP-005 used a global cubic that blew up
beyond the training speed range (extreme held-out bands). Test robust conditional models
that extrapolate sanely:
  - poly_clamp: same cubic, but evaluate at speed CLAMPED to the training range (flat beyond).
  - knn_speed : conditional mean = average of the k nearest normal-train points IN SPEED
                (local, no global functional form; flat-ish beyond range via nearest).
Compare both + unconditional on matched-range and leave-one-band-out. Reuses cache.
"""
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


# ---- trend models: fit on (Xtr, str_), predict mean at query speeds ----
def poly_clamp_fit(X, s, deg=3):
    coef, *_ = np.linalg.lstsq(np.vander(s, deg + 1), X, rcond=None)
    return dict(coef=coef, lo=s.min(), hi=s.max(), deg=deg)


def poly_clamp_pred(m, s):
    sc = np.clip(s, m["lo"], m["hi"])
    return np.vander(sc, m["deg"] + 1) @ m["coef"]


def knn_fit(X, s, k=15):
    return dict(X=X, s=s, k=k)


def knn_pred(m, s):
    # for each query speed, mean of k nearest training speeds' features
    out = np.empty((len(s), m["X"].shape[1]))
    order = np.argsort(m["s"])
    ss = m["s"][order]; XX = m["X"][order]
    idx = np.searchsorted(ss, s)
    for i, q in enumerate(s):
        j = idx[i]; lo = max(0, j - m["k"]); hi = min(len(ss), j + m["k"])
        # pick k nearest by |speed diff|
        cand = np.arange(lo, hi)
        nn = cand[np.argsort(np.abs(ss[cand] - q))[:m["k"]]]
        out[i] = XX[nn].mean(0)
    return out


TRENDS = {
    "unconditional": None,
    "poly_clamp": (poly_clamp_fit, poly_clamp_pred),
    "knn_speed": (knn_fit, knn_pred),
}


def score(name, Xtr, str_, Xte_n, ste_n, Xte_f, ste_f, K=32):
    if name == "unconditional":
        m = pca_fit(Xtr, K)
        return auc(pca_err(m, Xte_n), pca_err(m, Xte_f))
    fit, pred = TRENDS[name]
    tm = fit(Xtr, str_)
    Rtr = Xtr - pred(tm, str_)
    m = pca_fit(Rtr, K)
    sn = pca_err(m, Xte_n - pred(tm, ste_n))
    sf = pca_err(m, Xte_f - pred(tm, ste_f))
    return auc(sn, sf)


def main():
    cache = np.load(os.path.join(OUT, "cache_ch1_r16_s256.npz"), allow_pickle=True)["obj"].item()
    Xn, sn = cache["normal"]["time"]; Xf, sf = cache["fault"]["time"]

    print("### MATCHED-RANGE (60/40) ###")
    rng = np.random.default_rng(0); idx = rng.permutation(len(Xn)); cut = int(0.6 * len(Xn))
    tr, te = idx[:cut], idx[cut:]
    for name in TRENDS:
        a = score(name, Xn[tr], sn[tr], Xn[te], sn[te], Xf, sf)
        print(f"[{name:14s}] AUC={a:.3f}")

    print("\n### LEAVE-ONE-BAND-OUT ###")
    bands = [(12, 20), (20, 28), (28, 36), (36, 44), (44, 52), (52, 61)]
    agg = {n: [] for n in TRENDS}
    for lo, hi in bands:
        te_n = (sn >= lo) & (sn < hi); tr_n = ~te_n; te_f = (sf >= lo) & (sf < hi)
        if te_n.sum() < 8 or te_f.sum() < 8 or tr_n.sum() < 40:
            continue
        line = f"  {lo}-{hi:>2} Hz:"
        for name in TRENDS:
            a = score(name, Xn[tr_n], sn[tr_n], Xn[te_n], sn[te_n], Xf[te_f], sf[te_f])
            agg[name].append(a); line += f"  {name}={a:.3f}"
        print(line)
    print("  MEAN:", "  ".join(f"{n}={np.mean(v):.3f}" for n, v in agg.items() if v))


if __name__ == "__main__":
    main()
