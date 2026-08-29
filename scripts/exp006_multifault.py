"""EXP-006: multi-fault-type external validity. For every MaFaulDa fault category present,
compare three normal-only detectors on accel col1:
  - baseline    : unconditional PCA on time-spectrum (the standard).
  - invariance  : PCA on ORDER-tracked spectrum (order-sync; the 'remove speed' competitor).
  - conditioning: kNN-in-speed conditional PCA on time-spectrum (OURS; 'model p(normal|speed)').
Protocols: matched-range (5-seed mean+-std) and leave-one-band-out (interpolation). Builds a
per-source feature cache (time + order spectra + speed), parsing each zip once.
"""
import io, os, zipfile
import numpy as np

FS = 50_000
HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data"); OUT = os.path.join(HERE, "..", "results")
ACC, REVS, SPER = 1, 16, 256
FAULTS = ["horizontal-misalignment", "vertical-misalignment", "imbalance", "underhang", "overhang"]


def tacho_rises(t):
    x = t.astype(float); x = (x - x.min()) / (np.ptp(x) + 1e-12); a = x > 0.5
    return np.where((~a[:-1]) & (a[1:]))[0]


def feats(sig, rises, revs=REVS, sper=SPER):
    if len(rises) < revs + 1:
        return
    L = revs * sper; nb = (len(rises) - 1) // revs; win = np.hanning(L)
    for b in range(nb):
        p0, p1 = rises[b * revs], rises[(b + 1) * revs]
        if p1 <= p0 or p1 > len(sig):
            continue
        seg = sig[p0:p1].astype(float)
        t = np.interp(np.linspace(0, len(seg) - 1, L), np.arange(len(seg)), seg)
        loc = rises[b * revs:(b + 1) * revs + 1]
        rev = np.interp(np.arange(p0, p1), loc, np.arange(len(loc)))
        grid = np.linspace(rev[0], rev[-1], L, endpoint=False)
        o = np.interp(grid, rev, seg)
        yield (np.log1p(np.abs(np.fft.rfft((t - t.mean()) * win))),
               np.log1p(np.abs(np.fft.rfft((o - o.mean()) * win))))


def build(zip_path):
    T, O, S = [], [], []
    with zipfile.ZipFile(zip_path) as zf:
        for n in sorted(x for x in zf.namelist() if x.lower().endswith(".csv")):
            arr = np.loadtxt(io.BytesIO(zf.read(n)), delimiter=",")
            r = tacho_rises(arr[:, 0]); frot = FS / np.median(np.diff(r)) if len(r) > 2 else np.nan
            for t, o in feats(arr[:, ACC], r):
                T.append(t); O.append(o); S.append(frot)
    return np.asarray(T), np.asarray(O), np.asarray(S)


def cache_for(stem):
    p = os.path.join(OUT, f"mf_{stem}.npz")
    if os.path.exists(p):
        d = np.load(p); return d["T"], d["O"], d["S"]
    zp = os.path.join(DATA, f"{stem}.zip")
    if not os.path.exists(zp):
        return None
    try:
        with zipfile.ZipFile(zp) as zf:
            if not any(x.lower().endswith(".csv") for x in zf.namelist()):
                return None
    except Exception:
        return None
    T, O, S = build(zp); np.savez_compressed(p, T=T, O=O, S=S); return T, O, S


def pca_fit(X, k=32):
    mu = X.mean(0); sd = X.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
    return dict(mu=mu, sd=sd, comp=Vt[:min(k, X.shape[0] - 2)])


def pe(m, X):
    Xs = (X - m["mu"]) / m["sd"]; return np.sqrt(((Xs - (Xs @ m["comp"].T) @ m["comp"]) ** 2).mean(1))


def knn(Xtr, st, sq, k=15):
    o = np.argsort(st); ss = st[o]; XX = Xtr[o]; idx = np.searchsorted(ss, sq); out = np.empty((len(sq), Xtr.shape[1]))
    for i, q in enumerate(sq):
        j = idx[i]; lo = max(0, j - k); hi = min(len(ss), j + k); c = np.arange(lo, hi)
        nn = c[np.argsort(np.abs(ss[c] - q))[:k]]; out[i] = XX[nn].mean(0)
    return out


def auc(neg, pos):
    a = np.concatenate([neg, pos]); _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); av = (cs - c + cs + 1) / 2.0; r = av[inv]; n1 = len(pos)
    return (r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (len(neg) * n1)


def main():
    Tn, On, Sn = cache_for_normal()
    print(f"normal windows={len(Tn)}  speed {Sn.min():.0f}-{Sn.max():.0f} Hz\n")
    bands = [(12, 20), (20, 28), (28, 36), (36, 44), (44, 52), (52, 61)]
    print(f"{'fault':22s} {'n':>5}  {'base':>12} {'invariance':>12} {'cond(ours)':>12}   {'LOBO base/inv/ours'}")
    pooled = {"T": [], "O": [], "S": []}
    for stem in FAULTS:
        c = cache_for(stem)
        if c is None:
            print(f"{stem:22s}  (not downloaded yet)"); continue
        Tf, Of, Sf = c
        pooled["T"].append(Tf); pooled["O"].append(Of); pooled["S"].append(Sf)
        rows = eval_fault(Tn, On, Sn, Tf, Of, Sf, bands)
        print_row(stem, len(Tf), rows)
    if pooled["T"]:
        Tf = np.vstack(pooled["T"]); Of = np.vstack(pooled["O"]); Sf = np.concatenate(pooled["S"])
        rows = eval_fault(Tn, On, Sn, Tf, Of, Sf, bands)
        print_row("POOLED", len(Tf), rows)


def eval_fault(Tn, On, Sn, Tf, Of, Sf, bands, K=32, k=15):
    # matched-range 5 seeds
    b, iv, co = [], [], []
    for seed in range(5):
        rng = np.random.default_rng(seed); idx = rng.permutation(len(Tn)); cut = int(0.6 * len(Tn))
        tr, te = idx[:cut], idx[cut:]
        mb = pca_fit(Tn[tr], K); b.append(auc(pe(mb, Tn[te]), pe(mb, Tf)))
        mo = pca_fit(On[tr], K); iv.append(auc(pe(mo, On[te]), pe(mo, Of)))
        Rtr = Tn[tr] - knn(Tn[tr], Sn[tr], Sn[tr], k); mc = pca_fit(Rtr, K)
        sc_n = pe(mc, Tn[te] - knn(Tn[tr], Sn[tr], Sn[te], k))
        sc_f = pe(mc, Tf - knn(Tn[tr], Sn[tr], Sf, k))
        co.append(auc(sc_n, sc_f))
    # LOBO
    lb, li, lc = [], [], []
    for lo, hi in bands:
        ten = (Sn >= lo) & (Sn < hi); trn = ~ten; tef = (Sf >= lo) & (Sf < hi)
        if ten.sum() < 8 or tef.sum() < 8 or trn.sum() < 40:
            continue
        mb = pca_fit(Tn[trn], K); lb.append(auc(pe(mb, Tn[ten]), pe(mb, Tf[tef])))
        mo = pca_fit(On[trn], K); li.append(auc(pe(mo, On[ten]), pe(mo, Of[tef])))
        Rtr = Tn[trn] - knn(Tn[trn], Sn[trn], Sn[trn], k); mc = pca_fit(Rtr, K)
        lc.append(auc(pe(mc, Tn[ten] - knn(Tn[trn], Sn[trn], Sn[ten], k)),
                      pe(mc, Tf[tef] - knn(Tn[trn], Sn[trn], Sf[tef], k))))
    return dict(b=np.array(b), iv=np.array(iv), co=np.array(co),
                lb=np.mean(lb) if lb else np.nan, li=np.mean(li) if li else np.nan, lc=np.mean(lc) if lc else np.nan)


def print_row(stem, n, r):
    print(f"{stem:22s} {n:5d}  {r['b'].mean():.3f}+-{r['b'].std():.2f} {r['iv'].mean():.3f}+-{r['iv'].std():.2f} "
          f"{r['co'].mean():.3f}+-{r['co'].std():.2f}   {r['lb']:.3f}/{r['li']:.3f}/{r['lc']:.3f}")


def cache_for_normal():
    # build normal cache with time+order
    p = os.path.join(OUT, "mf_normal.npz")
    if os.path.exists(p):
        d = np.load(p); return d["T"], d["O"], d["S"]
    T, O, S = build(os.path.join(DATA, "normal.zip")); np.savez_compressed(p, T=T, O=O, S=S); return T, O, S


if __name__ == "__main__":
    main()
