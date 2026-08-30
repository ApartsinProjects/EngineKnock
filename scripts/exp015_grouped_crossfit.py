"""EXP-015: leakage-free, cross-fitted evaluation (addresses review W3/W4/W5).
MaFaulDa healthy is 49 recordings at ~49 distinct speeds, so the window-level matched-speed
split leaks (train/test share recordings and the kNN mean uses same-recording windows). Here we:
 (a) re-extract accel col1 with per-window RECORDING IDs (source file index);
 (b) LEAVE-ONE-SPEED-OUT (LOSO): for each held-out healthy recording, train the detector on the
     OTHER healthy recordings, cross-fitting the conditional mean (the held-out recording never
     contributes to its own m(s)); test held-out-normal vs all fault windows;
 (c) report base/inv/cond mean AUC over the 49 held-out recordings, plus interpolation vs
     extrapolation (whether the held-out speed is inside the training speed range);
 (d) validate AUC against scikit-learn.
Output: results/exp015_grouped.json.
"""
import io, os, json, zipfile, numpy as np
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


def build(stem, rec_offset):
    T, O, S, R = [], [], [], []
    with zipfile.ZipFile(os.path.join(DATA, f"{stem}.zip")) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        for ri, n in enumerate(names):
            arr = np.loadtxt(io.BytesIO(zf.read(n)), delimiter=",")
            rises = tacho_rises(arr[:, 0])
            for t, o in feats(arr[:, ACC], rises):
                T.append(t); O.append(o); S.append(FS / np.median(np.diff(rises))); R.append(rec_offset + ri)
    return np.asarray(T, np.float32), np.asarray(O, np.float32), np.asarray(S, np.float32), np.asarray(R), len(names)


def get_cache():
    p = os.path.join(OUT, "grp_cache.npz")
    if os.path.exists(p):
        d = np.load(p)
        return {k: d[k] for k in d.files}
    Tn, On, Sn, Rn, nrec = build("normal", 0)
    Tf, Of, Sf = [], [], []
    off = 1000
    for stem in FAULTS:
        t, o, s, r, k = build(stem, off); off += 1000
        Tf.append(t); Of.append(o); Sf.append(s)
    Tf = np.vstack(Tf); Of = np.vstack(Of); Sf = np.concatenate(Sf)
    obj = dict(Tn=Tn, On=On, Sn=Sn, Rn=Rn, Tf=Tf, Of=Of, Sf=Sf)
    np.savez_compressed(p, **obj); return obj


def knn_mean(Xtr, st, sq, k=15):
    o = np.argsort(st); ss = st[o]; XX = Xtr[o]; idx = np.searchsorted(ss, sq)
    out = np.empty((len(sq), Xtr.shape[1]), np.float32)
    for i, q in enumerate(sq):
        j = idx[i]; lo = max(0, j - k); hi = min(len(ss), j + k); c = np.arange(lo, hi)
        out[i] = XX[c[np.argsort(np.abs(ss[c] - q))[:k]]].mean(0)
    return out


def pca(Xtr, k=32):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False); comp = Vt[:min(k, Xtr.shape[0] - 2)]
    return lambda X: np.sqrt((((X - mu) / sd - (((X - mu) / sd) @ comp.T) @ comp) ** 2).mean(1))


def auc(neg, pos):
    a = np.concatenate([neg, pos]); _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); av = (cs - c + cs + 1) / 2.0; r = av[inv]; n1 = len(pos)
    return float((r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (len(neg) * n1))


def main():
    C = get_cache()
    Tn, On, Sn, Rn = C["Tn"], C["On"], C["Sn"], C["Rn"]
    Tf, Of, Sf = C["Tf"], C["Of"], C["Sf"]
    recs = np.unique(Rn)
    print(f"healthy windows={len(Tn)} across {len(recs)} recordings; fault windows={len(Tf)}")
    # sklearn AUC validation on one case
    try:
        from sklearn.metrics import roc_auc_score
        a, b = np.random.default_rng(0).normal(0, 1, 200), np.random.default_rng(1).normal(0.5, 1, 200)
        mine = auc(a, b); sk = roc_auc_score([0]*200+[1]*200, np.concatenate([a, b]))
        print(f"AUC self-check vs sklearn: mine={mine:.4f} sklearn={sk:.4f} (match={abs(mine-sk)<1e-6})")
    except Exception as e:
        print("sklearn check skipped:", e)

    res = {t: [] for t in ("base", "inv", "cond")}
    interp = {t: [] for t in ("base", "inv", "cond")}; extrap = {t: [] for t in ("base", "inv", "cond")}
    for rid in recs:
        te = Rn == rid; tr = ~te
        if te.sum() < 5 or tr.sum() < 40:
            continue
        s_te = Sn[te]; s_lo, s_hi = Sn[tr].min(), Sn[tr].max()
        is_interp = (s_te.mean() >= s_lo) and (s_te.mean() <= s_hi)
        # base (time PCA), inv (order PCA), cond (cross-fitted kNN-in-speed on time)
        fb = pca(Tn[tr]); fo = pca(On[tr])
        mtr = knn_mean(Tn[tr], Sn[tr], Sn[tr]); fc = pca(Tn[tr] - mtr)
        scores = {
            "base": (fb(Tn[te]), fb(Tf)),
            "inv": (fo(On[te]), fo(Of)),
            "cond": (fc(Tn[te] - knn_mean(Tn[tr], Sn[tr], Sn[te])), fc(Tf - knn_mean(Tn[tr], Sn[tr], Sf))),
        }
        for t in ("base", "inv", "cond"):
            a = auc(*scores[t]); res[t].append(a)
            (interp if is_interp else extrap)[t].append(a)
    out = {}
    print("\n### LOSO (leave-one-speed-out, cross-fitted) mean AUC over held-out recordings ###")
    for t in ("base", "inv", "cond"):
        v = np.array(res[t]); out[t] = dict(mean=float(v.mean()), std=float(v.std()), n=len(v))
        print(f"  {t:5s}: {v.mean():.3f} ± {v.std():.3f}  (n={len(v)})")
    print("  interpolation vs extrapolation (cond):",
          f"interp={np.mean(interp['cond']):.3f} (n={len(interp['cond'])})",
          f"extrap={np.mean(extrap['cond']):.3f} (n={len(extrap['cond'])})")
    out["interp_extrap"] = {t: dict(interp=float(np.mean(interp[t])) if interp[t] else None,
                                    extrap=float(np.mean(extrap[t])) if extrap[t] else None) for t in res}
    json.dump(out, open(os.path.join(OUT, "exp015_grouped.json"), "w"), indent=2)
    print("saved results/exp015_grouped.json")


if __name__ == "__main__":
    main()
