"""EXP-015b: memory-light leakage-free LOSO (W3/W4). Parse ONLY normal.zip with per-recording
IDs (49 files); reuse existing mf_<fault>.npz caches for the pooled fault positive set. For each
held-out healthy recording, train base/inv/cond on the other recordings (conditional mean
cross-fitted, held-out recording excluded), test held-out-normal vs all faults. Reports mean AUC
and interpolation vs extrapolation. Validates AUC against scikit-learn.
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
        yield (np.log1p(np.abs(np.fft.rfft((t - t.mean()) * win))).astype(np.float32),
               np.log1p(np.abs(np.fft.rfft((o - o.mean()) * win))).astype(np.float32))


def build_normal():
    p = os.path.join(OUT, "loso_normal.npz")
    if os.path.exists(p):
        d = np.load(p); return d["T"], d["O"], d["S"], d["R"]
    T, O, S, R = [], [], [], []
    with zipfile.ZipFile(os.path.join(DATA, "normal.zip")) as zf:
        for ri, n in enumerate(sorted(x for x in zf.namelist() if x.lower().endswith(".csv"))):
            arr = np.loadtxt(io.BytesIO(zf.read(n)), delimiter=","); rises = tacho_rises(arr[:, 0])
            for t, o in feats(arr[:, ACC], rises):
                T.append(t); O.append(o); S.append(FS / np.median(np.diff(rises))); R.append(ri)
            del arr
    T = np.asarray(T, np.float32); O = np.asarray(O, np.float32); S = np.asarray(S, np.float32); R = np.asarray(R)
    np.savez_compressed(p, T=T, O=O, S=S, R=R); return T, O, S, R


def load_faults():
    Ts, Os = [], []
    for stem in FAULTS:
        d = np.load(os.path.join(OUT, f"mf_{stem}.npz"))
        Ts.append(d["T"].astype(np.float32)); Os.append(d["O"].astype(np.float32))
    return np.vstack(Ts), np.vstack(Os)


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
    def score(X, chunk=2000):
        out = np.empty(len(X), np.float32)
        for i in range(0, len(X), chunk):
            xs = (X[i:i+chunk] - mu) / sd
            out[i:i+chunk] = np.sqrt(((xs - (xs @ comp.T) @ comp) ** 2).mean(1))
        return out
    return score


def auc(neg, pos):
    a = np.concatenate([neg, pos]); _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); av = (cs - c + cs + 1) / 2.0; r = av[inv]; n1 = len(pos)
    return float((r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (len(neg) * n1))


def main():
    Tn, On, Sn, Rn = build_normal(); Tf, Of = load_faults()
    recs = np.unique(Rn)
    print(f"healthy windows={len(Tn)} across {len(recs)} recordings; fault windows={len(Tf)}")
    try:
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0); a = rng.normal(0, 1, 300); b = rng.normal(0.4, 1, 300)
        print(f"AUC vs sklearn: mine={auc(a,b):.4f} sk={roc_auc_score([0]*300+[1]*300, np.r_[a,b]):.4f}")
    except Exception as e:
        print("sklearn skipped:", e)
    res = {t: [] for t in ("base", "inv", "cond")}
    ie = {t: {"interp": [], "extrap": []} for t in ("base", "inv", "cond")}
    for rid in recs:
        te = Rn == rid; tr = ~te
        if te.sum() < 5 or tr.sum() < 40:
            continue
        s_lo, s_hi = Sn[tr].min(), Sn[tr].max(); is_in = s_lo <= Sn[te].mean() <= s_hi
        fb = pca(Tn[tr]); fo = pca(On[tr]); fc = pca(Tn[tr] - knn_mean(Tn[tr], Sn[tr], Sn[tr]))
        Sf = _fault_speeds(Tf)
        def cond_score(Xq, Sq, chunk=1500):
            out = np.empty(len(Xq), np.float32)
            for i in range(0, len(Xq), chunk):
                r = Xq[i:i+chunk] - knn_mean(Tn[tr], Sn[tr], Sq[i:i+chunk])
                out[i:i+chunk] = fc(r)
            return out
        a = {"base": auc(fb(Tn[te]), fb(Tf)), "inv": auc(fo(On[te]), fo(Of)),
             "cond": auc(cond_score(Tn[te], Sn[te]), cond_score(Tf, Sf))}
        for t in a:
            res[t].append(a[t]); ie[t]["interp" if is_in else "extrap"].append(a[t])
    out = {}
    print("\n### LOSO (leave-one-speed-out, cross-fitted) mean AUC over held-out recordings ###")
    for t in ("base", "inv", "cond"):
        v = np.array(res[t]); out[t] = dict(mean=float(v.mean()), std=float(v.std()), n=len(v),
                                            interp=float(np.mean(ie[t]["interp"])) if ie[t]["interp"] else None,
                                            extrap=float(np.mean(ie[t]["extrap"])) if ie[t]["extrap"] else None)
        print(f"  {t:5s}: {v.mean():.3f} ± {v.std():.3f} (n={len(v)}) | interp={out[t]['interp']} extrap={out[t]['extrap']}")
    json.dump(out, open(os.path.join(OUT, "exp015_grouped.json"), "w"), indent=2)
    print("saved results/exp015_grouped.json")


# fault speeds are needed for cond scoring; recompute from mf caches' S
_FAULT_S = None
def _fault_speeds(Tf):
    global _FAULT_S
    if _FAULT_S is None:
        Ss = [np.load(os.path.join(OUT, f"mf_{s}.npz"))["S"].astype(np.float32) for s in FAULTS]
        _FAULT_S = np.concatenate(Ss)
    return _FAULT_S


if __name__ == "__main__":
    main()
