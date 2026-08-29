"""EXP-006b: memory-light multi-fault validation (rewrite of exp006 that OOM'd on underhang).
Processes one fault at a time in float32; pooled AUC is built from accumulated per-window
SCORES (tiny 1-D arrays), never by stacking fault feature matrices. Reuses mf_*.npz caches;
parses overhang if its cache is missing. Saves results to results/exp006_results.json.

Detectors (normal-only, accel col1): baseline (uncond PCA/time), invariance (PCA/order),
conditioning (kNN-in-speed PCA/time). Protocols: matched-range 5-seed, leave-one-band-out.
"""
import io, os, json, zipfile
import numpy as np

FS = 50_000
HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data"); OUT = os.path.join(HERE, "..", "results")
ACC, REVS, SPER = 1, 16, 256
FAULTS = ["horizontal-misalignment", "vertical-misalignment", "imbalance", "underhang", "overhang"]
BANDS = [(12, 20), (20, 28), (28, 36), (36, 44), (44, 52), (52, 61)]


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


def load_cache(stem):
    p = os.path.join(OUT, f"mf_{stem}.npz")
    if not os.path.exists(p):
        zp = os.path.join(DATA, f"{stem}.zip")
        if not os.path.exists(zp):
            return None
        T, O, S = build(zp); np.savez_compressed(p, T=T, O=O, S=S)
    d = np.load(p)
    return d["T"].astype(np.float32), d["O"].astype(np.float32), d["S"].astype(np.float32)


def pca_fit(X, k=32):
    mu = X.mean(0); sd = X.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
    return dict(mu=mu, sd=sd, comp=Vt[:min(k, X.shape[0] - 2)].astype(np.float32))


def pe(m, X):
    Xs = (X - m["mu"]) / m["sd"]
    return np.sqrt(((Xs - (Xs @ m["comp"].T) @ m["comp"]) ** 2).mean(1))


def knn(Xtr, st, sq, k=15):
    o = np.argsort(st); ss = st[o]; XX = Xtr[o]; idx = np.searchsorted(ss, sq)
    out = np.empty((len(sq), Xtr.shape[1]), dtype=np.float32)
    for i, q in enumerate(sq):
        j = idx[i]; lo = max(0, j - k); hi = min(len(ss), j + k); c = np.arange(lo, hi)
        nn = c[np.argsort(np.abs(ss[c] - q))[:k]]; out[i] = XX[nn].mean(0)
    return out


def auc(neg, pos):
    a = np.concatenate([neg, pos]); _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); av = (cs - c + cs + 1) / 2.0; r = av[inv]; n1 = len(pos)
    return float((r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (len(neg) * n1))


def main():
    Tn, On, Sn = load_cache("normal")
    K, k = 32, 15
    # ---- precompute normal-side models + normal-test scores ----
    seeds = list(range(5))
    matched = {}  # seed -> dict(models, neg_scores{det})
    for s in seeds:
        rng = np.random.default_rng(s); idx = rng.permutation(len(Tn)); cut = int(0.6 * len(Tn))
        tr, te = idx[:cut], idx[cut:]
        mb = pca_fit(Tn[tr], K); mo = pca_fit(On[tr], K)
        Rtr = Tn[tr] - knn(Tn[tr], Sn[tr], Sn[tr], k); mc = pca_fit(Rtr, K)
        neg = {"base": pe(mb, Tn[te]), "inv": pe(mo, On[te]),
               "cond": pe(mc, Tn[te] - knn(Tn[tr], Sn[tr], Sn[te], k))}
        matched[s] = dict(tr=tr, mb=mb, mo=mo, mc=mc, neg=neg)
    lobo = {}  # band -> models + neg scores + mask counts
    for bi, (lo, hi) in enumerate(BANDS):
        ten = (Sn >= lo) & (Sn < hi); trn = ~ten
        if ten.sum() < 8 or trn.sum() < 40:
            continue
        mb = pca_fit(Tn[trn], K); mo = pca_fit(On[trn], K)
        Rtr = Tn[trn] - knn(Tn[trn], Sn[trn], Sn[trn], k); mc = pca_fit(Rtr, K)
        neg = {"base": pe(mb, Tn[ten]), "inv": pe(mo, On[ten]),
               "cond": pe(mc, Tn[ten] - knn(Tn[trn], Sn[trn], Sn[ten], k))}
        lobo[bi] = dict(trn=trn, mb=mb, mo=mo, mc=mc, neg=neg, lo=lo, hi=hi)

    pooled_matched = {s: {d: [] for d in ("base", "inv", "cond")} for s in seeds}
    pooled_lobo = {bi: {d: [] for d in ("base", "inv", "cond")} for bi in lobo}
    results = {}
    hdr = f"{'fault':22s} {'n':>6}  {'base':>11} {'invariance':>11} {'cond(ours)':>11}   LOBO base/inv/ours"
    print(hdr)
    for stem in FAULTS:
        c = load_cache(stem)
        if c is None:
            print(f"{stem:22s}  (missing)"); continue
        Tf, Of, Sf = c
        # matched 5-seed
        mm = {d: [] for d in ("base", "inv", "cond")}
        for s in seeds:
            M = matched[s]
            fs = {"base": pe(M["mb"], Tf), "inv": pe(M["mo"], Of),
                  "cond": pe(M["mc"], Tf - knn(Tn[M["tr"]], Sn[M["tr"]], Sf, k))}
            for d in ("base", "inv", "cond"):
                mm[d].append(auc(M["neg"][d], fs[d])); pooled_matched[s][d].append(fs[d])
        # lobo
        ll = {d: [] for d in ("base", "inv", "cond")}
        for bi, L in lobo.items():
            tef = (Sf >= L["lo"]) & (Sf < L["hi"])
            if tef.sum() < 8:
                continue
            fs = {"base": pe(L["mb"], Tf[tef]), "inv": pe(L["mo"], Of[tef]),
                  "cond": pe(L["mc"], Tf[tef] - knn(Tn[L["trn"]], Sn[L["trn"]], Sf[tef], k))}
            for d in ("base", "inv", "cond"):
                ll[d].append(auc(L["neg"][d], fs[d])); pooled_lobo[bi][d].append(fs[d])
        row = {d: (float(np.mean(mm[d])), float(np.std(mm[d]))) for d in mm}
        row_l = {d: float(np.mean(ll[d])) for d in ll}
        results[stem] = dict(n=int(len(Tf)), matched=row, lobo=row_l)
        print(f"{stem:22s} {len(Tf):6d}  "
              f"{row['base'][0]:.3f}±{row['base'][1]:.2f} {row['inv'][0]:.3f}±{row['inv'][1]:.2f} "
              f"{row['cond'][0]:.3f}±{row['cond'][1]:.2f}   "
              f"{row_l['base']:.3f}/{row_l['inv']:.3f}/{row_l['cond']:.3f}")
        del Tf, Of, Sf, c

    # pooled
    pm = {d: [] for d in ("base", "inv", "cond")}
    for s in seeds:
        for d in ("base", "inv", "cond"):
            pos = np.concatenate(pooled_matched[s][d]); pm[d].append(auc(matched[s]["neg"][d], pos))
    pl = {d: [] for d in ("base", "inv", "cond")}
    for bi in lobo:
        for d in ("base", "inv", "cond"):
            if pooled_lobo[bi][d]:
                pos = np.concatenate(pooled_lobo[bi][d]); pl[d].append(auc(lobo[bi]["neg"][d], pos))
    prow = {d: (float(np.mean(pm[d])), float(np.std(pm[d]))) for d in pm}
    prow_l = {d: float(np.mean(pl[d])) for d in pl}
    results["POOLED"] = dict(matched=prow, lobo=prow_l)
    print(f"{'POOLED':22s} {'':6s}  "
          f"{prow['base'][0]:.3f}±{prow['base'][1]:.2f} {prow['inv'][0]:.3f}±{prow['inv'][1]:.2f} "
          f"{prow['cond'][0]:.3f}±{prow['cond'][1]:.2f}   "
          f"{prow_l['base']:.3f}/{prow_l['inv']:.3f}/{prow_l['cond']:.3f}")
    with open(os.path.join(OUT, "exp006_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/exp006_results.json")


if __name__ == "__main__":
    main()
