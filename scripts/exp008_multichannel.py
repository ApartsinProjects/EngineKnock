"""EXP-008: does speed-conditioning win across CHANNELS (not just accel col1)? Extract time-
spectra for channels {1 underhang-axial, 4 overhang-axial, 7 microphone} per source, and test
unconditional vs conditional (kNN-in-speed) pooled over all faults, per channel and FUSED
(score-average across channels). Memory-light. Parses once into mf_mc_<stem>.npz.
"""
import io, os, json, zipfile, numpy as np
FS = 50_000
HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data"); OUT = os.path.join(HERE, "..", "results")
CHANS = [1, 4, 7]; CH_NAME = {1: "underhang-axial", 4: "overhang-axial", 7: "microphone"}
REVS, SPER = 16, 256
FAULTS = ["horizontal-misalignment", "vertical-misalignment", "imbalance", "underhang", "overhang"]
BANDS = [(12, 20), (20, 28), (28, 36), (36, 44), (44, 52), (52, 61)]


def tacho_rises(t):
    x = t.astype(float); x = (x - x.min()) / (np.ptp(x) + 1e-12); a = x > 0.5
    return np.where((~a[:-1]) & (a[1:]))[0]


def feats_multi(arr, rises, chans=CHANS, revs=REVS, sper=SPER):
    if len(rises) < revs + 1:
        return
    L = revs * sper; nb = (len(rises) - 1) // revs; win = np.hanning(L)
    for b in range(nb):
        p0, p1 = rises[b * revs], rises[(b + 1) * revs]
        if p1 <= p0 or p1 > arr.shape[0]:
            continue
        row = []
        for ch in chans:
            seg = arr[p0:p1, ch].astype(float)
            t = np.interp(np.linspace(0, len(seg) - 1, L), np.arange(len(seg)), seg)
            row.append(np.log1p(np.abs(np.fft.rfft((t - t.mean()) * win))))
        yield row


def build_mc(stem):
    p = os.path.join(OUT, f"mf_mc_{stem}.npz")
    if os.path.exists(p):
        d = np.load(p); return {ch: d[f"c{ch}"].astype(np.float32) for ch in CHANS}, d["S"].astype(np.float32)
    zp = os.path.join(DATA, f"{stem}.zip")
    cols = {ch: [] for ch in CHANS}; S = []
    with zipfile.ZipFile(zp) as zf:
        for n in sorted(x for x in zf.namelist() if x.lower().endswith(".csv")):
            arr = np.loadtxt(io.BytesIO(zf.read(n)), delimiter=",")
            r = tacho_rises(arr[:, 0]); frot = FS / np.median(np.diff(r)) if len(r) > 2 else np.nan
            for row in feats_multi(arr, r):
                for ch, v in zip(CHANS, row):
                    cols[ch].append(v)
                S.append(frot)
    save = {f"c{ch}": np.asarray(cols[ch], np.float32) for ch in CHANS}; save["S"] = np.asarray(S, np.float32)
    np.savez_compressed(p, **save)
    return {ch: save[f"c{ch}"] for ch in CHANS}, save["S"]


def knn(Xtr, st, sq, k=15):
    o = np.argsort(st); ss = st[o]; XX = Xtr[o]; idx = np.searchsorted(ss, sq)
    out = np.empty((len(sq), Xtr.shape[1]), np.float32)
    for i, q in enumerate(sq):
        j = idx[i]; lo = max(0, j - k); hi = min(len(ss), j + k); c = np.arange(lo, hi)
        out[i] = XX[c[np.argsort(np.abs(ss[c] - q))[:k]]].mean(0)
    return out


def pca(Xtr, k=32):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False); comp = Vt[:k]
    return lambda X: np.sqrt((((X - mu) / sd - (((X - mu) / sd) @ comp.T) @ comp) ** 2).mean(1))


def auc(neg, pos):
    a = np.concatenate([neg, pos]); _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); av = (cs - c + cs + 1) / 2.0; r = av[inv]; n1 = len(pos)
    return float((r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (len(neg) * n1))


def zscore_scores(s, ref):  # standardize a score array by ref (normal-train) stats for fusion
    return (s - ref.mean()) / (ref.std() + 1e-8)


def main():
    Nc, Sn = build_mc("normal")
    faults = [(s, *build_mc(s)) for s in FAULTS if os.path.exists(os.path.join(DATA, f"{s}.zip"))]
    res = {}
    seeds = range(3)
    for cond in (False, True):
        tag = "cond" if cond else "uncond"
        # per-channel + fused, pooled, matched (3 seeds) and LOBO
        perch = {ch: {"m": [], "l": []} for ch in CHANS}; fused = {"m": [], "l": []}
        for seed in seeds:
            rng = np.random.default_rng(seed); idx = rng.permutation(len(Sn)); cut = int(0.6 * len(Sn))
            tr, te = idx[:cut], idx[cut:]
            negs = {}; poss = {}
            for ch in CHANS:
                Xtr = Nc[ch][tr]
                if cond:
                    sc = pca(Xtr - knn(Xtr, Sn[tr], Sn[tr]))
                    ntr_ref = sc(Xtr - knn(Xtr, Sn[tr], Sn[tr]))
                    neg = sc(Nc[ch][te] - knn(Xtr, Sn[tr], Sn[te]))
                    pos = np.concatenate([sc(fc[ch] - knn(Xtr, Sn[tr], fs)) for _, fc, fs in faults])
                else:
                    sc = pca(Xtr); ntr_ref = sc(Xtr)
                    neg = sc(Nc[ch][te]); pos = np.concatenate([sc(fc[ch]) for _, fc, fs in faults])
                perch[ch]["m"].append(auc(neg, pos))
                negs[ch] = zscore_scores(neg, ntr_ref); poss[ch] = zscore_scores(pos, ntr_ref)
            fused["m"].append(auc(np.mean([negs[c] for c in CHANS], 0), np.mean([poss[c] for c in CHANS], 0)))
        # LOBO (1 pass, no seeds)
        for lo, hi in BANDS:
            ten = (Sn >= lo) & (Sn < hi); trn = ~ten
            if ten.sum() < 8 or trn.sum() < 40:
                continue
            negs = {}; poss = {}
            for ch in CHANS:
                Xtr = Nc[ch][trn]
                if cond:
                    sc = pca(Xtr - knn(Xtr, Sn[trn], Sn[trn])); ref = sc(Xtr - knn(Xtr, Sn[trn], Sn[trn]))
                    neg = sc(Nc[ch][ten] - knn(Xtr, Sn[trn], Sn[ten]))
                    pos = [sc(fc[ch][(fs >= lo) & (fs < hi)] - knn(Xtr, Sn[trn], fs[(fs >= lo) & (fs < hi)]))
                           for _, fc, fs in faults if ((fs >= lo) & (fs < hi)).sum() >= 8]
                else:
                    sc = pca(Xtr); ref = sc(Xtr); neg = sc(Nc[ch][ten])
                    pos = [sc(fc[ch][(fs >= lo) & (fs < hi)]) for _, fc, fs in faults if ((fs >= lo) & (fs < hi)).sum() >= 8]
                if not pos:
                    continue
                pos = np.concatenate(pos); perch[ch]["l"].append(auc(neg, pos))
                negs[ch] = zscore_scores(neg, ref); poss[ch] = zscore_scores(pos, ref)
            if len(negs) == len(CHANS):
                fused["l"].append(auc(np.mean([negs[c] for c in CHANS], 0), np.mean([poss[c] for c in CHANS], 0)))
        for ch in CHANS:
            res[f"{CH_NAME[ch]}-{tag}"] = dict(matched=float(np.mean(perch[ch]["m"])), lobo=float(np.mean(perch[ch]["l"])))
        res[f"FUSED-{tag}"] = dict(matched=float(np.mean(fused["m"])), lobo=float(np.mean(fused["l"])))
    for k in sorted(res):
        print(f"[{k:26s}] matched={res[k]['matched']:.3f}  cross-speed(LOBO)={res[k]['lobo']:.3f}")
    json.dump(res, open(os.path.join(OUT, "exp008_mc_results.json"), "w"), indent=2)
    print("\nsaved results/exp008_mc_results.json")


if __name__ == "__main__":
    main()
