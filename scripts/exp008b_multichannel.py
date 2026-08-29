"""EXP-008b: memory-light multi-channel (rewrite of exp008 that hit MemoryError). Build each
source's 3-channel cache one at a time (streaming zf.open), then eval loading one fault at a
time and pooling via accumulated per-channel scores. Channels {1 underhang-axial, 4 overhang-
axial, 7 microphone}. Unconditional vs conditional (kNN-in-speed), per channel + fused.
"""
import io, os, json, zipfile, gc, numpy as np
FS = 50_000
HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data"); OUT = os.path.join(HERE, "..", "results")
CHANS = [1, 4, 7]; CH_NAME = {1: "underhang-axial", 4: "overhang-axial", 7: "microphone"}
REVS, SPER = 16, 256
FAULTS = ["horizontal-misalignment", "vertical-misalignment", "imbalance", "underhang", "overhang"]
BANDS = [(12, 20), (20, 28), (28, 36), (36, 44), (44, 52), (52, 61)]


def tacho_rises(t):
    x = t.astype(float); x = (x - x.min()) / (np.ptp(x) + 1e-12); a = x > 0.5
    return np.where((~a[:-1]) & (a[1:]))[0]


def parse_source(stem):
    """Build+save mf_mc_<stem>.npz if missing. Streams each CSV via zf.open (low peak mem)."""
    p = os.path.join(OUT, f"mf_mc_{stem}.npz")
    if os.path.exists(p):
        return
    zp = os.path.join(DATA, f"{stem}.zip")
    cols = {ch: [] for ch in CHANS}; S = []
    L = REVS * SPER; win = np.hanning(L)
    with zipfile.ZipFile(zp) as zf:
        for n in sorted(x for x in zf.namelist() if x.lower().endswith(".csv")):
            with zf.open(n) as fh:                      # streaming, not full-decompress-to-bytes
                arr = np.loadtxt(fh, delimiter=",")
            r = tacho_rises(arr[:, 0])
            if len(r) < REVS + 1:
                del arr; continue
            nb = (len(r) - 1) // REVS; frot = FS / np.median(np.diff(r))
            for b in range(nb):
                p0, p1 = r[b * REVS], r[(b + 1) * REVS]
                if p1 <= p0 or p1 > arr.shape[0]:
                    continue
                for ch in CHANS:
                    seg = arr[p0:p1, ch].astype(float)
                    t = np.interp(np.linspace(0, len(seg) - 1, L), np.arange(len(seg)), seg)
                    cols[ch].append(np.log1p(np.abs(np.fft.rfft((t - t.mean()) * win))).astype(np.float32))
                S.append(frot)
            del arr
    save = {f"c{ch}": np.asarray(cols[ch], np.float32) for ch in CHANS}; save["S"] = np.asarray(S, np.float32)
    np.savez_compressed(p, **save); del cols, S, save; gc.collect()
    print(f"  built mf_mc_{stem}.npz")


def load_mc(stem):
    d = np.load(os.path.join(OUT, f"mf_mc_{stem}.npz"))
    return {ch: d[f"c{ch}"].astype(np.float32) for ch in CHANS}, d["S"].astype(np.float32)


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


def main():
    for s in ["normal"] + FAULTS:
        parse_source(s)
    Nc, Sn = load_mc("normal")
    res = {}
    for cond in (False, True):
        tag = "cond" if cond else "uncond"
        # precompute normal-side scorers + neg scores, matched(3 seeds) and LOBO(bands)
        seeds = list(range(3))
        Mstate = {}
        for seed in seeds:
            rng = np.random.default_rng(seed); idx = rng.permutation(len(Sn)); cut = int(0.6 * len(Sn))
            tr, te = idx[:cut], idx[cut:]; sc = {}; neg = {}; negz = {}
            for ch in CHANS:
                Xtr = Nc[ch][tr]
                if cond:
                    f = pca(Xtr - knn(Xtr, Sn[tr], Sn[tr])); ref = f(Xtr - knn(Xtr, Sn[tr], Sn[tr]))
                    ng = f(Nc[ch][te] - knn(Xtr, Sn[tr], Sn[te]))
                else:
                    f = pca(Xtr); ref = f(Xtr); ng = f(Nc[ch][te])
                sc[ch] = (f, tr, ref); neg[ch] = ng; negz[ch] = (ng - ref.mean()) / (ref.std() + 1e-8)
            Mstate[seed] = dict(tr=tr, te=te, sc=sc, neg=neg, negz=negz)
        Lstate = {}
        for lo, hi in BANDS:
            ten = (Sn >= lo) & (Sn < hi); trn = ~ten
            if ten.sum() < 8 or trn.sum() < 40:
                continue
            sc = {}; neg = {}; negz = {}
            for ch in CHANS:
                Xtr = Nc[ch][trn]
                if cond:
                    f = pca(Xtr - knn(Xtr, Sn[trn], Sn[trn])); ref = f(Xtr - knn(Xtr, Sn[trn], Sn[trn]))
                    ng = f(Nc[ch][ten] - knn(Xtr, Sn[trn], Sn[ten]))
                else:
                    f = pca(Xtr); ref = f(Xtr); ng = f(Nc[ch][ten])
                sc[ch] = (f, trn, ref); neg[ch] = ng; negz[ch] = (ng - ref.mean()) / (ref.std() + 1e-8)
            Lstate[(lo, hi)] = dict(trn=trn, ten=ten, sc=sc, neg=neg, negz=negz)
        # accumulate fault positive scores per channel
        pos_m = {seed: {ch: [] for ch in CHANS} for seed in seeds}
        pos_l = {b: {ch: [] for ch in CHANS} for b in Lstate}
        for stem in FAULTS:
            Fc, Sf = load_mc(stem)
            for seed in seeds:
                st = Mstate[seed]
                for ch in CHANS:
                    f, tr, ref = st["sc"][ch]
                    p = f(Fc[ch] - knn(Nc[ch][tr], Sn[tr], Sf)) if cond else f(Fc[ch])
                    pos_m[seed][ch].append(p)
            for (lo, hi), st in Lstate.items():
                m = (Sf >= lo) & (Sf < hi)
                if m.sum() < 8:
                    continue
                for ch in CHANS:
                    f, trn, ref = st["sc"][ch]
                    p = f(Fc[ch][m] - knn(Nc[ch][trn], Sn[trn], Sf[m])) if cond else f(Fc[ch][m])
                    pos_l[(lo, hi)][ch].append(p)
            del Fc, Sf; gc.collect()
        # aggregate: per channel + fused (z-scored score average)
        perch = {ch: {"m": [], "l": []} for ch in CHANS}; fused = {"m": [], "l": []}
        for seed in seeds:
            st = Mstate[seed]; negz = st["negz"]; posz = {}
            for ch in CHANS:
                pos = np.concatenate(pos_m[seed][ch]); perch[ch]["m"].append(auc(st["neg"][ch], pos))
                _, tr, ref = st["sc"][ch]; posz[ch] = (pos - ref.mean()) / (ref.std() + 1e-8)
            fused["m"].append(auc(np.mean([negz[c] for c in CHANS], 0), np.mean([posz[c] for c in CHANS], 0)))
        for b, st in Lstate.items():
            if not all(pos_l[b][ch] for ch in CHANS):
                continue
            negz = st["negz"]; posz = {}
            for ch in CHANS:
                pos = np.concatenate(pos_l[b][ch]); perch[ch]["l"].append(auc(st["neg"][ch], pos))
                _, trn, ref = st["sc"][ch]; posz[ch] = (pos - ref.mean()) / (ref.std() + 1e-8)
            fused["l"].append(auc(np.mean([negz[c] for c in CHANS], 0), np.mean([posz[c] for c in CHANS], 0)))
        for ch in CHANS:
            res[f"{CH_NAME[ch]}-{tag}"] = dict(matched=float(np.mean(perch[ch]["m"])), lobo=float(np.mean(perch[ch]["l"])))
        res[f"FUSED-{tag}"] = dict(matched=float(np.mean(fused["m"])), lobo=float(np.mean(fused["l"])))
    for k in sorted(res):
        print(f"[{k:26s}] matched={res[k]['matched']:.3f}  cross-speed(LOBO)={res[k]['lobo']:.3f}")
    json.dump(res, open(os.path.join(OUT, "exp008_mc_results.json"), "w"), indent=2)
    print("saved results/exp008_mc_results.json")


if __name__ == "__main__":
    main()
