"""EXP-010: does the CONDITIONING principle generalize to a different dataset and to a
DISCRETE attribute? On MIMII-DG (DCASE 2022 Task 2, domain generalization), condition the
normal-only detector on the provided operating ATTRIBUTE and compare to unconditional, on the
SOURCE and TARGET domains. bearing = continuous attribute (rotation speed vel), valve =
categorical attribute (operation pattern pat). Uses section_00 (attribute-varying section).
Parallels the MaFaulDa held-out-speed result: conditioning should help the TARGET (shifted) domain.

Features: log-mel (128) pooled mean+std -> 256-dim per 10s clip (wave reader + torchaudio
MelSpectrogram transform; no load backend needed). Detector: PCA recon error, unconditional vs
attribute-conditional (kNN-in-attr for numeric, per-group mean for categorical). Metric: AUC
(normal vs anomaly) per domain. Official DCASE baselines are CITED, not re-run.
"""
import os, io, re, json, wave, zipfile, numpy as np
import torch, torchaudio
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data", "mimii"); OUT = os.path.join(HERE, "..", "results")
MEL = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128)
PARSE = re.compile(r"section_(\d+)_(source|target)_(train|test)_(normal|anomaly)_(\d+)_(.*)\.wav")


def clip_feature(raw):
    w = wave.open(io.BytesIO(raw)); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    lm = torch.log1p(MEL(torch.tensor(x)))                     # (128, T)
    return torch.cat([lm.mean(1), lm.std(1)]).numpy().astype(np.float32)   # 256-dim


def build_cache(machine, section="00"):
    p = os.path.join(OUT, f"mimii_{machine}_s{section}.npz")
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True); return d["X"], d["meta"]
    X, meta = [], []
    with zipfile.ZipFile(os.path.join(DATA, f"{machine}.zip")) as z:
        for n in sorted(x for x in z.namelist() if x.endswith(".wav")):
            m = PARSE.match(n.split("/")[-1])
            if not m or m.group(1) != section:
                continue
            sec, dom, split, lab, idx, attr = m.groups()
            X.append(clip_feature(z.read(n))); meta.append((dom, split, lab, attr))
    X = np.asarray(X); meta = np.asarray(meta, dtype=object)
    np.savez_compressed(p, X=X, meta=meta); return X, meta


def parse_attr(tok):
    """'vel_6' -> ('num', 6.0); 'pat_00' -> ('cat','00')."""
    parts = tok.split("_"); val = parts[-1]
    try:
        return ("num", float(val))
    except ValueError:
        return ("cat", val)


def pca(Xtr, k=32):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False); comp = Vt[:min(k, Xtr.shape[0] - 2)]
    return lambda X: np.sqrt((((X - mu) / sd - (((X - mu) / sd) @ comp.T) @ comp) ** 2).mean(1))


def cond_mean_num(Xtr, atr, aq, k=15):
    o = np.argsort(atr); ss = atr[o]; XX = Xtr[o]; idx = np.searchsorted(ss, aq)
    out = np.empty((len(aq), Xtr.shape[1]), np.float32)
    for i, q in enumerate(aq):
        j = idx[i]; lo = max(0, j - k); hi = min(len(ss), j + k); c = np.arange(lo, hi)
        out[i] = XX[c[np.argsort(np.abs(ss[c] - q))[:k]]].mean(0)
    return out


def cond_mean_cat(Xtr, atr, aq):
    gm = {}; glob = Xtr.mean(0)
    for a in set(atr):
        m = atr == a
        if m.sum() >= 3:
            gm[a] = Xtr[m].mean(0)
    return np.array([gm.get(a, glob) for a in aq], np.float32)


def auc(neg, pos):
    a = np.concatenate([neg, pos]); _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); av = (cs - c + cs + 1) / 2.0; r = av[inv]; n1 = len(pos)
    return float((r[len(neg):].sum() - n1 * (n1 + 1) / 2) / (len(neg) * n1))


def run_machine(machine):
    X, meta = build_cache(machine)
    dom = np.array([m[0] for m in meta]); split = np.array([m[1] for m in meta])
    lab = np.array([m[2] for m in meta]); atok = np.array([m[3] for m in meta])
    kind = parse_attr(atok[0])[0]
    aval = np.array([parse_attr(t)[1] for t in atok]) if kind == "num" else atok
    tr = split == "train"                                   # normal-only (source 990 + target 10)
    Xtr, atr = X[tr], aval[tr]
    res = {}
    unc = pca(Xtr)
    if kind == "num":
        con = pca(Xtr - cond_mean_num(Xtr, atr.astype(float), atr.astype(float)))
    else:
        con = pca(Xtr - cond_mean_cat(Xtr, atr, atr))
    for d in ["source", "target"]:
        te = (split == "test") & (dom == d)
        Xd, ld, ad = X[te], lab[te], aval[te]
        neg = Xd[ld == "normal"]; pos = Xd[ld == "anomaly"]
        an = ad[ld == "normal"]; ap = ad[ld == "anomaly"]
        au_u = auc(unc(neg), unc(pos))
        if kind == "num":
            cn = con(neg - cond_mean_num(Xtr, atr.astype(float), an.astype(float)))
            cp = con(pos - cond_mean_num(Xtr, atr.astype(float), ap.astype(float)))
        else:
            cn = con(neg - cond_mean_cat(Xtr, atr, an)); cp = con(pos - cond_mean_cat(Xtr, atr, ap))
        au_c = auc(cn, cp)
        res[d] = dict(uncond=au_u, cond=au_c)
        print(f"  {machine:8s} {d:6s}: uncond AUC={au_u:.3f}  cond AUC={au_c:.3f}  (attr={kind})")
    return res


def main():
    all_res = {}
    machines = [m for m in ["bearing", "valve", "fan", "gearbox", "slider"]
                if os.path.exists(os.path.join(DATA, f"{m}.zip"))]
    for machine in machines:
        all_res[machine] = run_machine(machine)
    json.dump(all_res, open(os.path.join(OUT, "exp010_mimii_results.json"), "w"), indent=2)
    print("\nSummary (target-domain = the domain-shifted / unseen-attribute case):")
    for m, r in all_res.items():
        print(f"  {m}: source {r['source']['uncond']:.3f}->{r['source']['cond']:.3f}, "
              f"target {r['target']['uncond']:.3f}->{r['target']['cond']:.3f}")


if __name__ == "__main__":
    main()
