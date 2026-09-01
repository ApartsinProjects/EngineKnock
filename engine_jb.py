"""Supporting real-hardware check on the Engine Journal Bearings dataset (Mendeley 3fcrrdjjvk).
This is a random-vibration shaker test of engine journal bearings: 134 four-channel accelerometer
records (44 healthy, 90 faulty) at ~296 Hz, ~2 s, across RPM {1000,1500,2000} (a genuine operating
covariate) and Humidity/Temp (nuisance covariates). LIMITATIONS: low rate, short, small, shaker-driven,
NOT a running engine -> a supporting decision-rule replication, not a headline result.

Tests the paper's decision rule on real hardware:
  - IGNORE: PCA recon-error on pooled healthy spectra.
  - CONDITION on RPM (genuine operating covariate): subtract per-RPM healthy mean, then PCA.
  - CONDITION on Humidity / Temp (NUISANCE covariates): should NOT help (echoes MIMII-fan finding).
Reports healthy-vs-faulty ROC-AUC, pooled and leave-one-RPM-out (leakage-free across the covariate)."""
import os, io, re, zipfile, json, numpy as np
from sklearn.metrics import roc_auc_score

ZIP = "data/engine_jb/engine_jb_2.zip"
CH = 4  # accelerometer channels: 'Channel 1'..'Channel 4'


def load():
    z = zipfile.ZipFile(ZIP)
    recs = []  # (label, rpm, hum, temp, feat)
    for n in z.namelist():
        if not n.lower().endswith(".csv"):
            continue
        lab = 0 if "Healthy" in n else 1
        rpm = int(re.search(r"(\d+) RPM", n).group(1))
        hum = int(re.search(r"(\d+)% Humidity", n).group(1)) if re.search(r"(\d+)% Humidity", n) else -1
        tm = re.search(r"(-?\d+)\s*deg", n); temp = int(tm.group(1)) if tm else -999
        raw = z.read(n).decode("utf-8", "replace").splitlines()
        rows = [r.split(",") for r in raw]
        hdr = [h.strip().strip('"') for h in rows[0]]
        try:
            cidx = [hdr.index(f"Channel {i}") for i in range(1, CH + 1)]
        except ValueError:
            continue
        sig = []
        for r in rows[1:]:
            if len(r) <= max(cidx):
                continue
            try:
                sig.append([float(r[c].strip().strip('"')) for c in cidx])
            except ValueError:
                continue
        sig = np.asarray(sig, float)
        if len(sig) < 128:
            continue
        # per-channel log-magnitude spectrum, concatenated -> feature
        feats = []
        for c in range(CH):
            x = sig[:, c] - sig[:, c].mean()
            L = 512; xx = np.zeros(L); m = min(len(x), L); xx[:m] = x[:m] * np.hanning(L)[:m] if m == L else x[:m] * np.hanning(m)
            feats.append(np.log1p(np.abs(np.fft.rfft(xx))))
        recs.append((lab, rpm, hum, temp, np.concatenate(feats).astype(np.float32)))
    return recs


def pca_scorer(Xtr, k=16):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False)
    comp = Vt[:min(k, max(1, Xtr.shape[0] - 2))]
    return lambda X: np.sqrt((((X - mu) / sd - (((X - mu) / sd) @ comp.T) @ comp) ** 2).mean(1))


def group_mean(Xtr, gtr, gq):
    """Per-group healthy mean (discrete conditioning); fallback to global for unseen groups."""
    g = Xtr.mean(0); tab = {}
    for gv in np.unique(gtr):
        tab[gv] = Xtr[gtr == gv].mean(0)
    return np.array([tab.get(v, g) for v in gq])


def auc(neg, pos):
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def main():
    recs = load()
    lab = np.array([r[0] for r in recs]); rpm = np.array([r[1] for r in recs])
    hum = np.array([r[2] for r in recs]); temp = np.array([r[3] for r in recs])
    X = np.vstack([r[4] for r in recs])
    H = lab == 0; F = lab == 1
    print(f"[jb] {len(recs)} records: {H.sum()} healthy / {F.sum()} faulty; dim {X.shape[1]}; "
          f"RPM {sorted(set(rpm.tolist()))}", flush=True)
    res = {"n_healthy": int(H.sum()), "n_faulty": int(F.sum()), "dim": int(X.shape[1]),
           "rate_hz": 296, "note": "shaker-response, low-rate/short/small; supporting only"}

    # pooled (in-distribution reference)
    f = pca_scorer(X[H]); res["pooled_ignore_auc"] = round(auc(f(X[H]), f(X[F])), 3)
    fc = pca_scorer(X[H] - group_mean(X[H], rpm[H], rpm[H]))
    res["pooled_condition_rpm_auc"] = round(auc(fc(X[H] - group_mean(X[H], rpm[H], rpm[H])),
                                                fc(X[F] - group_mean(X[H], rpm[H], rpm[F]))), 3)
    fh = pca_scorer(X[H] - group_mean(X[H], hum[H], hum[H]))
    res["pooled_condition_humidity_auc"] = round(auc(fh(X[H] - group_mean(X[H], hum[H], hum[H])),
                                                     fh(X[F] - group_mean(X[H], hum[H], hum[F]))), 3)

    # leave-one-RPM-out (leakage-free across the operating covariate)
    ig, cond = [], []
    for r in sorted(set(rpm.tolist())):
        trH = H & (rpm != r); teH = H & (rpm == r); teF = F & (rpm == r)
        if teH.sum() < 3 or teF.sum() < 3 or trH.sum() < 8:
            continue
        fi = pca_scorer(X[trH]); ig.append(auc(fi(X[teH]), fi(X[teF])))
        fcc = pca_scorer(X[trH] - group_mean(X[trH], rpm[trH], rpm[trH]))
        cond.append(auc(fcc(X[teH] - group_mean(X[trH], rpm[trH], rpm[teH])),
                        fcc(X[teF] - group_mean(X[trH], rpm[trH], rpm[teF]))))
    res["LORO_ignore_auc"] = round(float(np.mean(ig)), 3) if ig else None
    res["LORO_condition_rpm_auc"] = round(float(np.mean(cond)), 3) if cond else None
    res["LORO_folds"] = len(ig)

    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/engine_jb.json", "w"), indent=2)
    print("[jb] RESULT", json.dumps(res), flush=True)
    print("[jb] === DONE ===", flush=True)


if __name__ == "__main__":
    main()
