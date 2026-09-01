"""Second benchmark (Fable's ceiling-raiser + objection-killer): University of Ottawa bearing data under
TIME-VARYING speed (Mendeley v43hmbwxpm). Unlike MaFaulDa (constant speed per recording), speed sweeps
WITHIN each 10 s recording (encoder ch2, 1024 CPR, ~9-30 Hz), so conditioning on the per-window
instantaneous speed is genuine continuous-covariate conditioning, not conditioning-on-recording. Faults
are order-locked (bearing defect orders) -> this is the FAVOURABLE regime for order tracking (remove).

Runs the same three treatments as MaFaulDa, leakage-free leave-one-recording-out over healthy files:
  IGNORE  : PCA recon-error on revolution-normalized spectra (REV), pooled across speed.
  CONDITION: subtract kNN-in-(instantaneous)-speed healthy mean from REV, then PCA.
  REMOVE  : PCA recon-error on order-tracked spectra (ORD).
Classes: H (healthy) vs I/O/B (+C combined). fs=200 kHz. Extracts features per file (memory-light)."""
import os, sys, glob, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from scipy.io import loadmat

FS = 200_000; CPR = 1024; REVS = 8; SPER = 512; NFFT_HZ = 8192
L = REVS * SPER
FEAT = "results/ottawa_feats.npz"
os.makedirs("results", exist_ok=True)


def log(m): print(f"[ottawa] {m}", flush=True)


def edges(enc):
    x = (enc - enc.min()) / (np.ptp(enc) + 1e-12); a = x > 0.5
    return np.where((~a[:-1]) & (a[1:]))[0]


def file_feats(path):
    m = loadmat(path); vib = m["Channel_1"].ravel().astype(np.float64); enc = m["Channel_2"].ravel().astype(np.float64)
    e = edges(enc)
    revm = e[::CPR]                     # one marker per shaft revolution (every 1024 encoder edges)
    if len(revm) < REVS + 1:
        return None
    win = np.hanning(L); H, R_, O, S = [], [], [], []
    for b in range((len(revm) - 1) // REVS):
        p0, p1 = revm[b*REVS], revm[(b+1)*REVS]
        if p1 <= p0 or p1 > len(vib):
            continue
        seg = vib[p0:p1] - vib[p0:p1].mean()
        speed = REVS / ((p1 - p0) / FS)                      # instantaneous shaft Hz for this window
        # HZ: physical-Hz spectrum (decimate to keep NFFT modest), fixed bins
        dec = max(1, len(seg) // NFFT_HZ); sd = seg[::dec]
        h = np.zeros(NFFT_HZ); n = min(len(sd), NFFT_HZ); h[:n] = sd[:n] * np.hanning(n)
        HZ = np.log1p(np.abs(np.fft.rfft(h)))
        # REV: resample window to common length (revolution-normalized), FFT
        t = np.interp(np.linspace(0, len(seg)-1, L), np.arange(len(seg)), seg)
        REV = np.log1p(np.abs(np.fft.rfft(t * win)))
        # ORD: order tracking - resample against shaft angle using the REVS+1 revolution markers
        loc = revm[b*REVS:(b+1)*REVS+1]
        ang = np.interp(np.arange(p0, p1), loc, np.arange(len(loc)))     # fractional revolutions
        grid = np.linspace(ang[0], ang[-1], L, endpoint=False); o = np.interp(grid, ang, seg)
        ORD = np.log1p(np.abs(np.fft.rfft(o * win)))
        H.append(HZ.astype(np.float32)); R_.append(REV.astype(np.float32)); O.append(ORD.astype(np.float32)); S.append(speed)
    if not S:
        return None
    return np.array(H), np.array(R_), np.array(O), np.array(S, np.float32)


def extract_all():
    if os.path.exists(FEAT):
        d = np.load(FEAT); log("feats cache HIT"); return {k: d[k] for k in d.files}
    files = sorted(glob.glob("data/ottawa/*.mat"))
    HZ, REV, ORD, S, CLS, REC = [], [], [], [], [], []
    for ri, path in enumerate(files):
        cls = os.path.basename(path)[0]           # H/I/O/B/C
        r = file_feats(path)
        if r is None:
            log(f"  skip {path}"); continue
        h, rv, o, s = r
        HZ.append(h); REV.append(rv); ORD.append(o); S.append(s)
        CLS.append(np.array([cls]*len(s))); REC.append(np.full(len(s), ri))
        log(f"  {os.path.basename(path)}: {len(s)} win, speed {s.min():.1f}-{s.max():.1f} Hz")
    out = dict(HZ=np.vstack(HZ), REV=np.vstack(REV), ORD=np.vstack(ORD),
               S=np.concatenate(S), CLS=np.concatenate(CLS), REC=np.concatenate(REC))
    np.savez_compressed(FEAT, **out); log(f"saved {FEAT}")
    return out


def knn_mean(Xtr, st, sq, k=15, chunk=2000):
    o = np.argsort(st); ss = st[o]; XX = Xtr[o]; n = len(ss); w = min(2*k, n)
    idx = np.searchsorted(ss, sq); wof = np.arange(w)
    out = np.empty((len(sq), Xtr.shape[1]), np.float32)
    for c0 in range(0, len(sq), chunk):
        c1 = min(c0+chunk, len(sq)); qs = sq[c0:c1][:, None]
        start = np.clip(idx[c0:c1]-k, 0, max(0, n-w))[:, None]; cand = start + wof
        d = np.abs(ss[cand]-qs); sel = np.argsort(d, 1)[:, :k]; nn = np.take_along_axis(cand, sel, 1)
        out[c0:c1] = XX[nn].mean(1)
    return out


def pca(Xtr, k=32):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-8
    _, _, Vt = np.linalg.svd((Xtr-mu)/sd, full_matrices=False); comp = Vt[:min(k, max(1, Xtr.shape[0]-2))]
    def sc(X, ch=4000):
        out = np.empty(len(X), np.float32)
        for i in range(0, len(X), ch):
            xs = (X[i:i+ch]-mu)/sd; out[i:i+ch] = np.sqrt(((xs-(xs@comp.T)@comp)**2).mean(1))
        return out
    return sc


def main():
    from sklearn.metrics import roc_auc_score
    from scipy.stats import wilcoxon
    D = extract_all()
    REV, ORD, HZ, S, CLS, REC = D["REV"], D["ORD"], D["HZ"], D["S"], D["CLS"], D["REC"]
    Hmask = CLS == "H"; Fmask = ~Hmask
    log(f"healthy windows {Hmask.sum()} ({len(np.unique(REC[Hmask]))} recs); fault {Fmask.sum()}; "
        f"speed {S.min():.1f}-{S.max():.1f} Hz; classes {sorted(set(CLS.tolist()))}")
    def auc(neg, pos): return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))
    Xf_rev, Xf_ord, Sf = REV[Fmask], ORD[Fmask], S[Fmask]
    hrec = np.unique(REC[Hmask])
    acc = {m: [] for m in ("ignore", "condition", "remove")}
    for rid in hrec:
        te = (REC == rid) & Hmask; tr = Hmask & (REC != rid)
        if te.sum() < 5 or tr.sum() < 40:
            continue
        fi = pca(REV[tr]); acc["ignore"].append(auc(fi(REV[te]), fi(Xf_rev)))
        fr = pca(ORD[tr]); acc["remove"].append(auc(fr(ORD[te]), fr(Xf_ord)))
        mtr = knn_mean(REV[tr], S[tr], S[tr]); fc = pca(REV[tr]-mtr)
        acc["condition"].append(auc(fc(REV[te]-knn_mean(REV[tr], S[tr], S[te])),
                                    fc(Xf_rev-knn_mean(REV[tr], S[tr], Sf))))
    def ci(v, n=2000):
        v = np.asarray(v); rng = np.random.default_rng(1)
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
        return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    res = {"per_class_auc": {}, "LOSO": {m: {"mean": round(float(np.mean(v)), 3), "ci95": ci(v), "n": len(v)}
                                          for m, v in acc.items()}}
    # per-fault-class condition AUC (pooled healthy model, leave-one-healthy-rec-out averaged is above;
    # here a simple per-class breakdown using all healthy as reference for context)
    fi = pca(REV[Hmask]); fr = pca(ORD[Hmask]); mtr = knn_mean(REV[Hmask], S[Hmask], S[Hmask]); fc = pca(REV[Hmask]-mtr)
    hs_i = fi(REV[Hmask]); hs_r = fr(ORD[Hmask]); hs_c = fc(REV[Hmask]-mtr)
    for c in ["I", "O", "B", "C"]:
        cm = CLS == c
        if cm.sum() < 5: continue
        res["per_class_auc"][c] = {
            "ignore": round(auc(hs_i, fi(REV[cm])), 3),
            "condition": round(auc(hs_c, fc(REV[cm]-knn_mean(REV[Hmask], S[Hmask], S[cm]))), 3),
            "remove": round(auc(hs_r, fr(ORD[cm])), 3)}
    try:
        res["wilcoxon_p"] = {"condition_vs_ignore": float(wilcoxon(acc["condition"], acc["ignore"]).pvalue),
                             "condition_vs_remove": float(wilcoxon(acc["condition"], acc["remove"]).pvalue),
                             "ignore_vs_remove": float(wilcoxon(acc["ignore"], acc["remove"]).pvalue)}
    except Exception as e:
        res["wilcoxon_p"] = {"err": str(e)[:60]}
    json.dump(res, open("results/ottawa.json", "w"), indent=2)
    log("LOSO: " + json.dumps(res["LOSO"]))
    log("per-class: " + json.dumps(res["per_class_auc"]))
    log("wilcoxon: " + json.dumps(res["wilcoxon_p"]))
    log("=== DONE ===")


if __name__ == "__main__":
    main()
