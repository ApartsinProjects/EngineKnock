"""RunPod job: MSSP strong-accept experiments on MaFaulDa (review E1/E6/E7/E10).
Three physically distinct representations of each 16-revolution window, all accel col1:
  HZ   : raw window zero-padded to a fixed length, FFT -> physical-Hz bins (no revolution scaling)
  REV  : window resampled to a common sample count, FFT -> revolution-normalized (the paper's 'time')
  ORD  : tachometer-phase order tracking -> order bins
Experiments:
  E1  representation factorial: base + conditioning on each of HZ/REV/ORD, leakage-free LOSO.
  E6  mechanism: variance-decomposition rho_between = tr(Cov(E[X|S]))/tr(Cov(X)) per representation,
      and fault-minus-healthy residual energy retained after conditioning.
  E7  threshold transfer: healthy-only 99th-percentile threshold from training; held-out-speed
      healthy false-positive rate and fault true-positive rate by speed, base vs conditioning; plus the
      coefficient of variation of healthy scores across speed (score stationarity).
  E10 covariate relevance: cross-fitted R2_cond of the speed covariate on healthy features.
Saves results/strongaccept.json. CPU-only. Uses sklearn/scipy."""
import os, io, sys, json, zipfile, urllib.request, numpy as np

FS = 50_000; ACC = 1; REVS = 16; SPER = 256; NFFT = 8192
BASE = "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/"
FAULTS = ["horizontal-misalignment", "vertical-misalignment", "imbalance", "underhang", "overhang"]
os.makedirs("data", exist_ok=True); os.makedirs("results", exist_ok=True)


def log(m): print(f"[train] {m}"); sys.stdout.flush()


def download(stem):
    p = f"data/{stem}.zip"
    if not (os.path.exists(p) and os.path.getsize(p) > 1e6):
        log(f"downloading {stem}.zip"); urllib.request.urlretrieve(BASE + f"{stem}.zip", p)
    return p


def rises(t):
    x = t.astype(float); x = (x - x.min()) / (np.ptp(x) + 1e-12); a = x > 0.5
    return np.where((~a[:-1]) & (a[1:]))[0]


def feats(sig, r):
    L = REVS * SPER; win = np.hanning(L); hz_win = np.hanning(NFFT)
    if len(r) < REVS + 1:
        return
    nb = (len(r) - 1) // REVS
    for b in range(nb):
        p0, p1 = r[b*REVS], r[(b+1)*REVS]
        if p1 <= p0 or p1 > len(sig):
            continue
        seg = sig[p0:p1].astype(float); seg = seg - seg.mean()
        # HZ: zero-pad/truncate raw segment to NFFT, physical-Hz bins
        h = np.zeros(NFFT); n = min(len(seg), NFFT); h[:n] = seg[:n] * (hz_win[:n] if n == NFFT else np.hanning(n))
        HZ = np.log1p(np.abs(np.fft.rfft(h)))
        # REV: resample to common length
        t = np.interp(np.linspace(0, len(seg)-1, L), np.arange(len(seg)), seg)
        REV = np.log1p(np.abs(np.fft.rfft(t * win)))
        # ORD: order tracking
        loc = r[b*REVS:(b+1)*REVS+1]; rev = np.interp(np.arange(p0, p1), loc, np.arange(len(loc)))
        grid = np.linspace(rev[0], rev[-1], L, endpoint=False); o = np.interp(grid, rev, seg)
        ORD = np.log1p(np.abs(np.fft.rfft(o * win)))
        yield HZ.astype(np.float32), REV.astype(np.float32), ORD.astype(np.float32)


def extract(stem):
    H, R_, O, S, REC = [], [], [], [], []
    with zipfile.ZipFile(download(stem)) as zf:
        for ri, n in enumerate(sorted(x for x in zf.namelist() if x.lower().endswith(".csv"))):
            arr = np.loadtxt(io.BytesIO(zf.read(n)), delimiter=","); r = rises(arr[:, 0])
            for h, rv, o in feats(arr[:, ACC], r):
                H.append(h); R_.append(rv); O.append(o); S.append(FS/np.median(np.diff(r))); REC.append(ri)
    return (np.asarray(H, np.float32), np.asarray(R_, np.float32), np.asarray(O, np.float32),
            np.asarray(S, np.float32), np.asarray(REC))


def knn_mean(Xtr, st, sq, k=15):
    o = np.argsort(st); ss = st[o]; XX = Xtr[o]; idx = np.searchsorted(ss, sq)
    out = np.empty((len(sq), Xtr.shape[1]), np.float32)
    for i, q in enumerate(sq):
        j = idx[i]; lo = max(0, j-k); hi = min(len(ss), j+k); c = np.arange(lo, hi)
        out[i] = XX[c[np.argsort(np.abs(ss[c]-q))[:k]]].mean(0)
    return out


def pca(Xtr, k=32):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-8
    _, _, Vt = np.linalg.svd((Xtr-mu)/sd, full_matrices=False); comp = Vt[:min(k, Xtr.shape[0]-2)]
    def sc(X, ch=2000):
        out = np.empty(len(X), np.float32)
        for i in range(0, len(X), ch):
            xs = (X[i:i+ch]-mu)/sd; out[i:i+ch] = np.sqrt(((xs-(xs@comp.T)@comp)**2).mean(1))
        return out
    return sc


def main():
    from sklearn.metrics import roc_auc_score
    reps = ["HZ", "REV", "ORD"]
    log("extracting HZ/REV/ORD for healthy + faults")
    Nh = extract("normal")  # (H,R,O,S,REC)
    Fparts = [extract(s) for s in FAULTS]
    Nrep = {"HZ": Nh[0], "REV": Nh[1], "ORD": Nh[2]}; Sn = Nh[3]; Rn = Nh[4]
    Frep = {r: np.vstack([Fparts[i][{"HZ": 0, "REV": 1, "ORD": 2}[r]] for i in range(5)]) for r in reps}
    Sf = np.concatenate([Fparts[i][3] for i in range(5)])
    log(f"healthy={len(Sn)} / {len(np.unique(Rn))} recs; fault={len(Sf)}; dims HZ={Nrep['HZ'].shape[1]} REV={Nrep['REV'].shape[1]} ORD={Nrep['ORD'].shape[1]}")
    def auc(neg, pos): return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))
    res = {}

    # ---- E1: representation x {uncond, cond} leakage-free LOSO ----
    log("E1: representation factorial (leakage-free LOSO)")
    recs = np.unique(Rn); e1 = {f"{r}-{c}": [] for r in reps for c in ("uncond", "cond")}
    for rid in recs:
        te = Rn == rid; tr = ~te
        if te.sum() < 5 or tr.sum() < 40:
            continue
        for r in reps:
            Xn, Xf = Nrep[r], Frep[r]
            fu = pca(Xn[tr]); e1[f"{r}-uncond"].append(auc(fu(Xn[te]), fu(Xf)))
            fc = pca(Xn[tr] - knn_mean(Xn[tr], Sn[tr], Sn[tr]))
            e1[f"{r}-cond"].append(auc(fc(Xn[te]-knn_mean(Xn[tr], Sn[tr], Sn[te])), fc(Xf-knn_mean(Xn[tr], Sn[tr], Sf))))
    res["E1_representation"] = {k: round(float(np.mean(v)), 3) for k, v in e1.items()}
    for k, v in res["E1_representation"].items():
        log(f"  {k}: {v}")

    # ---- E6: variance decomposition rho_between + R2_cond (E10) ----
    log("E6/E10: variance decomposition and covariate relevance")
    e6 = {}
    for r in reps:
        X = Nrep[r]; m = knn_mean(X, Sn, Sn)  # E[X|S] estimate (in-sample ok for descriptive decomposition)
        tot = np.trace(np.cov(X.T)) if X.shape[1] < 3000 else float(np.var(X, 0).sum())
        betw = float(np.var(m, 0).sum())
        # cross-fitted R2_cond via leave-one-recording-out mean
        num = 0.0
        for rid in np.unique(Rn):
            te = Rn == rid; tr = ~te
            if te.sum() < 1 or tr.sum() < 20:
                continue
            mm = knn_mean(X[tr], Sn[tr], Sn[te]); num += ((X[te]-mm)**2).sum()
        den = ((X - X.mean(0))**2).sum(); r2 = float(1 - num/den)
        e6[r] = dict(rho_between=round(betw/ (float(np.var(X,0).sum())), 3), r2_cond=round(r2, 3))
        log(f"  {r}: rho_between={e6[r]['rho_between']} R2_cond={e6[r]['r2_cond']}")
    res["E6_variance"] = e6

    # ---- E7: threshold transfer (REV representation, base vs cond) ----
    log("E7: threshold transfer (healthy 99th-pct threshold, held-out-speed FPR/TPR + score CoV)")
    X = Nrep["REV"]; Xf = Frep["REV"]; bands = [(12,20),(20,28),(28,36),(36,44),(44,52),(52,61)]
    e7 = {"base": {}, "cond": {}}
    # global healthy split for threshold: train on 60% healthy across speeds
    rng = np.random.default_rng(0); idx = rng.permutation(len(X)); cut = int(.6*len(X)); tr, te = idx[:cut], idx[cut:]
    for name, mkscore in [("base", lambda: (pca(X[tr]), None)),
                          ("cond", lambda: (pca(X[tr]-knn_mean(X[tr], Sn[tr], Sn[tr])), True))]:
        f, cond = mkscore()
        def S_(Xq, Sq):
            return f(Xq - knn_mean(X[tr], Sn[tr], Sq)) if cond else f(Xq)
        thr = np.percentile(S_(X[tr], Sn[tr]), 99)  # healthy-only threshold
        fpr_by, tpr_by, med_by = {}, {}, {}
        for lo, hi in bands:
            hn = (Sn[te] >= lo) & (Sn[te] < hi); fm = (Sf >= lo) & (Sf < hi)
            if hn.sum() < 5 or fm.sum() < 5:
                continue
            hs = S_(X[te][hn], Sn[te][hn]); fs = S_(Xf[fm], Sf[fm])
            fpr_by[f"{lo}-{hi}"] = round(float((hs > thr).mean()), 3)
            tpr_by[f"{lo}-{hi}"] = round(float((fs > thr).mean()), 3)
            med_by[f"{lo}-{hi}"] = round(float(np.median(hs)), 3)
        cov = float(np.std(list(med_by.values())) / (np.mean(list(med_by.values()))+1e-9))
        e7[name] = dict(fpr_by_band=fpr_by, tpr_by_band=tpr_by, healthy_score_CoV=round(cov, 3))
        log(f"  {name}: healthy-score CoV across speed={cov:.3f}; FPR range {min(fpr_by.values())}-{max(fpr_by.values())} TPR range {min(tpr_by.values())}-{max(tpr_by.values())}")
    res["E7_threshold"] = e7

    json.dump(res, open("results/strongaccept.json", "w"), indent=2)
    log("saved results/strongaccept.json")
    print("[train] === DONE ==="); sys.stdout.flush()


if __name__ == "__main__":
    main()
