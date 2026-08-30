"""RunPod job (v2, PERSISTENT): MSSP strong-accept experiments with S3-volume feature caching.
FIRST run: download MaFaulDa, extract HZ/REV/ORD features + speeds + recording IDs, save one npz,
and UPLOAD it to the network volume under cache/. LATER runs: download that ~300MB cache from the
volume and skip the 13GB download + extraction entirely (minutes, not 30+ min).
Also saves RICH results (per-recording AUC arrays, per-band curves, raw config) to results/ for
reanalysis, in addition to summary numbers. CPU-only. Needs boto3, scikit-learn, scipy."""
import os, io, sys, json, zipfile, urllib.request, numpy as np

FS = 50_000; ACC = 1; REVS = 16; SPER = 256; NFFT = 8192
BASE = "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/"
FAULTS = ["horizontal-misalignment", "vertical-misalignment", "imbalance", "underhang", "overhang"]
CACHE_KEY = "cache/mafaulda_feats_v2.npz"
os.makedirs("data", exist_ok=True); os.makedirs("results", exist_ok=True)


def log(m): print(f"[train] {m}"); sys.stdout.flush()


def s3():
    import boto3
    k = json.load(open("runpod_storage.key"))
    cli = boto3.client("s3", endpoint_url=k["endpoint"], aws_access_key_id=k["access_key"],
                       aws_secret_access_key=k["secret_key"], region_name="us-ca-2")
    return cli, k["volume_id"]


def rises(t):
    x = t.astype(float); x = (x - x.min()) / (np.ptp(x) + 1e-12); a = x > 0.5
    return np.where((~a[:-1]) & (a[1:]))[0]


def feats(sig, r):
    L = REVS * SPER; win = np.hanning(L)
    if len(r) < REVS + 1:
        return
    for b in range((len(r) - 1) // REVS):
        p0, p1 = r[b*REVS], r[(b+1)*REVS]
        if p1 <= p0 or p1 > len(sig):
            continue
        seg = sig[p0:p1].astype(float); seg = seg - seg.mean()
        h = np.zeros(NFFT); n = min(len(seg), NFFT); h[:n] = seg[:n] * np.hanning(n)
        HZ = np.log1p(np.abs(np.fft.rfft(h)))
        t = np.interp(np.linspace(0, len(seg)-1, L), np.arange(len(seg)), seg)
        REV = np.log1p(np.abs(np.fft.rfft(t * win)))
        loc = r[b*REVS:(b+1)*REVS+1]; rev = np.interp(np.arange(p0, p1), loc, np.arange(len(loc)))
        grid = np.linspace(rev[0], rev[-1], L, endpoint=False); o = np.interp(grid, rev, seg)
        ORD = np.log1p(np.abs(np.fft.rfft(o * win)))
        yield HZ.astype(np.float32), REV.astype(np.float32), ORD.astype(np.float32)


def extract(stem):
    p = f"data/{stem}.zip"
    if not (os.path.exists(p) and os.path.getsize(p) > 1e6):
        log(f"  download {stem}.zip"); urllib.request.urlretrieve(BASE + f"{stem}.zip", p)
    H, R_, O, S, REC = [], [], [], [], []
    with zipfile.ZipFile(p) as zf:
        for ri, n in enumerate(sorted(x for x in zf.namelist() if x.lower().endswith(".csv"))):
            arr = np.loadtxt(io.BytesIO(zf.read(n)), delimiter=","); r = rises(arr[:, 0])
            for a, b, c in feats(arr[:, ACC], r):
                H.append(a); R_.append(b); O.append(c); S.append(FS/np.median(np.diff(r))); REC.append(ri)
    return (np.asarray(H, np.float32), np.asarray(R_, np.float32), np.asarray(O, np.float32),
            np.asarray(S, np.float32), np.asarray(REC))


def get_features():
    """Return dict with healthy + fault HZ/REV/ORD, speeds, recording ids. Cached on the volume."""
    cli, vol = s3()
    try:
        log(f"checking volume cache {CACHE_KEY}")
        cli.download_file(vol, CACHE_KEY, "feats.npz")
        d = np.load("feats.npz"); log("  cache HIT: loaded features from volume (skipped download+extract)")
        return {k: d[k] for k in d.files}
    except Exception as e:
        log(f"  cache MISS ({str(e)[:50]}); building features")
    Nh = extract("normal")
    Fp = [extract(s) for s in FAULTS]
    out = dict(Nh_HZ=Nh[0], Nh_REV=Nh[1], Nh_ORD=Nh[2], Nh_S=Nh[3], Nh_R=Nh[4],
               F_HZ=np.vstack([Fp[i][0] for i in range(5)]),
               F_REV=np.vstack([Fp[i][1] for i in range(5)]),
               F_ORD=np.vstack([Fp[i][2] for i in range(5)]),
               F_S=np.concatenate([Fp[i][3] for i in range(5)]),
               F_FAULT=np.concatenate([np.full(len(Fp[i][3]), i) for i in range(5)]))
    np.savez_compressed("feats.npz", **out)
    log(f"  uploading feature cache ({os.path.getsize('feats.npz')/1e6:.0f} MB) to volume")
    cli.upload_file("feats.npz", vol, CACHE_KEY)
    return out


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
    def sc(X, ch=4000):
        out = np.empty(len(X), np.float32)
        for i in range(0, len(X), ch):
            xs = (X[i:i+ch]-mu)/sd; out[i:i+ch] = np.sqrt(((xs-(xs@comp.T)@comp)**2).mean(1))
        return out
    return sc


def main():
    from sklearn.metrics import roc_auc_score
    from scipy.stats import wilcoxon
    D = get_features()
    reps = ["HZ", "REV", "ORD"]
    Nrep = {"HZ": D["Nh_HZ"], "REV": D["Nh_REV"], "ORD": D["Nh_ORD"]}; Sn = D["Nh_S"]; Rn = D["Nh_R"]
    Frep = {"HZ": D["F_HZ"], "REV": D["F_REV"], "ORD": D["F_ORD"]}; Sf = D["F_S"]; Ff = D["F_FAULT"]
    log(f"healthy={len(Sn)}/{len(np.unique(Rn))} recs; fault={len(Sf)}; dims {[Nrep[r].shape[1] for r in reps]}")
    def auc(neg, pos): return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))
    res = {"config": {"k_nn": 15, "k_pca": 32, "revs": REVS, "sper": SPER, "nfft": NFFT, "auc": "sklearn"}}

    # E1 representation x {uncond,cond} leakage-free LOSO (save per-recording arrays)
    log("E1: representation factorial (leakage-free LOSO)")
    recs = np.unique(Rn); e1 = {f"{r}-{c}": [] for r in reps for c in ("uncond", "cond")}
    for rid in recs:
        te = Rn == rid; tr = ~te
        if te.sum() < 5 or tr.sum() < 40:
            continue
        for r in reps:
            Xn, Xf = Nrep[r], Frep[r]
            fu = pca(Xn[tr]); e1[f"{r}-uncond"].append(auc(fu(Xn[te]), fu(Xf)))
            fc = pca(Xn[tr]-knn_mean(Xn[tr], Sn[tr], Sn[tr]))
            e1[f"{r}-cond"].append(auc(fc(Xn[te]-knn_mean(Xn[tr], Sn[tr], Sn[te])), fc(Xf-knn_mean(Xn[tr], Sn[tr], Sf))))
    res["E1_representation"] = {k: {"mean": round(float(np.mean(v)), 3), "per_recording": [round(x, 4) for x in v]} for k, v in e1.items()}
    for k in e1: log(f"  {k}: {res['E1_representation'][k]['mean']}")

    # E6 variance decomposition + E10 R2_cond per representation
    log("E6/E10: variance decomposition + covariate relevance")
    e6 = {}
    for r in reps:
        X = Nrep[r]; m = knn_mean(X, Sn, Sn)
        rho = float(np.var(m, 0).sum() / (np.var(X, 0).sum()+1e-12))
        num = 0.0
        for rid in np.unique(Rn):
            te = Rn == rid; tr = ~te
            if te.sum() < 1 or tr.sum() < 20:
                continue
            num += ((X[te]-knn_mean(X[tr], Sn[tr], Sn[te]))**2).sum()
        r2 = float(1 - num/((X-X.mean(0))**2).sum())
        e6[r] = dict(rho_between=round(rho, 3), r2_cond=round(r2, 3))
        log(f"  {r}: rho_between={e6[r]['rho_between']} R2_cond={e6[r]['r2_cond']}")
    res["E6_variance"] = e6

    # E7 threshold transfer (REV, base vs cond), save per-band curves
    log("E7: threshold transfer")
    X = Nrep["REV"]; Xf = Frep["REV"]; bands = [(12,20),(20,28),(28,36),(36,44),(44,52),(52,61)]
    rng = np.random.default_rng(0); idx = rng.permutation(len(X)); cut = int(.6*len(X)); tr, te = idx[:cut], idx[cut:]
    e7 = {}
    for name, cond in [("base", False), ("cond", True)]:
        f = pca(X[tr]-knn_mean(X[tr], Sn[tr], Sn[tr])) if cond else pca(X[tr])
        S_ = (lambda Xq, Sq: f(Xq-knn_mean(X[tr], Sn[tr], Sq))) if cond else (lambda Xq, Sq: f(Xq))
        thr = float(np.percentile(S_(X[tr], Sn[tr]), 99))
        fpr, tpr, med = {}, {}, {}
        for lo, hi in bands:
            hn = (Sn[te] >= lo) & (Sn[te] < hi); fm = (Sf >= lo) & (Sf < hi)
            if hn.sum() < 5 or fm.sum() < 5:
                continue
            hs = S_(X[te][hn], Sn[te][hn]); fs = S_(Xf[fm], Sf[fm])
            fpr[f"{lo}-{hi}"] = round(float((hs > thr).mean()), 3); tpr[f"{lo}-{hi}"] = round(float((fs > thr).mean()), 3)
            med[f"{lo}-{hi}"] = round(float(np.median(hs)), 4)
        cov = float(np.std(list(med.values())) / (np.mean(list(med.values()))+1e-9))
        e7[name] = dict(threshold=thr, fpr_by_band=fpr, tpr_by_band=tpr, healthy_median_by_band=med, healthy_score_CoV=round(cov, 3))
        log(f"  {name}: healthy-score CoV={cov:.3f} FPR {min(fpr.values())}-{max(fpr.values())} TPR {min(tpr.values())}-{max(tpr.values())}")
    res["E7_threshold"] = e7

    json.dump(res, open("results/strongaccept.json", "w"), indent=2)
    # also drop raw per-recording E1 as csv-ish for reanalysis
    json.dump({k: [round(x, 4) for x in v] for k, v in e1.items()}, open("results/e1_per_recording.json", "w"), indent=2)
    log("saved results/strongaccept.json + results/e1_per_recording.json")
    print("[train] === DONE ==="); sys.stdout.flush()


if __name__ == "__main__":
    main()
