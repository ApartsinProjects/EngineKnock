"""RunPod job: improved METHOD + deployment PROTOCOL, loading the persistent feature cache.
Turns the mined negatives into method: (1) SUPPORT-AWARE conditioning that softly blends the local
speed-conditional mean toward the global healthy mean when the query speed leaves the training
support, so extrapolation cannot hurt; (2) a healthy-only R2_cond decision rule; (3) THRESHOLD
TRANSFER as the headline: one healthy-only alarm threshold, held fixed across speed, with false-
positive rate, detection rate, and healthy-score drift (coefficient of variation) reported per speed.
Compares IGNORE / ADJUST / ADJUST-SUPPORT-AWARE / REMOVE, on leakage-free LOSO AUC and threshold
transfer. Uses cached MaFaulDa features (fast). CPU-only."""
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
    for reg in ("us-ks-2", "us-ca-2", "us-east-1"):
        cli = boto3.client("s3", endpoint_url=k["endpoint"], aws_access_key_id=k["access_key"],
                           aws_secret_access_key=k["secret_key"], region_name=reg)
        try:
            cli.list_objects_v2(Bucket=k["volume_id"], MaxKeys=1); return cli, k["volume_id"]
        except Exception:
            continue
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
    if os.path.exists("feats.npz"):
        d = np.load("feats.npz"); log("local feats.npz found (skipped S3+extract)")
        return {k: d[k] for k in d.files}
    cli, vol = s3()
    try:
        cli.download_file(vol, CACHE_KEY, "feats.npz"); d = np.load("feats.npz")
        log("cache HIT"); return {k: d[k] for k in d.files}
    except Exception as e:
        log(f"cache MISS ({str(e)[:40]}); building")
    Nh = extract("normal"); Fp = [extract(s) for s in FAULTS]
    out = dict(Nh_HZ=Nh[0], Nh_REV=Nh[1], Nh_ORD=Nh[2], Nh_S=Nh[3], Nh_R=Nh[4],
               F_HZ=np.vstack([Fp[i][0] for i in range(5)]), F_REV=np.vstack([Fp[i][1] for i in range(5)]),
               F_ORD=np.vstack([Fp[i][2] for i in range(5)]), F_S=np.concatenate([Fp[i][3] for i in range(5)]),
               F_FAULT=np.concatenate([np.full(len(Fp[i][3]), i) for i in range(5)]))
    np.savez_compressed("feats.npz", **out)
    try: cli.upload_file("feats.npz", vol, CACHE_KEY); log("uploaded cache")
    except Exception as e: log(f"cache upload failed {str(e)[:40]}")
    return out


def knn_meandist(Xtr, st, sq, k=15, chunk=2000):
    """Vectorized: local mean AND mean neighbour speed-distance (for support-awareness)."""
    o = np.argsort(st); ss = st[o]; XX = Xtr[o]; n = len(ss); w = min(2 * k, n)
    idx = np.searchsorted(ss, sq); wof = np.arange(w)
    out = np.empty((len(sq), Xtr.shape[1]), np.float32); dist = np.empty(len(sq), np.float32)
    for c0 in range(0, len(sq), chunk):
        c1 = min(c0 + chunk, len(sq)); qs = sq[c0:c1][:, None]
        start = np.clip(idx[c0:c1] - k, 0, max(0, n - w))[:, None]; cand = start + wof
        d = np.abs(ss[cand] - qs); sel = np.argsort(d, 1)[:, :k]; nn = np.take_along_axis(cand, sel, 1)
        out[c0:c1] = XX[nn].mean(1); dist[c0:c1] = np.take_along_axis(d, sel, 1).mean(1)
    return out, dist


def support_mean(Xtr, st, sq, k=15, radius=None):
    """Support-aware conditional mean: blend local mean toward global mean by exp(-(d/radius)^2)."""
    if radius is None:
        radius = float(np.median(np.diff(np.sort(np.unique(np.round(st, 1)))))) * 2 + 1e-6
    m_loc, d = knn_meandist(Xtr, st, sq, k); g = Xtr.mean(0)
    w = np.exp(-(d / radius) ** 2)[:, None]
    return w * m_loc + (1 - w) * g


def knn_mean(Xtr, st, sq, k=15):
    m, _ = knn_meandist(Xtr, st, sq, k); return m


def knn_scalar(vtr, st, sq, k=25, chunk=2000):
    """Vectorized kNN-in-speed local median and MAD of a scalar (per-speed score normalization)."""
    o = np.argsort(st); ss = st[o]; vv = vtr[o]; n = len(ss); ww = min(2 * k, n)
    idx = np.searchsorted(ss, sq); wof = np.arange(ww)
    med = np.empty(len(sq), np.float32); mad = np.empty(len(sq), np.float32)
    for c0 in range(0, len(sq), chunk):
        c1 = min(c0 + chunk, len(sq)); qs = sq[c0:c1][:, None]
        start = np.clip(idx[c0:c1] - k, 0, max(0, n - ww))[:, None]; cand = start + wof
        d = np.abs(ss[cand] - qs); sel = np.argsort(d, 1)[:, :k]; nn = np.take_along_axis(cand, sel, 1)
        vsel = vv[nn]; m = np.median(vsel, 1); med[c0:c1] = m
        mad[c0:c1] = np.median(np.abs(vsel - m[:, None]), 1) + 1e-9
    return med, mad


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
    D = get_features()
    Xn = D["Nh_REV"]; On = D["Nh_ORD"]; Sn = D["Nh_S"]; Rn = D["Nh_R"]
    Xf = D["F_REV"]; Of = D["F_ORD"]; Sf = D["F_S"]; Ff = D["F_FAULT"]
    # stratified fault subsample (memory/time; AUC-preserving). SUB=0 -> use all.
    SUB = int(os.environ.get("FAULT_SUB", "6000"))
    if 0 < SUB < len(Sf):
        rng = np.random.default_rng(0); keep = []
        for fc in np.unique(Ff):
            ii = np.where(Ff == fc)[0]; keep.append(rng.choice(ii, min(len(ii), SUB // len(np.unique(Ff))), replace=False))
        sel = np.sort(np.concatenate(keep)); Xf, Of, Sf, Ff = Xf[sel], Of[sel], Sf[sel], Ff[sel]
    log(f"healthy={len(Sn)}/{len(np.unique(Rn))} recs; fault={len(Sf)} (SUB={SUB})")
    def auc(neg, pos): return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))
    res = {}

    # ---- leakage-free LOSO AUC: ignore / adjust / adjust-support-aware / remove ----
    log("LOSO AUC: ignore / adjust / adjust-support-aware / remove")
    recs = np.unique(Rn); acc = {m: [] for m in ("ignore", "ignore_qnorm", "adjust", "adjust_support", "remove")}
    edge = {m: {"interior": [], "edge": []} for m in ("adjust", "adjust_support")}
    for fold, rid in enumerate(recs):
        te = Rn == rid; tr = ~te
        if te.sum() < 5 or tr.sum() < 40:
            continue
        if fold % 10 == 0:
            log(f"  ...fold {fold}/{len(recs)}")
        lo, hi = Sn[tr].min(), Sn[tr].max(); interior = lo <= Sn[te].mean() <= hi
        fi = pca(Xn[tr]); acc["ignore"].append(auc(fi(Xn[te]), fi(Xf)))
        # ignore_qnorm: per-speed median/MAD normalization of the ignore score (feature space untouched)
        tref = fi(Xn[tr])
        def qn(Xq, Sq):
            m, d = knn_scalar(tref, Sn[tr], Sq); return (fi(Xq) - m) / d
        acc["ignore_qnorm"].append(auc(qn(Xn[te], Sn[te]), qn(Xf, Sf)))
        fr = pca(On[tr]); acc["remove"].append(auc(fr(On[te]), fr(Of)))
        fa = pca(Xn[tr]-knn_mean(Xn[tr], Sn[tr], Sn[tr]))
        a1 = auc(fa(Xn[te]-knn_mean(Xn[tr], Sn[tr], Sn[te])), fa(Xf-knn_mean(Xn[tr], Sn[tr], Sf)))
        acc["adjust"].append(a1); edge["adjust"]["interior" if interior else "edge"].append(a1)
        fs = pca(Xn[tr]-support_mean(Xn[tr], Sn[tr], Sn[tr]))
        a2 = auc(fs(Xn[te]-support_mean(Xn[tr], Sn[tr], Sn[te])), fs(Xf-support_mean(Xn[tr], Sn[tr], Sf)))
        acc["adjust_support"].append(a2); edge["adjust_support"]["interior" if interior else "edge"].append(a2)
    # cluster bootstrap 95% CI over per-recording AUCs
    def boot_ci(v, n=2000):
        v = np.asarray(v); rng = np.random.default_rng(1)
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
        return round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)
    res["LOSO_auc"] = {m: {"mean": round(float(np.mean(v)), 3), "ci95": boot_ci(v), "n": len(v)}
                       for m, v in acc.items()}
    # paired Wilcoxon: adjust vs ignore, adjust vs remove, adjust_support vs adjust, ignore_qnorm vs ignore
    try:
        from scipy.stats import wilcoxon
        pair = {}
        for x, y in [("adjust", "ignore"), ("adjust", "remove"), ("adjust_support", "adjust"), ("ignore_qnorm", "ignore")]:
            if len(acc[x]) == len(acc[y]) and len(acc[x]) > 5:
                pair[f"{x}_vs_{y}"] = float(wilcoxon(acc[x], acc[y]).pvalue)
        res["wilcoxon_p"] = pair
    except Exception as e:
        res["wilcoxon_p"] = {"err": str(e)[:60]}
    res["edge_vs_interior"] = {m: {"interior": round(float(np.mean(edge[m]["interior"])), 3),
                                   "edge": round(float(np.mean(edge[m]["edge"])), 3) if edge[m]["edge"] else None}
                               for m in edge}
    for m in acc: log(f"  {m}: LOSO AUC {res['LOSO_auc'][m]['mean']} CI{res['LOSO_auc'][m]['ci95']}")
    log(f"  edge bands adjust={res['edge_vs_interior']['adjust']['edge']} support-aware={res['edge_vs_interior']['adjust_support']['edge']}")

    # ---- threshold transfer: one healthy-only threshold, fixed across speed ----
    # 3-way healthy split so the threshold is set on data the PCA never saw (else in-sample PCA overfit
    # makes the 99th-pct threshold too tight and held-out healthy FPR -> 1.0). trn=fit PCA + conditional
    # mean reference; cal=set the fixed threshold; tst=measure healthy FPR across speed. Faults give TPR.
    # Payoff metric: at ONE healthy-calibrated threshold (~1% FPR from a disjoint calibration split),
    # report mean detection rate (TPR) per speed band and its ACROSS-BAND stability (CoV). A stable
    # operating point = high, speed-uniform TPR. Averaged over seeds. Objection (v): ignore_qnorm should
    # match average TPR but NOT its uniformity, so score-normalization != conditioning the features.
    log("threshold transfer: TPR at a fixed ~1% FPR threshold, per speed band, seed-averaged")
    bands = [(12,20),(20,28),(28,36),(36,44),(44,52),(52,61)]
    SEEDS = int(os.environ.get("TT_SEEDS", "5"))
    arms = ("ignore", "ignore_qnorm", "adjust", "adjust_support")
    agg = {m: {"fpr": [], "tpr_bands": [], "tpr_cov": []} for m in arms}
    for seed in range(SEEDS):
        idx = np.random.default_rng(seed).permutation(len(Xn))
        a = int(.5*len(Xn)); b = int(.7*len(Xn)); trn, cal, tst = idx[:a], idx[a:b], idx[b:]
        fi = pca(Xn[trn]); ref = fi(Xn[cal])
        fa = pca(Xn[trn]-knn_mean(Xn[trn], Sn[trn], Sn[trn]))
        fs = pca(Xn[trn]-support_mean(Xn[trn], Sn[trn], Sn[trn]))
        S = {
            "ignore": lambda Xq, Sq: fi(Xq),
            "ignore_qnorm": lambda Xq, Sq: (lambda s, mm: (s - mm[0]) / mm[1])(fi(Xq), knn_scalar(ref, Sn[cal], Sq)),
            "adjust": lambda Xq, Sq: fa(Xq-knn_mean(Xn[trn], Sn[trn], Sq)),
            "adjust_support": lambda Xq, Sq: fs(Xq-support_mean(Xn[trn], Sn[trn], Sq)),
        }
        for m in arms:
            thr = float(np.percentile(S[m](Xn[cal], Sn[cal]), 99))
            agg[m]["fpr"].append(float((S[m](Xn[tst], Sn[tst]) > thr).mean()))
            tb = []
            for lo, hi in bands:
                fmk = (Sf >= lo) & (Sf < hi)
                if fmk.sum() < 3: tb.append(np.nan); continue
                tb.append(float((S[m](Xf[fmk], Sf[fmk]) > thr).mean()))
            tb = np.array(tb); agg[m]["tpr_bands"].append(tb)
            vv = tb[~np.isnan(tb)]; agg[m]["tpr_cov"].append(float(np.std(vv) / (np.mean(vv) + 1e-9)))
    tt = {}
    for m in arms:
        tbm = np.nanmean(np.vstack(agg[m]["tpr_bands"]), 0)
        tt[m] = dict(overall_fpr=round(float(np.mean(agg[m]["fpr"])), 3),
                     tpr_mean=round(float(np.nanmean(tbm)), 3),
                     tpr_by_band=[round(float(x), 3) for x in tbm],
                     tpr_across_band_cov=round(float(np.mean(agg[m]["tpr_cov"])), 3))
        log(f"  {m}: FPR {tt[m]['overall_fpr']} | TPR@FPR mean {tt[m]['tpr_mean']} | across-band CoV {tt[m]['tpr_across_band_cov']}")
    res["threshold_transfer"] = {"seeds": SEEDS, "bands": [f"{lo}-{hi}" for lo, hi in bands], "arms": tt}

    json.dump(res, open("results/recipe.json", "w"), indent=2)
    log("saved results/recipe.json")
    print("[train] === DONE ==="); sys.stdout.flush()


if __name__ == "__main__":
    main()
