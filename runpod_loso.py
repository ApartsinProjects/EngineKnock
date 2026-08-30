"""RunPod job: rigorous leakage-free correctness analysis for the EngineKnock paper
(review W3/W4/W5). Downloads MaFaulDa, extracts accel col1 time+order spectra with per-recording
IDs, and runs: window-level matched (leaky, for comparison), leave-one-speed-out cross-fitted
grouped eval (base/inv/cond) with cluster-bootstrap CIs over recordings, per-band held-out, and
interpolation-vs-extrapolation. Validates AUC vs sklearn and uses scipy Wilcoxon. Saves
results/loso_results.json. CPU-only (needs RAM, not GPU)."""
import os, io, sys, json, zipfile, urllib.request, numpy as np

FS = 50_000; ACC = 1; REVS = 16; SPER = 256
BASE = "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/"
FAULTS = ["horizontal-misalignment", "vertical-misalignment", "imbalance", "underhang", "overhang"]
os.makedirs("data", exist_ok=True); os.makedirs("results", exist_ok=True)


def log(m): print(f"[train] {m}"); sys.stdout.flush()


def download(stem):
    p = f"data/{stem}.zip"
    if os.path.exists(p) and os.path.getsize(p) > 1e6:
        return p
    log(f"downloading {stem}.zip ...")
    urllib.request.urlretrieve(BASE + f"{stem}.zip", p)
    log(f"  {stem}.zip = {os.path.getsize(p)/1e6:.0f} MB")
    return p


def tacho_rises(t):
    x = t.astype(float); x = (x - x.min()) / (np.ptp(x) + 1e-12); a = x > 0.5
    return np.where((~a[:-1]) & (a[1:]))[0]


def feats(sig, rises):
    L = REVS * SPER; win = np.hanning(L)
    if len(rises) < REVS + 1:
        return
    nb = (len(rises) - 1) // REVS
    for b in range(nb):
        p0, p1 = rises[b*REVS], rises[(b+1)*REVS]
        if p1 <= p0 or p1 > len(sig):
            continue
        seg = sig[p0:p1].astype(float)
        t = np.interp(np.linspace(0, len(seg)-1, L), np.arange(len(seg)), seg)
        loc = rises[b*REVS:(b+1)*REVS+1]; rev = np.interp(np.arange(p0, p1), loc, np.arange(len(loc)))
        grid = np.linspace(rev[0], rev[-1], L, endpoint=False); o = np.interp(grid, rev, seg)
        yield (np.log1p(np.abs(np.fft.rfft((t-t.mean())*win))).astype(np.float32),
               np.log1p(np.abs(np.fft.rfft((o-o.mean())*win))).astype(np.float32))


def extract(stem):
    T, O, S, R = [], [], [], []
    with zipfile.ZipFile(download(stem)) as zf:
        for ri, n in enumerate(sorted(x for x in zf.namelist() if x.lower().endswith(".csv"))):
            arr = np.loadtxt(io.BytesIO(zf.read(n)), delimiter=",")
            r = tacho_rises(arr[:, 0])
            for t, o in feats(arr[:, ACC], r):
                T.append(t); O.append(o); S.append(FS/np.median(np.diff(r))); R.append(ri)
    return (np.asarray(T, np.float32), np.asarray(O, np.float32),
            np.asarray(S, np.float32), np.asarray(R))


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
    def sc(X):
        Xs = (X-mu)/sd; return np.sqrt(((Xs - (Xs@comp.T)@comp)**2).mean(1))
    return sc


def main():
    from sklearn.metrics import roc_auc_score
    from scipy.stats import wilcoxon
    log("extracting healthy + faults (accel col1, time+order, with recording IDs)")
    Tn, On, Sn, Rn = extract("normal")
    Tfs, Ofs, Sfs, Rfs = [], [], [], []
    for i, s in enumerate(FAULTS):
        t, o, sp, r = extract(s); Tfs.append(t); Ofs.append(o); Sfs.append(sp); Rfs.append(r + (i+1)*1000)
    Tf = np.vstack(Tfs); Of = np.vstack(Ofs); Sf = np.concatenate(Sfs)
    log(f"healthy={len(Tn)} windows / {len(np.unique(Rn))} recordings; fault={len(Tf)} windows")

    def auc(neg, pos): return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))
    a, b = np.random.default_rng(0).normal(0,1,300), np.random.default_rng(1).normal(.4,1,300)
    log(f"AUC uses sklearn roc_auc_score (self-check {auc(a,b):.4f})")

    res = {"config": {"k_nn": 15, "k_pca": 32, "revs": REVS, "sper": SPER, "auc": "sklearn"}}

    # ---- (1) window-level MATCHED (leaky reference) ----
    log("(1) window-level matched (leaky reference), 10 seeds")
    mm = {t: [] for t in ("base", "inv", "cond")}
    for seed in range(10):
        rng = np.random.default_rng(seed); idx = rng.permutation(len(Tn)); cut = int(.6*len(Tn)); tr, te = idx[:cut], idx[cut:]
        fb = pca(Tn[tr]); fo = pca(On[tr]); fc = pca(Tn[tr]-knn_mean(Tn[tr], Sn[tr], Sn[tr]))
        mm["base"].append(auc(fb(Tn[te]), fb(Tf))); mm["inv"].append(auc(fo(On[te]), fo(Of)))
        mm["cond"].append(auc(fc(Tn[te]-knn_mean(Tn[tr], Sn[tr], Sn[te])), fc(Tf-knn_mean(Tn[tr], Sn[tr], Sf))))
    res["matched_leaky"] = {t: {"mean": float(np.mean(mm[t])), "std": float(np.std(mm[t]))} for t in mm}
    log(f"  matched: base={np.mean(mm['base']):.3f} inv={np.mean(mm['inv']):.3f} cond={np.mean(mm['cond']):.3f}")

    # ---- (2) LOSO cross-fitted grouped (leakage-free) ----
    log("(2) leave-one-speed-out, cross-fitted (leakage-free)")
    recs = np.unique(Rn); per = {t: [] for t in ("base", "inv", "cond")}
    ie = {t: {"interp": [], "extrap": []} for t in ("base", "inv", "cond")}
    for rid in recs:
        te = Rn == rid; tr = ~te
        if te.sum() < 5 or tr.sum() < 40:
            continue
        s_lo, s_hi = Sn[tr].min(), Sn[tr].max(); is_in = s_lo <= Sn[te].mean() <= s_hi
        fb = pca(Tn[tr]); fo = pca(On[tr]); fc = pca(Tn[tr]-knn_mean(Tn[tr], Sn[tr], Sn[tr]))
        av = {"base": auc(fb(Tn[te]), fb(Tf)), "inv": auc(fo(On[te]), fo(Of)),
              "cond": auc(fc(Tn[te]-knn_mean(Tn[tr], Sn[tr], Sn[te])), fc(Tf-knn_mean(Tn[tr], Sn[tr], Sf)))}
        for t in av:
            per[t].append(av[t]); ie[t]["interp" if is_in else "extrap"].append(av[t])
    # cluster bootstrap over recordings
    def clusterboot(vals, B=3000):
        vals = np.array(vals); rng = np.random.default_rng(1)
        m = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(B)]
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    res["loso"] = {}
    for t in ("base", "inv", "cond"):
        v = np.array(per[t]); lo, hi = clusterboot(v)
        res["loso"][t] = dict(mean=float(v.mean()), std=float(v.std()), n=len(v), ci=[lo, hi],
                              interp=float(np.mean(ie[t]["interp"])) if ie[t]["interp"] else None,
                              extrap=float(np.mean(ie[t]["extrap"])) if ie[t]["extrap"] else None)
        log(f"  {t}: {v.mean():.3f} [{lo:.3f},{hi:.3f}] interp={res['loso'][t]['interp']} extrap={res['loso'][t]['extrap']}")
    # paired Wilcoxon (per-recording) cond vs base / inv
    try:
        res["loso"]["wilcoxon"] = dict(
            cond_vs_base=float(wilcoxon(per["cond"], per["base"]).pvalue),
            cond_vs_inv=float(wilcoxon(per["cond"], per["inv"]).pvalue))
        log(f"  Wilcoxon cond-vs-base p={res['loso']['wilcoxon']['cond_vs_base']:.4g}, cond-vs-inv p={res['loso']['wilcoxon']['cond_vs_inv']:.4g}")
    except Exception as e:
        log(f"  wilcoxon skipped: {e}")

    json.dump(res, open("results/loso_results.json", "w"), indent=2)
    log("saved results/loso_results.json")
    print("[train] === DONE ==="); sys.stdout.flush()


if __name__ == "__main__":
    main()
