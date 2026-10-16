"""Second-benchmark frame-blindness test on the KFUPM cross-speed set (Mendeley y8mv28hjwb).
Same-session normal+imbalance (confound-clean: normal Apr12-13, imbalance starts 6 min after normal
ends; misalignment is 8 days later so it is EXCLUDED as cross-session). Speed is constant per 1-s
segment (600/900/1200 rpm), so the competing-equivariance effect appears across speeds: a fixed-Hz
resonance lands at different orders at each rpm (smeared in the order frame), a 1x synchronous component
lands at different Hz (smeared in the physical-frequency frame).

Two things:
  (A) CONFOUND CHECK (run-disjoint placebo, matched speed): train on some normal runs at one speed, score
      other normal runs vs yet others, all healthy -> want ~0.5 (no run/segment-identity shortcut).
  (B) FRAME BLINDNESS (imbalance), leave-one-speed-out: per-frame AUC, HZ (physical-frequency, fixed bins)
      vs ORD (rpm-normalized order spectrum). Prediction (replicating MaFaulDa): HZ near chance on
      imbalance (1x smeared across 10/15/20 Hz), ORD detects it (always order 1).
Channel 1. Cache features to results/kfupm_feats.npz."""
import os, io, zipfile, glob, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
import runpod_strongaccept_v2 as J   # for the speed-conditioned (kNN-in-speed mean) detector

FS = 25600; NFFT_HZ = 8192; REVS = 8; SPER = 256; L = REVS * SPER
FEAT = "results/kfupm_feats.npz"


def log(m): print(f"[kfupm] {m}", flush=True)


def seg_features(sig, rpm):
    sig = sig.astype(np.float64); sig = sig - sig.mean()
    # HZ: physical-frequency, fixed bins (first NFFT_HZ samples)
    h = np.zeros(NFFT_HZ); n = min(len(sig), NFFT_HZ); h[:n] = sig[:n] * np.hanning(NFFT_HZ)[:n] if n == NFFT_HZ else sig[:n] * np.hanning(n)
    HZ = np.log1p(np.abs(np.fft.rfft(h)))
    # ORD: resample by known constant rpm to a fixed samples-per-rev grid over REVS revolutions
    spr = FS / (rpm / 60.0)                       # samples per revolution at this constant speed
    need = int(REVS * spr)
    if need > len(sig): need = len(sig)
    seg = sig[:need]
    grid = np.linspace(0, len(seg) - 1, L)
    o = np.interp(grid, np.arange(len(seg)), seg) * np.hanning(L)
    ORD = np.log1p(np.abs(np.fft.rfft(o)))
    return HZ.astype(np.float32), ORD.astype(np.float32)


def extract():
    if os.path.exists(FEAT):
        d = np.load(FEAT, allow_pickle=True); log("feats cache HIT"); return {k: d[k] for k in d.files}
    HZ, ORD, SP, CON, RUN = [], [], [], [], []
    for zp in sorted(glob.glob("data/kfupm/*.zip")):
        cond = "normal" if "normal" in zp else "imbalance"
        with zipfile.ZipFile(zp) as z:
            mats = [x for x in z.namelist() if x.endswith(".mat")]
            for nm in mats:
                m = loadmat(io.BytesIO(z.read(nm)))
                sig = np.asarray(m["signals"])[0]      # channel 1
                rpm = int(np.asarray(m["speed_rpm"]).ravel()[0])
                run = str(np.asarray(m["original_file"]).ravel()[0])
                hz, od = seg_features(sig, rpm)
                HZ.append(hz); ORD.append(od); SP.append(rpm); CON.append(cond); RUN.append(run)
        log(f"  {os.path.basename(zp)}: cumulative {len(SP)} segments")
    out = dict(HZ=np.vstack(HZ), ORD=np.vstack(ORD), SP=np.array(SP), CON=np.array(CON), RUN=np.array(RUN))
    os.makedirs("results", exist_ok=True); np.savez_compressed(FEAT, **out); log(f"saved {FEAT}")
    return out


def pca(Xtr, k=32):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    _, _, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False); comp = Vt[:min(k, max(1, Xtr.shape[0] - 2))]
    return lambda X: np.sqrt((((X - mu) / sd - (((X - mu) / sd) @ comp.T) @ comp) ** 2).mean(1))


def auc(a, b): return float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]))
def boot(v, n=2000):
    v = np.asarray(v); rng = np.random.default_rng(1)
    return [round(float(np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)], p)), 3) for p in (2.5, 97.5)]


def main():
    d = extract()
    HZ, ORD, SP, CON, RUN = d["HZ"], d["ORD"], d["SP"], d["CON"], d["RUN"]
    H = CON == "normal"; IMB = CON == "imbalance"
    speeds = sorted(np.unique(SP).tolist())
    log(f"normal {H.sum()} / imbalance {IMB.sum()}; speeds {speeds}; dims HZ{HZ.shape[1]} ORD{ORD.shape[1]}")
    res = {}

    SPf = SP.astype(np.float64)
    def cond_scorer(X, trmask):
        mtr = J.knn_mean(X[trmask], SPf[trmask], SPf[trmask]); f = J.pca(X[trmask] - mtr)
        return lambda Xq, Sq: f(Xq - J.knn_mean(X[trmask], SPf[trmask], Sq))

    # (A) run-disjoint placebo at matched speed, speed-conditioned (healthy only) -> want ~0.5
    rng = np.random.default_rng(0); plac = {"HZ": [], "ORD": []}
    for sp in speeds:
        for fr, X in [("HZ", HZ), ("ORD", ORD)]:
            runs = sorted(np.unique(RUN[H & (SP == sp)]).tolist())
            if len(runs) < 6: continue
            rng.shuffle(runs); a = runs[:len(runs)//3]; b = runs[len(runs)//3:2*len(runs)//3]; c = runs[2*len(runs)//3:]
            ma = H & (SP == sp) & np.isin(RUN, a); mb = H & (SP == sp) & np.isin(RUN, b); mc = H & (SP == sp) & np.isin(RUN, c)
            if ma.sum() < 20 or mb.sum() < 10 or mc.sum() < 10: continue
            S = cond_scorer(X, ma); plac[fr].append(auc(S(X[mb], SPf[mb]), S(X[mc], SPf[mc])))
    res["placebo_matched_speed"] = {fr: {"auc": round(float(np.mean(v)), 3), "n": len(v)} for fr, v in plac.items() if v}

    # (B) frame blindness on imbalance, leave-one-speed-out, SPEED-CONDITIONED detector per frame
    fb = {"HZ": [], "ORD": []}
    for held in speeds:
        trH = H & (SP != held); teH = H & (SP == held); teF = IMB & (SP == held)
        if teH.sum() < 10 or teF.sum() < 10: continue
        for fr, X in [("HZ", HZ), ("ORD", ORD)]:
            S = cond_scorer(X, trH); fb[fr].append(auc(S(X[teH], SPf[teH]), S(X[teF], SPf[teF])))
    res["imbalance_frame_auc"] = {fr: {"auc": round(float(np.mean(v)), 3), "ci95": boot(v), "n_folds": len(v)} for fr, v in fb.items()}
    res["verdict"] = ("REPLICATES (HZ blind to imbalance, ORD detects)"
                      if res["imbalance_frame_auc"]["ORD"]["auc"] - res["imbalance_frame_auc"]["HZ"]["auc"] > 0.1
                      else "does not replicate")
    json.dump(res, open("results/kfupm_frameblind.json", "w"), indent=2)
    log("placebo (want ~0.5): " + json.dumps(res["placebo_matched_speed"]))
    log("imbalance per-frame: " + json.dumps(res["imbalance_frame_auc"]))
    log("VERDICT: " + res["verdict"])
    log("=== DONE ===")


if __name__ == "__main__":
    main()
