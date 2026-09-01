"""Detection-latency / regime-early-warning protocol on MaFaulDa (the operational claim).
Uses the condition-aware detector with a healthy-only 99th-pct threshold and an m-of-n persistence rule
(alarm when m of the last n windows exceed threshold). Two curves:
  (A) windows-to-alarm vs injected anomaly INTENSITY: inject alpha * (mean healthy->fault residual
      direction) into held-out HEALTHY windows; at alpha=0 the alarm rate is the false-alarm rate, and
      windows-to-alarm should fall as alpha grows. A controlled sensitivity calibration.
  (B) latency vs false-alarm trade-off: sweep the persistence rule n and report median windows-to-alarm
      on real faults against the healthy false-alarm rate under the same rule.
Loads the shared feature cache (feats.npz). CPU only."""
import os, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import runpod_strongaccept_v2 as J


def pca(Xtr, k=32):
    return J.pca(Xtr, k)


def persistence_alarm(over, m, n):
    """Return index of first window where m of the last n are over threshold, else -1."""
    c = np.convolve(over.astype(int), np.ones(n, int), "full")[:len(over)]
    hit = np.where(c >= m)[0]
    return int(hit[0]) if len(hit) else -1


def main():
    D = J.get_features()
    Xn, Sn, Rn = D["Nh_REV"], D["Nh_S"], D["Nh_R"]
    Xf, Sf, Ff = D["F_REV"], D["F_S"], D["F_FAULT"]
    # condition-aware detector on a 60% healthy split; threshold from training healthy
    rng = np.random.default_rng(0); idx = rng.permutation(len(Xn)); cut = int(.6 * len(Xn))
    tr, te = idx[:cut], idx[cut:]
    m_tr = J.knn_mean(Xn[tr], Sn[tr], Sn[tr]); f = pca(Xn[tr] - m_tr)
    def score(X, S): return f(X - J.knn_mean(Xn[tr], Sn[tr], S))
    thr = float(np.percentile(score(Xn[tr], Sn[tr]), 99))
    res = {"threshold_pct": 99, "n_test_healthy": int(len(te))}

    # fault residual direction (mean fault minus conditional mean), unit-normalized in feature space
    rf = Xf - J.knn_mean(Xn[tr], Sn[tr], Sf)
    direction = rf.mean(0); direction = direction / (np.linalg.norm(direction) + 1e-9)
    hres = Xn[te] - J.knn_mean(Xn[tr], Sn[tr], Sn[te])
    base_scale = np.sqrt((hres ** 2).sum(1)).mean()  # typical healthy residual norm

    # (A) windows-to-alarm vs injected intensity (persistence 3-of-5)
    m, n = 3, 5
    curveA = []
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        inj = Xn[te] + alpha * base_scale * direction  # inject along fault direction
        s = score(inj, Sn[te]); over = s > thr
        lat = persistence_alarm(over, m, n)
        curveA.append({"alpha": alpha, "alarm_rate": round(float(over.mean()), 3),
                       "windows_to_alarm": lat})
    res["injected_intensity_curve"] = {"rule": f"{m}-of-{n}", "points": curveA}

    # (B) latency vs FPR: per real fault recording, windows-to-alarm under rule; healthy FPR same rule
    bysweep = []
    for n_ in [1, 3, 5, 8]:
        m_ = max(1, n_ // 2 + 1)
        # healthy false alarm under rule (stream held-out healthy sorted by speed)
        order = np.argsort(Sn[te]); ho = (score(Xn[te][order], Sn[te][order]) > thr)
        h_alarm = persistence_alarm(ho, m_, n_)
        h_fpr = float(ho.mean())
        # fault latency: split faults into pseudo-streams by fault class, sorted by speed
        lats = []
        for fc in np.unique(Ff):
            msk = Ff == fc; o = np.argsort(Sf[msk])
            s = score(Xf[msk][o], Sf[msk][o]) > thr
            l = persistence_alarm(s, m_, n_)
            if l >= 0:
                lats.append(l)
        bysweep.append({"rule": f"{m_}-of-{n_}", "healthy_fpr": round(h_fpr, 3),
                        "median_windows_to_alarm": int(np.median(lats)) if lats else None,
                        "faults_detected": len(lats)})
    res["latency_fpr_tradeoff"] = bysweep

    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/latency.json", "w"), indent=2)
    print("[latency] injected:", json.dumps(res["injected_intensity_curve"]), flush=True)
    print("[latency] tradeoff:", json.dumps(res["latency_fpr_tradeoff"]), flush=True)
    print("[latency] === DONE ===", flush=True)


if __name__ == "__main__":
    main()
