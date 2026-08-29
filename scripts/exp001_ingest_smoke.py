"""EXP-001 MaFaulDa ingest + order-resampling smoke test.

Reads a few files directly from data/normal.zip (no full extract), confirms channel
layout, estimates rotation frequency from the tachometer, order-resamples one accel
channel to fixed samples/rev, and saves an overlay plot for visual cycle-alignment check.

Channel layout (per MaFaulDa docs): col0 tachometer; col1-3 underhang accel (axial,
radial, tangential); col4-6 overhang accel; col7 microphone. fs=50 kHz, 5 s, 250k rows.
Normal files are named by nominal rotation frequency in Hz (e.g. "12.288.csv").
"""
import io, os, sys, zipfile
import numpy as np

FS = 50_000
ZIP = os.path.join(os.path.dirname(__file__), "..", "data", "normal.zip")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def list_files(zf):
    return sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))


def load_csv_bytes(b):
    return np.loadtxt(io.BytesIO(b), delimiter=",")


def est_rot_hz_from_tacho(tacho, fs=FS):
    """One pulse per revolution: detect rising edges at mid-amplitude, f = fs/mean_gap."""
    x = tacho.astype(float)
    x = (x - x.min()) / (np.ptp(x) + 1e-12)
    thr = 0.5
    above = x > thr
    rises = np.where((~above[:-1]) & (above[1:]))[0]
    if len(rises) < 3:
        return np.nan, 0
    gaps = np.diff(rises)
    gap = np.median(gaps)
    return fs / gap, len(rises)


def order_resample(sig, tacho, samples_per_rev=720, fs=FS):
    """Map each sample to cumulative revolutions via tacho pulses, then resample the signal
    onto a uniform revolution grid at `samples_per_rev` points/rev."""
    x = tacho.astype(float)
    x = (x - x.min()) / (np.ptp(x) + 1e-12)
    above = x > 0.5
    rises = np.where((~above[:-1]) & (above[1:]))[0]
    if len(rises) < 3:
        return None, None
    # cumulative revolution index at each pulse = 0,1,2,...; interpolate rev vs sample-index
    n = len(sig)
    idx = np.arange(n)
    rev_at_pulse = np.arange(len(rises))
    rev = np.interp(idx, rises, rev_at_pulse, left=np.nan, right=np.nan)
    valid = ~np.isnan(rev)
    rev_v, sig_v = rev[valid], sig[valid]
    r0, r1 = np.ceil(rev_v[0]), np.floor(rev_v[-1])
    grid = np.arange(r0, r1, 1.0 / samples_per_rev)
    res = np.interp(grid, rev_v, sig_v)
    return grid, res.reshape(-1, samples_per_rev)  # (n_rev, samples_per_rev)


def main():
    if not os.path.exists(ZIP):
        print("ZIP not present yet:", ZIP); sys.exit(2)
    with zipfile.ZipFile(ZIP) as zf:
        files = list_files(zf)
        print(f"normal.zip contains {len(files)} csv files")
        print("first/last:", files[0], files[-1])
        # pick 3 spanning the speed range
        pick = [files[0], files[len(files) // 2], files[-1]]
        summary = []
        for name in pick:
            b = zf.read(name)
            arr = load_csv_bytes(b)
            stem = os.path.splitext(os.path.basename(name))[0]
            nominal = float(stem) if stem.replace(".", "").isdigit() else np.nan
            frot, npulse = est_rot_hz_from_tacho(arr[:, 0])
            print(f"\n== {name} ==")
            print("  shape:", arr.shape)
            print("  per-col mean/std:",
                  [f"{arr[:,c].mean():.3f}/{arr[:,c].std():.3f}" for c in range(arr.shape[1])])
            print(f"  nominal RPM-Hz from name: {nominal:.3f} | tacho est: {frot:.3f} Hz "
                  f"({npulse} pulses) | ratio est/nominal: {frot/nominal:.3f}")
            summary.append((name, nominal, frot))
        # order-resample the largest-file underhang-axial channel (col1)
        name = pick[1]
        arr = load_csv_bytes(zf.read(name))
        grid, mat = order_resample(arr[:, 1], arr[:, 0], samples_per_rev=720)
        if mat is not None:
            print(f"\norder-resampled {name}: {mat.shape[0]} revolutions x {mat.shape[1]} samp/rev")
            np.save(os.path.join(OUT, "exp001_order_mat.npy"), mat)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                # left: overlay 20 revolutions in ANGLE domain (should align)
                for r in range(min(20, mat.shape[0])):
                    ax[0].plot(np.linspace(0, 360, mat.shape[1]), mat[r], lw=0.4, alpha=0.5)
                ax[0].set_title(f"order domain: 20 revs overlaid\n{os.path.basename(name)}")
                ax[0].set_xlabel("shaft angle (deg)"); ax[0].set_ylabel("underhang axial accel")
                # right: same count of FIXED-TIME windows (should NOT align as well)
                w = int(FS / est_rot_hz_from_tacho(arr[:, 0])[0])
                for r in range(20):
                    seg = arr[r * w:(r + 1) * w, 1]
                    ax[1].plot(np.linspace(0, 360, len(seg)), seg, lw=0.4, alpha=0.5)
                ax[1].set_title("fixed-time windows overlaid (control)")
                ax[1].set_xlabel("normalized window (deg-equiv)")
                fig.tight_layout()
                p = os.path.join(OUT, "exp001_order_overlay.png")
                fig.savefig(p, dpi=110)
                print("saved plot:", p)
            except Exception as e:
                print("plot skipped:", e)
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
