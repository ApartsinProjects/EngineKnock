"""Mechanism figure for why REMOVAL (order tracking) is the worst treatment under speed shift.
Left: mean healthy PHYSICAL-Hz spectra for low/mid/high speed bands -- fixed structural resonances
sit at the SAME frequencies regardless of speed. Right: the same healthy windows in the ORDER domain
-- those fixed-Hz resonances MIGRATE across order bins as speed changes, so order normalization smears
the very energy that carries fault contrast. This is the physical reason conditioning (which keeps the
Hz structure and subtracts the speed-indexed healthy mean) beats removal. Uses feats.npz."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 50_000; NFFT = 8192; SPER = 256; REVS = 16
D = np.load("feats.npz")
HZ, ORD, S = D["Nh_HZ"], D["Nh_ORD"], D["Nh_S"]
bands = [(12, 22, "low 12-22 Hz"), (28, 40, "mid 28-40 Hz"), (48, 61, "high 48-61 Hz")]
cols = ["#1f77b4", "#ff7f0e", "#d62728"]

hz_axis = np.fft.rfftfreq(NFFT, 1 / FS)                 # physical Hz for HZ rep
ord_axis = np.arange(ORD.shape[1]) / SPER               # orders (cycles per revolution) for ORD rep

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
hzmax = 2000  # focus on the structural-resonance band
hz_m = hz_axis <= hzmax
ord_m = ord_axis <= (hzmax / (S.mean()))                # comparable order span
for (lo, hi, lab), c in zip(bands, cols):
    m = (S >= lo) & (S < hi)
    if m.sum() < 3:
        continue
    ax[0].plot(hz_axis[hz_m], HZ[m][:, hz_m].mean(0), c, lw=1.1, label=f"{lab} (n={m.sum()})")
    ax[1].plot(ord_axis[ord_m], ORD[m][:, ord_m].mean(0), c, lw=1.1, label=lab)
ax[0].set_title("Physical-Hz domain (ignore/condition)\nresonances fixed across speed")
ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("log magnitude"); ax[0].legend(fontsize=8)
ax[1].set_title("Order domain (remove / order tracking)\nsame resonances migrate across orders")
ax[1].set_xlabel("order (cycles / revolution)"); ax[1].legend(fontsize=8)
for a in ax:
    a.grid(alpha=0.25)
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/fig_mechanism.png", dpi=150)
print("[fig] wrote results/fig_mechanism.png", flush=True)
