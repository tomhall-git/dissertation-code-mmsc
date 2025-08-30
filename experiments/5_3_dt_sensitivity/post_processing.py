# experiments/5_3_dt_sensitivity/post_processing.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Plotting for experiment 3

HERE = os.path.dirname(__file__)
CANDIDATES = [
    os.path.join(HERE, "results", "stability_timeseries.csv"),
    os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "stability_timeseries.csv"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "results", "stability_timeseries.csv"),
    "/Users/tomhall/Documents/Dissertation Materials/fokker-planck-fem/results/stability_timeseries.csv",
]

CSV_PATH = None
for p in CANDIDATES:
    if os.path.isfile(p):
        CSV_PATH = p
        break

if CSV_PATH is None:
    raise FileNotFoundError("Could not find stability_timeseries.csv")

OUTDIR = os.path.join(HERE, "results")
os.makedirs(OUTDIR, exist_ok=True)

data = np.genfromtxt(CSV_PATH, delimiter=",", names=True, dtype=float, encoding="utf-8")
t    = data["t"]
mass = data["mass_psi"]
negm = data["negative_mass_psihat"]

def sci_10x_formatter():
    def _fmt(y, _):
        if y == 0:
            return "0"
        sign = "-" if y < 0 else ""
        ay = abs(y)
        exp = int(np.floor(np.log10(ay)))
        coef = ay / (10 ** exp)
        if np.isclose(coef, 1.0, atol=1e-8, rtol=0.0):
            return rf"{sign}$10^{{{exp}}}$"
        return rf"{sign}${coef:.1f}\times 10^{{{exp}}}$"
    return FuncFormatter(_fmt)

plt.rcParams.update({
    "text.usetex": True,        
    "font.family": "serif",      
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# Plot mass vs neg. mass

fig, ax1 = plt.subplots(figsize=(7.5, 3.6))

ln1, = ax1.plot(t, mass, linewidth=1.5, color = '#00008B', label=r"$\mathrm{mass}(\hat\psi)$")
ax1.set_xlabel(r"$\mathrm{time}(t)$")
ax1.set_ylabel(r"$\mathrm{total\ mass}(\hat\psi)$")

ax2 = ax1.twinx()
ln2, = ax2.plot(t, negm, "--", linewidth=0.5, color = '#8B0000', label=r"$\mathrm{neg\_mass}(\hat\psi)$")
ax2.set_ylabel(r"$\mathrm{negative\ mass}(\hat\psi)$")


lines = [ln1, ln2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="lower right", frameon=False)

ax1.set_yticks([1e2])
ax1.set_yticklabels([r"$10^2$"])

ax2.set_yticks([-1e-1, -1e0])
ax2.set_yticklabels([r"$-10^{-1}$", r"$-1$"])
ax2.invert_yaxis()

ax1.grid(False)
ax2.grid(False)

fig.tight_layout()
out_path = os.path.join(OUTDIR, "mass_negmass_clean.png")
fig.savefig(out_path, dpi=1200)
plt.close(fig)

print("CSV:", CSV_PATH)
print("Saved figure:", out_path)