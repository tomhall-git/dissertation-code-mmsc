# experiments/5_4_structure/post_processing.py
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plotting code for experiment 4

HERE = os.path.dirname(__file__)
CANDIDATES = [
    os.path.join(HERE, "results", "timestep_table.csv"),
    os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "timestep_table.csv"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "timestep_table.csv"),
]

CSV_PATH = None
for p in CANDIDATES:
    if os.path.isfile(p):
        CSV_PATH = p
        break

if CSV_PATH is None:
    raise FileNotFoundError("Could not find timestep_table.csv")

OUTDIR = os.path.join(HERE, "results")
os.makedirs(OUTDIR, exist_ok=True)

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    rows = [row for row in reader]

dt = float(rows[0]["dt"])
nsteps = len(rows)
t = np.linspace(dt, nsteps * dt, nsteps)
min_psihat = np.array([float(r["min_psihat"]) for r in rows])
neg_mass   = np.array([float(r["neg_mass_psihat"]) for r in rows])

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

fig, ax1 = plt.subplots(figsize=(7.5, 3.6))

ln1, = ax1.plot(t, min_psihat, "--", linewidth=0.5, color="#00008B", label=r"$\min \hat\psi$")
ax1.set_xlabel(r"$\mathrm{time} (t)$")
ax1.set_ylabel(r"$\min \hat\psi$")
ax1.tick_params(axis="y")

ax1.set_yticks([-1e7, -5e7])
ax1.set_yticklabels([r"$-1 \times 10^7$", r"$-5 \times 10^7$"])

ax2 = ax1.twinx()
ln2, = ax2.plot(t, neg_mass, linewidth=1.5, color="#8B0000", label=r"$\mathrm{neg\_mass}(\hat\psi)$")
ax2.set_ylabel(r"$\mathrm{total\ negative\ mass}(\hat\psi)$")
ax2.tick_params(axis="y")
ax2.invert_yaxis()

ax2.set_yticks([-1e-1])
ax2.set_yticklabels([r"$-10^{-1}$"])

lines = [ln1, ln2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="lower right", frameon=False)

fig.tight_layout()
out_path = os.path.join(OUTDIR, "min_negmass_timeseries.png")
fig.savefig(out_path, dpi=1200)
plt.close(fig)

print("CSV:", CSV_PATH)
print("Saved figure:", out_path)