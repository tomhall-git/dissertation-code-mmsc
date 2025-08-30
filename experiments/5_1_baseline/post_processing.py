# experiments/5_1_baseline/post_processing.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

# To produce mass conservation plot

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "results", "mass_timeseries.csv")
OUTDIR = os.path.join(HERE, "results")
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "mass_vs_time_linear.png")

# LaTeX style
plt.rcParams.update({
    "text.usetex": True,     
    "font.family": "serif", 
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

def main():
    if not os.path.exists(CSV):
        raise FileNotFoundError(f"Could not find {CSV}")

    # Load only the numeric columns we need
    data = np.genfromtxt(
        CSV, delimiter=",", names=True, encoding=None,
        usecols=("time", "mass")
    )
    t = np.asarray(data["time"], dtype=float)
    mass = np.asarray(data["mass"], dtype=float)

    # Initial mass and residual
    M0 = mass[0]
    residual = np.abs(mass - M0)

    variation = float(np.ptp(mass))       
    span = max(1.0, 50.0 * variation)     
    ymin, ymax = M0 - 0.5 * span, M0 + 0.5 * span

    # Dual axes plot
    fig, ax1 = plt.subplots(figsize=(7.5, 3.6))

    # Left axis: mass
    line_mass, = ax1.plot(t, mass, linewidth=1.5, color="#00008B", label=r"total mass ($\psi$)")
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$m(t)$")
    ax1.set_ylim(ymin, ymax)

    fmt_left = ScalarFormatter(useOffset=False)
    fmt_left.set_scientific(False)
    ax1.yaxis.set_major_formatter(fmt_left)

    ax1.xaxis.set_major_locator(MaxNLocator(5))
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    ax1.grid(False)

    # Right axis: residual
    ax2 = ax1.twinx()
    line_resid, = ax2.plot(
        t, residual, linewidth=1.0, linestyle="--", color="darkred",
        label=r"$|m(t)-m(0)|$"
    )
    ax2.set_ylabel(r"$|m(t)-m(0)|$")
    ax2.set_yscale("log")

    # Force only one tick at 1e-8 for readability
    ax2.set_yticks([1e-8])
    ax2.get_yaxis().set_major_formatter(mticker.LogFormatterMathtext())
    ax2.yaxis.set_minor_locator(mticker.NullLocator())

    # Shared legend
    lines = [line_mass, line_resid]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", frameon = False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure:", OUT)

if __name__ == "__main__":
    main()