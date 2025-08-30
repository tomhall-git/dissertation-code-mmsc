# experiments/5_2_mesh/post_processing_x.py
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

# Post processing for figure 

OUTROOT = os.path.join(os.path.dirname(__file__), "results")

def main():
    with open(os.path.join(OUTROOT, "convergence_x.json")) as f:
        errs = json.load(f)

    errs = sorted(errs, key=lambda r: r["hx"])
    hx = np.array([r["hx"] for r in errs], float)
    Ex = np.array([r["E"]  for r in errs], float)

    plt.rcParams.update({
        "text.usetex": True,        
        "font.family": "serif",      
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    # data
    ax.loglog(hx, Ex, color = "r", linewidth=2.0, label=r"$E_x$ ($q$ fixed)")

    # reference O(h) anchored at finest point
    h_ref, E_ref = hx[-1], Ex[-1]
    shift = 1.1
    ax.loglog(hx, shift * E_ref * (hx / h_ref), "--", color="k", linewidth=1.2,
              label=r"$\mathcal{O}(h_x)$")

    ax.set_xlabel(r"$h_x$")
    ax.set_ylabel(r"$E_x$ (M-weighted lumped $L^2$-norm)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(False)
    ax.set_yscale("log")

    ymin, ymax = ax.get_ylim()
    ymid = 10 ** (0.5 * (np.log10(ymin) + np.log10(ymax)))
    ax.set_yticks([ymid])
    ax.set_yticklabels([r"$10^{-2}$"])
    ax.yaxis.set_minor_locator(NullLocator())  # remove minor ticks

    fig.tight_layout()
    png = os.path.join(OUTROOT, "convergence_x.png")
    pdf = os.path.join(OUTROOT, "convergence_x.pdf")
    fig.savefig(png, dpi=1200); fig.savefig(pdf)
    print("Saved:", png, "and", pdf)

if __name__ == "__main__":
    main()