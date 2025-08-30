# experiments/5_2_mesh/runx.py
import os, json, numpy as np, time
from dolfinx import fem
import ufl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...solvers.main_solver import main_solver
from ...domain.mesh_x import get_domain_x
from ...domain.mesh_q import get_domain_q
from ...utils.parameters import b

# Convergence analysis in x
H_X = [0.5, 0.35, 0.25]
H_Q_FIXED = 0.85
OUTROOT = os.path.join(os.path.dirname(__file__), "results")

def lump_weights(V):
    from dolfinx.fem import petsc as fp
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    A = fp.assemble_matrix(fem.form(ufl.inner(u, v) * ufl.dx)); A.assemble()
    w = A.createVecLeft(); A.getRowSum(w)
    return w.array.copy()

def build_spaces_and_weights(lc_x, lc_q):
    mesh_x = get_domain_x(lc=lc_x)
    mesh_q = get_domain_q(lc=lc_q, radius=np.sqrt(b), b=b)
    Vx = fem.functionspace(mesh_x, ("CG", 1))
    Vq = fem.functionspace(mesh_q, ("CG", 1))
    omega_x = lump_weights(Vx); omega_q = lump_weights(Vq)
    # M(q) nodal values
    Mq_fun = fem.Function(Vq)
    def Mq_callable(x):
        r2 = x[0]**2 + x[1]**2
        return np.maximum(1.0 - r2 / b, 0.0)**(b/2)
    Mq_fun.interpolate(Mq_callable)
    return Vx, Vq, omega_x, omega_q, Mq_fun.x.array.copy()

def _nearest_map_indices(coords_src, coords_tgt):

    idx = np.empty(coords_tgt.shape[0], dtype=np.int64)
    for i in range(coords_tgt.shape[0]):
        d2 = np.sum((coords_src - coords_tgt[i, :])**2, axis=1)
        idx[i] = int(np.argmin(d2))
    return idx

def interpolate_fine_to_coarse_x(table_fine, Vx_fine, Vx_coarse):

    Xf = Vx_fine.tabulate_dof_coordinates()[:, :Vx_fine.mesh.geometry.dim]
    Xc = Vx_coarse.tabulate_dof_coordinates()[:, :Vx_coarse.mesh.geometry.dim]
    take = _nearest_map_indices(Xf, Xc)
    return table_fine[take, :].copy()

def l2x_weighted(tableA, tableB, omega_x, omega_q, Mq_vals):
    diff = tableA - tableB
    col_norms2 = (omega_x[:, None] * diff**2).sum(axis=0)
    return float(np.sqrt((col_norms2 * (omega_q * Mq_vals)).sum()))

def plot_convergence_x(errs, out_png, out_pdf):
    hx = np.array([e["hx"] for e in errs], float)
    E  = np.array([e["E"]  for e in errs], float)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.loglog(hx, E, marker="o", linewidth=1.5, label=r"$E_x$ (q\ fixed)")
    if len(hx) >= 2:
        hA, EA = hx[-1], E[-1]
        ax.loglog(hx, EA * (hx/hA)**2, "--", linewidth=1.0, label=r"$\mathcal{O}(h_x^2)$")
    for i in range(len(hx)-1):
        p = np.log(E[i]/E[i+1]) / np.log(hx[i]/hx[i+1])
        xm, ym = np.sqrt(hx[i]*hx[i+1]), np.sqrt(E[i]*E[i+1])
        ax.text(xm, ym, f"p≈{p:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel(r"$h_x$"); ax.set_ylabel(r"$E_x$ (M-weighted, lumped $L^2$)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(out_png, dpi=200); fig.savefig(out_pdf); plt.close(fig)

def run():
    os.makedirs(OUTROOT, exist_ok=True)

    runs = []
    for hx in H_X:
        outdir = os.path.join(OUTROOT, f"x_{hx:.3f}")
        os.makedirs(outdir, exist_ok=True)

        Vx_tmp, Vq_tmp, _, _, _ = build_spaces_and_weights(hx, H_Q_FIXED)
        Nx = Vx_tmp.dofmap.index_map.size_local; Nq = Vq_tmp.dofmap.index_map.size_local
        print(f"[x-accuracy] hx={hx:.3f}, hq={H_Q_FIXED:.3f}  (Nx={Nx}, Nq={Nq}) → {outdir}")
        t0 = time.time()
        table = main_solver(
            lc_x=hx, lc_q=H_Q_FIXED,
            max_fp_iter=6, fp_tol=5e-3, fp_verbose=False,
            relax_theta=0.3, psi_cap=1e6,
            outdir=outdir,
        )
        np.save(os.path.join(outdir, "psihat_final.npy"), table)
        print(f"[x-accuracy] done hx={hx:.3f} in {time.time()-t0:.1f}s")
        runs.append((hx, table))

    # spaces/weights (built once per level)
    spaces = {hx: dict(zip(
        ("Vx","Vq","omega_x","omega_q","Mq_vals"),
        build_spaces_and_weights(hx, H_Q_FIXED)
    )) for hx,_ in runs}

    # errors vs finest
    hx_f, tbl_f = runs[-1]; Vx_f = spaces[hx_f]["Vx"]
    errs = []
    for hx, tbl_c in runs[:-1]:
        Vx_c = spaces[hx]["Vx"]
        Wxi, Wq, Mq = spaces[hx]["omega_x"], spaces[hx]["omega_q"], spaces[hx]["Mq_vals"]
        tbl_f_on_c = interpolate_fine_to_coarse_x(tbl_f, Vx_f, Vx_c)
        E = l2x_weighted(tbl_c, tbl_f_on_c, Wxi, Wq, Mq)
        errs.append({"hx": hx, "E": E})

    errs.sort(key=lambda r: r["hx"], reverse=True)
    for i in range(len(errs)-1):
        h1,e1 = errs[i]["hx"], errs[i]["E"]; h2,e2 = errs[i+1]["hx"], errs[i+1]["E"]
        errs[i]["p"] = float(np.log(e1/e2)/np.log(h1/h2))

    out_json = os.path.join(OUTROOT, "convergence_x.json")
    with open(out_json, "w") as f: json.dump(errs, f, indent=2)
    print("\n[Convergence (x-refinement)]")
    for r in errs:
        print(f"h_x={r['hx']:.3f}  E={r['E']:.3e}" + (f"  p≈{r['p']:.2f}" if 'p' in r else ""))
    print("Saved:", out_json)

    png = os.path.join(OUTROOT, "convergence_x.png")
    pdf = os.path.join(OUTROOT, "convergence_x.pdf")
    plot_convergence_x(errs, png, pdf)
    print("Saved plots:", png, "and", pdf)

if __name__ == "__main__":
    run()