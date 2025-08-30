# experiments/5_2_mesh/runq.py
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

# q convergence study - not used, but code framework included for completeness
HX_FIXED = 0.25
H_Q = [0.75, 0.65, 0.55]
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
        r2 = x[0] ** 2 + x[1] ** 2
        return np.maximum(1.0 - r2 / b, 0.0) ** (b / 2)

    Mq_fun.interpolate(Mq_callable)
    return Vx, Vq, omega_x, omega_q, Mq_fun.x.array.copy()


def _nearest_map_indices(coords_src, coords_tgt):
    idx = np.empty(coords_tgt.shape[0], dtype=np.int64)
    for i in range(coords_tgt.shape[0]):
        d2 = np.sum((coords_src - coords_tgt[i, :]) ** 2, axis=1)
        idx[i] = int(np.argmin(d2))
    return idx


def interpolate_fine_to_coarse_q(table_fine, Vq_fine, Vq_coarse):
    Qf = Vq_fine.tabulate_dof_coordinates()[:, :Vq_fine.mesh.geometry.dim]
    Qc = Vq_coarse.tabulate_dof_coordinates()[:, :Vq_coarse.mesh.geometry.dim]
    take = _nearest_map_indices(Qf, Qc)
    return table_fine[:, take].copy()


def align_rows_to_reference_x(table_src, Vx_src, Vx_ref):
    Xs = Vx_src.tabulate_dof_coordinates()[:, :Vx_src.mesh.geometry.dim]
    Xr = Vx_ref.tabulate_dof_coordinates()[:, :Vx_ref.mesh.geometry.dim]
    take = _nearest_map_indices(Xs, Xr)  # for each ref node, pick nearest in src
    return table_src[take, :].copy()


def l2q_weighted_row(rowA, rowB, omega_q, Mq_vals):
    d = rowA - rowB
    return float(np.sqrt(np.sum(omega_q * Mq_vals * d ** 2)))


def Exq_error(tableA_on_xref, tableB_on_xref_qA, omega_x_ref, omega_q, Mq_vals):
    Nx = tableA_on_xref.shape[0]
    re = np.empty(Nx, float)
    for i in range(Nx):
        re[i] = l2q_weighted_row(tableA_on_xref[i, :], tableB_on_xref_qA[i, :], omega_q, Mq_vals)
    return float(np.sqrt(np.sum(omega_x_ref * re ** 2)))


def plot_convergence_q(errs, out_png, out_pdf):
    hq = np.array([e["hq"] for e in errs], float)
    E = np.array([e["E"] for e in errs], float)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.loglog(hq, E, marker="s", linewidth=1.5, label=r"$E_q$ (x\ fixed)")
    if len(hq) >= 2:
        hA, EA = hq[-1], E[-1]
        ax.loglog(hq, EA * (hq / hA) ** 2, "--", linewidth=1.0, color="black", label=r"$\mathcal{O}(h_q^2)$")
    ax.set_xlabel(r"$h_q$")
    ax.set_ylabel(r"$E_q$ (M-weighted, lumped $L^2$ in $q$; summed over $x$)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


def run():
    os.makedirs(OUTROOT, exist_ok=True)

    runs = []
    for hq in H_Q:
        outdir = os.path.join(OUTROOT, f"q_{hq:.3f}")
        os.makedirs(outdir, exist_ok=True)

        # quick size check
        Vx_tmp, Vq_tmp, _, _, _ = build_spaces_and_weights(HX_FIXED, hq)
        Nx = Vx_tmp.dofmap.index_map.size_local
        Nq = Vq_tmp.dofmap.index_map.size_local
        print(f"[q-accuracy] hx={HX_FIXED:.3f}, hq={hq:.3f}  (Nx={Nx}, Nq={Nq}) → {outdir}")

        npy_path = os.path.join(outdir, "psihat_final.npy")
        if os.path.exists(npy_path):
            table = np.load(npy_path)
            print(f"[q-accuracy] loaded existing {npy_path} (skipped solve)")
        else:
            t0 = time.time()
            table = main_solver(
                lc_x=HX_FIXED,
                lc_q=hq,
                max_fp_iter=6,
                fp_tol=5e-3,
                fp_verbose=False,
                relax_theta=0.3,
                psi_cap=1e6,
                outdir=outdir,
            )
            np.save(npy_path, table)
            print(f"[q-accuracy] done hq={hq:.3f} in {time.time() - t0:.1f}s")

        runs.append((hq, table))

    # Build spaces/weights once per level (used for error assembly)
    spaces = {
        hq: dict(
            zip(
                ("Vx", "Vq", "omega_x", "omega_q", "Mq_vals"),
                build_spaces_and_weights(HX_FIXED, hq),
            )
        )
        for hq, _ in runs
    }

    # Reference x-grid and weights from the finest-q level
    hq_f, tbl_f = runs[-1]
    Vx_ref = spaces[hq_f]["Vx"]
    Vq_f = spaces[hq_f]["Vq"]
    omega_x_ref = spaces[hq_f]["omega_x"]

    errs = []
    for hq, tbl_c in runs[:-1]:
        # spaces/weights for this coarse-q level
        Vx_c = spaces[hq]["Vx"]
        Vq_c = spaces[hq]["Vq"]
        omega_q_c = spaces[hq]["omega_q"]
        Mq_c = spaces[hq]["Mq_vals"]

        # Map the fine table to the coarse-q space (columns), keeping the fine run's x-grid
        tbl_f_on_cq = interpolate_fine_to_coarse_q(tbl_f, Vq_f, Vq_c)

        # Align the coarse run's x-rows to the reference x-grid (rows)
        tbl_c_on_xref = align_rows_to_reference_x(tbl_c, Vx_c, Vx_ref)

        # Also align the fine-on-coarse-q table's x-rows to the same reference x-grid
        # (note the fine run's Vx could differ from Vx_ref in another scenario)
        tbl_f_on_xref_qc = align_rows_to_reference_x(tbl_f_on_cq, Vx_ref, Vx_ref)

        # Mixed error using reference omega_x
        E = Exq_error(tbl_c_on_xref, tbl_f_on_xref_qc, omega_x_ref, omega_q_c, Mq_c)
        errs.append({"hq": hq, "E": E})

    # Sort coarse->fine and compute observed rates
    errs.sort(key=lambda r: r["hq"], reverse=True)
    for i in range(len(errs) - 1):
        h1, e1 = errs[i]["hq"], errs[i]["E"]
        h2, e2 = errs[i + 1]["hq"], errs[i + 1]["E"]
        errs[i]["p"] = float(np.log(e1 / e2) / np.log(h1 / h2))

    out_json = os.path.join(OUTROOT, "convergence_q.json")
    with open(out_json, "w") as f:
        json.dump(errs, f, indent=2)

    print("\n[Convergence (q-refinement)]")
    for r in errs:
        print(f"h_q={r['hq']:.3f}  E={r['E']:.3e}" + (f"  p≈{r['p']:.2f}" if 'p' in r else ""))
    print("Saved:", out_json)

    png = os.path.join(OUTROOT, "convergence_q.png")
    pdf = os.path.join(OUTROOT, "convergence_q.pdf")
    plot_convergence_q(errs, png, pdf)
    print("Saved plots:", png, "and", pdf)


if __name__ == "__main__":
    run()