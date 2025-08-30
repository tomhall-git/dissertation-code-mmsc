# experiments/5_2_mesh/post_processing_q.py
import os, json, numpy as np
from dolfinx import fem, geometry
import ufl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...domain.mesh_x import get_domain_x
from ...domain.mesh_q import get_domain_q
from ...utils.parameters import b

# same for q, again not used

OUTROOT   = os.path.join(os.path.dirname(__file__), "results")
HX_FIXED  = 0.35
H_Q       = [0.95, 0.85, 0.75]   # coarse -> fine

USE_Q_MASK = True
ALPHA_MASK = 0.97  

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
    Mq_fun = fem.Function(Vq)
    def Mq_callable(x):
        r2 = x[0]**2 + x[1]**2
        return np.maximum(1.0 - r2 / b, 0.0)**(b/2)
    Mq_fun.interpolate(Mq_callable)
    return Vx, Vq, omega_x, omega_q, Mq_fun.x.array.copy()

def _cells_for_points(mesh, P):

    npts, gdim = P.shape
    P3 = np.zeros((npts, 3), dtype=mesh.geometry.x.dtype)
    P3[:, :gdim] = P
    tdim = mesh.topology.dim
    bb = geometry.bb_tree(mesh, tdim)
    candidates = geometry.compute_collisions_points(bb, P3)
    colliding = geometry.compute_colliding_cells(mesh, candidates, P3)
    cells = np.full(npts, -1, dtype=np.int32)
    for i in range(npts):
        s = colliding.offsets[i]; e = colliding.offsets[i+1]
        if e > s:
            cells[i] = colliding.array[s]
        else:
            cs = candidates.offsets[i]; ce = candidates.offsets[i+1]
            if ce > cs:
                cells[i] = candidates.array[cs]
            else:
                cells[i] = 0  # extreme fallback (should be rare)
    return cells

def eval_function_on_points(u: fem.Function, P: np.ndarray):
    mesh = u.function_space.mesh
    npts, gdim = P.shape

    cells = _cells_for_points(mesh, P)
    cells = np.asarray(cells, dtype=np.int32)

    P3 = np.zeros((npts, 3), dtype=np.float64)
    P3[:, :gdim] = P

    vals = np.empty(npts, dtype=np.float64)
    for k in range(npts):
        xk = P3[k, :]                       # shape (3,)
        ck = np.array([cells[k]], np.int32) # shape (1,)
        v = u.eval(xk, ck)                 
        # v can be a tuple/list of arrays; pull the scalar out safely
        if isinstance(v, (list, tuple)):
            vals[k] = float(np.asarray(v[0]).ravel()[0])
        else:
            vals[k] = float(np.asarray(v).ravel()[0])
    return vals

def map_row_q_fine_to_coarse(row_vals_f, Vq_f, Vq_c):
    uf = fem.Function(Vq_f); uf.x.array[:] = row_vals_f
    Qc = Vq_c.tabulate_dof_coordinates()[:, :Vq_c.mesh.geometry.dim]
    return eval_function_on_points(uf, Qc)

def map_col_x_fine_to_coarse(col_vals_f, Vx_f, Vx_c):
    uf = fem.Function(Vx_f); uf.x.array[:] = col_vals_f
    Xc = Vx_c.tabulate_dof_coordinates()[:, :Vx_c.mesh.geometry.dim]
    return eval_function_on_points(uf, Xc)

# norms
def l2q_weighted_row(rowA, rowB, omega_q, Mq_vals, mask=None):
    d = rowA - rowB
    w = omega_q * Mq_vals if mask is None else omega_q * Mq_vals * mask
    return float(np.sqrt(np.sum(w * d**2)))

def Exq_error(tableA, tableB, omega_x, omega_q, Mq_vals, mask=None):
    Nx = tableA.shape[0]
    re = np.empty(Nx, float)
    for i in range(Nx):
        re[i] = l2q_weighted_row(tableA[i, :], tableB[i, :], omega_q, Mq_vals, mask)
    return float(np.sqrt(np.sum(omega_x * re**2)))

def plot_pairwise(hq_coarse, E_pair, out_png, out_pdf, order_ref=1):
    h = np.array(hq_coarse, float); E = np.array(E_pair, float)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.loglog(h, E, marker="s", linewidth=1.8, label=r"$E_q$ (pairwise)")
    if len(h) >= 2:
        hA, EA = h[-1], E[-1]
        ax.loglog(h, EA * (h/hA)**order_ref, "--", linewidth=1.2, color="black",
                  label=rf"$\mathcal{{O}}(h_q^{order_ref})$")
    ax.set_xlabel(r"$h_q$")
    ax.set_ylabel(r"$E_q$ (M-weighted, lumped $L^2$ in $q$; summed over $x$)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220); fig.savefig(out_pdf); plt.close(fig)

def run():
    # Load saved tables (from runq.py)
    runs = {}
    for hq in H_Q:
        folder = os.path.join(OUTROOT, f"q_{hq:.3f}")
        path = os.path.join(folder, "psihat_final.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing result for hq={hq:.3f}: {path}")
        runs[hq] = np.load(path)

    spaces = {
        hq: dict(zip(("Vx","Vq","omega_x","omega_q","Mq_vals"),
                     build_spaces_and_weights(HX_FIXED, hq)))
        for hq in H_Q
    }

    q_masks = {}
    if USE_Q_MASK:
        for hq in H_Q:
            Vq = spaces[hq]["Vq"]
            Qc = Vq.tabulate_dof_coordinates()[:, :Vq.mesh.geometry.dim]
            rq = np.sqrt(np.sum(Qc**2, axis=1))
            q_masks[hq] = (rq <= ALPHA_MASK*np.sqrt(b)).astype(float)
    else:
        q_masks = {hq: None for hq in H_Q}

    hq_coarse, E_pair, details = [], [], []
    for hc, hf in zip(H_Q[:-1], H_Q[1:]): 
        tbl_c = runs[hc]
        tbl_f = runs[hf]

        Vx_c, Vq_c = spaces[hc]["Vx"], spaces[hc]["Vq"]
        Vx_f, Vq_f = spaces[hf]["Vx"], spaces[hf]["Vq"]
        Wxc, Wqc, Mqc = spaces[hc]["omega_x"], spaces[hc]["omega_q"], spaces[hc]["Mq_vals"]
        maskc = q_masks[hc]

        Nx_f = tbl_f.shape[0]
        Nqc  = Vq_c.dofmap.index_map.size_local
        tbl_f_on_qc = np.empty((Nx_f, Nqc), dtype=tbl_f.dtype)
        for i in range(Nx_f):
            tbl_f_on_qc[i, :] = map_row_q_fine_to_coarse(tbl_f[i, :], Vq_f, Vq_c)

        Nx_c = Vx_c.dofmap.index_map.size_local
        tbl_f_on_xc_qc = np.empty((Nx_c, Nqc), dtype=tbl_f.dtype)
        for j in range(Nqc):
            col_f = tbl_f_on_qc[:, j]  
            tbl_f_on_xc_qc[:, j] = map_col_x_fine_to_coarse(col_f, Vx_f, Vx_c)

        tbl_c_on_xc_qc = tbl_c

        E = Exq_error(tbl_c_on_xc_qc, tbl_f_on_xc_qc, Wxc, Wqc, Mqc, mask=maskc)
        hq_coarse.append(hc); E_pair.append(E)
        details.append({"hq_coarse": hc, "hq_fine": hf, "E_pair": E})

    out_json = os.path.join(OUTROOT, "convergence_q_pairwise_robust.json")
    with open(out_json, "w") as f: json.dump(details, f, indent=2)
    print("Saved:", out_json)
    for d in details:
        print(f"h_q={d['hq_coarse']:.3f} vs {d['hq_fine']:.3f}  →  E_pair={d['E_pair']:.3e}")

    if len(E_pair) >= 2:
        p = np.log(E_pair[0]/E_pair[1]) / np.log(hq_coarse[0]/hq_coarse[1])
        print(f"Observed pairwise rate p≈{p:.2f}")

    plot_pairwise(
        hq_coarse, E_pair,
        os.path.join(OUTROOT, "convergence_q_pairwise_robust.png"),
        os.path.join(OUTROOT, "convergence_q_pairwise_robust.pdf"),
        order_ref=1
    )
    print("Saved plots: convergence_q_pairwise_robust.[png|pdf] in", OUTROOT)

if __name__ == "__main__":
    run()