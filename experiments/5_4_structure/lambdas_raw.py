# experiments/5_4_structure/lambdas_raw.py
from __future__ import annotations
import numpy as np
import basix.ufl
from dolfinx import fem

from ...utils.function import F_delta_num, Fp_delta_num, Fpp_delta_num

# Construction of lambda needs to be changed slightly to handle deliberate instability (so solver does not diverge immediately, and meaningful numerical behaviour can be observed)
def _make_DG0_tensor_space(V: fem.FunctionSpace):
    msh = V.mesh
    gdim = msh.geometry.dim
    cellname = msh.ufl_cell().cellname()
    element = basix.ufl.element("DG", cellname, 0, shape=(gdim, gdim))
    return fem.functionspace(mesh=msh, element=element)

def _lambda_edge_raw(a: float, b: float, delta: float) -> float:
    a = float(a); b = float(b)
    Fa, Fb = F_delta_num(a, delta), F_delta_num(b, delta)
    Fpa, Fpb = Fp_delta_num(a, delta), Fp_delta_num(b, delta)
    denom = Fpa - Fpb
    eps = 1e-14
    if abs(denom) > eps:
        return (Fa - Fb) / denom
    return Fpa / Fpp_delta_num(a, delta)

def _build_lambda_DG0_from_slice_raw(
    phi: fem.Function,
    delta: float,
    *,
    lam_min: float | None = None,
    lam_max: float | None = None,
) -> fem.Function:

    V = phi.function_space
    msh = V.mesh
    gdim = msh.geometry.dim
    tdim = msh.topology.dim

    W = _make_DG0_tensor_space(V)
    Lambda = fem.Function(W)

    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    coords = msh.geometry.x
    dofmap = V.dofmap

    num_local_cells = msh.topology.index_map(tdim).size_local
    values = np.zeros((num_local_cells, gdim, gdim), dtype=Lambda.x.array.dtype)

    for c in range(num_local_cells):
        verts = c2v.links(c)          # (d+1,) vertex indices
        X = coords[verts]             # (d+1)×gdim
        X0 = X[0]
        A = (X[1:] - X0).T            # gdim×tdim

        ATA = A.T @ A
        ATA_inv = np.linalg.pinv(ATA)
        B = A @ ATA_inv              

        cell_dofs = dofmap.cell_dofs(c)
        local_vals = phi.x.array[cell_dofs].astype(float)
        a0 = float(local_vals[0])

        D = np.zeros((tdim, tdim), dtype=float)
        for i in range(1, tdim + 1):
            ai = float(local_vals[i])
            lam_i = _lambda_edge_raw(ai, a0, delta)

            # New - clipping on large entries
            if lam_min is not None and lam_i < lam_min:
                lam_i = lam_min
            if lam_max is not None and lam_i > lam_max:
                lam_i = lam_max

            D[i - 1, i - 1] = lam_i

        L = B @ D @ A.T               # g×g
        values[c, :, :] = L

    Lambda.x.array[:] = values.reshape(-1)
    Lambda.x.scatter_forward()
    return Lambda

def tilde_Lambda_x_delta_raw(
    phi_x: fem.Function,
    delta: float,
    *,
    lam_min: float | None = None,
    lam_max: float | None = None,
) -> fem.Function:
    return _build_lambda_DG0_from_slice_raw(phi_x, delta, lam_min=lam_min, lam_max=lam_max)

def tilde_Lambda_q_delta_raw(
    phi_q: fem.Function,
    delta: float,
    *,
    lam_min: float | None = None,
    lam_max: float | None = None,
) -> fem.Function:
    return _build_lambda_DG0_from_slice_raw(phi_q, delta, lam_min=lam_min, lam_max=lam_max)