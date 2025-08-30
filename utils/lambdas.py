# utils/lambdas.py
from __future__ import annotations
import numpy as np
import basix.ufl
from dolfinx import fem

from .function import (
    F_delta_num,
    Fp_delta_num,
    Fpp_delta_num,
)

# Define lambda diagonal matrices 

def _make_DG0_tensor_space(V: fem.FunctionSpace):
    msh = V.mesh
    gdim = msh.geometry.dim
    cellname = msh.ufl_cell().cellname()
    element = basix.ufl.element("DG", cellname, 0, shape=(gdim, gdim))
    return fem.functionspace(mesh=msh, element=element)

def _lambda_edge(a: float, b: float, delta: float) -> float:
    a = float(a); b = float(b)
    a = a if a >= delta else delta
    b = b if b >= delta else delta

    Fa, Fb = F_delta_num(a, delta), F_delta_num(b, delta)
    Fpa, Fpb = Fp_delta_num(a, delta), Fp_delta_num(b, delta)
    denom = Fpa - Fpb
    eps = 1e-14
    if abs(denom) > eps:
        return (Fa - Fb) / denom
    return Fpa / Fpp_delta_num(a, delta)

def _build_lambda_DG0_from_slice(phi: fem.Function, delta: float) -> fem.Function:
    V = phi.function_space
    msh = V.mesh
    gdim = msh.geometry.dim
    tdim = msh.topology.dim  # equals gdim on simplices

    W = _make_DG0_tensor_space(V)
    Lambda = fem.Function(W)

    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    coords = msh.geometry.x
    dofmap = V.dofmap

    num_local_cells = msh.topology.index_map(tdim).size_local
    values = np.zeros((num_local_cells, gdim, gdim), dtype=Lambda.x.array.dtype)

    for c in range(num_local_cells):
        verts = c2v.links(c)                 # (d+1,) vertex indices
        X = coords[verts]                    # (d+1)×gdim physical coords
        X0 = X[0]
        A = (X[1:] - X0).T                   # gdim×tdim edge matrix

        # Geometric inverse
        ATA = A.T @ A
        ATA_inv = np.linalg.pinv(ATA)
        B = A @ ATA_inv                      # columns for i=1..d

        # Local CG1 nodal values (vertex-based dofs in same order)
        cell_dofs = dofmap.cell_dofs(c)
        local_vals = phi.x.array[cell_dofs].astype(float)  # length d+1
        a0 = float(local_vals[0])

        # Diagonal D with λ_i from divided differences between (i,0) using floored nodal values
        D = np.zeros((tdim, tdim), dtype=float)
        for i in range(1, tdim + 1):
            ai = float(local_vals[i])
            lam_i = _lambda_edge(ai, a0, delta)
            D[i - 1, i - 1] = lam_i

        L = B @ D @ A.T
        values[c, :, :] = L

    Lambda.x.array[:] = values.reshape(-1)
    Lambda.x.scatter_forward()
    return Lambda

def tilde_Lambda_x_delta(phi_x: fem.Function, delta: float) -> fem.Function:
    return _build_lambda_DG0_from_slice(phi_x, delta)

def tilde_Lambda_q_delta(phi_q: fem.Function, delta: float) -> fem.Function:
    return _build_lambda_DG0_from_slice(phi_q, delta)
