# utils/interpolant.py
from __future__ import annotations
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc

# Define interpolants for x,q, and full

def as_Vx_function(column_vals: np.ndarray, Vx: fem.FunctionSpace) -> fem.Function:
    f = fem.Function(Vx)
    f.x.array[:] = np.asarray(column_vals, dtype=f.x.array.dtype)
    return f

def as_Vq_function(row_vals: np.ndarray, Vq: fem.FunctionSpace) -> fem.Function:
    f = fem.Function(Vq)
    f.x.array[:] = np.asarray(row_vals, dtype=f.x.array.dtype)
    return f

# Full interpolant 

def pi_h_apply_nodewise(G, table: np.ndarray) -> np.ndarray:
    return G(table)

def lift_x_slice(table: np.ndarray, j: int, Vx: fem.FunctionSpace) -> fem.Function:
    return as_Vx_function(table[:, j], Vx)

def lift_q_slice(table: np.ndarray, i: int, Vq: fem.FunctionSpace) -> fem.Function:
    return as_Vq_function(table[i, :], Vq)


# Nodal table for slicing 

def make_tensor_table_from_separable(fx: fem.Function, fq: fem.Function) -> np.ndarray:
    vx = fx.x.array.copy()
    vq = fq.x.array.copy()
    return np.multiply.outer(vx, vq)  # expect shape (Nx, Nq)

# Mass lumping 

def lump_weights(V: fem.FunctionSpace) -> np.ndarray:
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    A = petsc.assemble_matrix(fem.form(a))
    A.assemble()
    w = A.createVecLeft()
    A.getRowSum(w)
    return w.array.copy()

def lumped_inner_product_xy(a_tbl: np.ndarray,
                            b_tbl: np.ndarray,
                            omega_x: np.ndarray,
                            omega_q: np.ndarray,
                            M_tbl: np.ndarray | None = None) -> float:

    W = np.multiply.outer(omega_x, omega_q)  # shape (Nx, Nq)
    core = a_tbl * b_tbl
    if M_tbl is not None:
        core = core * M_tbl
    return float(np.sum(W * core))
