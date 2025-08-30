# variational_formulations/variational_x.py
from __future__ import annotations
import ufl
from dolfinx import fem
from typing import Dict, Any

# Define x-space numerical scheme (4.1)
def x_space_form(
    Vx: fem.FunctionSpace,
    M_x: ufl.core.expr.Expr,         # scalar on Vx.mesh
    dt: float,
    epsilon: float,
    *,
    u_vec: ufl.core.expr.Expr,       # velocity on Vx.mesh (first gdim components used)
    Lambda_x: ufl.core.expr.Expr,  
) -> tuple[ufl.Argument, ufl.Argument, ufl.Form, ufl.Form, Dict[str, Any]]:

    psihat = ufl.TrialFunction(Vx)
    phi    = ufl.TestFunction(Vx)

    dx_v = ufl.Measure("dx", domain=Vx.mesh, metadata={"quadrature_rule": "vertex"})

    # Nodal coefficients
    pi_mass_lhs = fem.Function(Vx)
    pi_diff_lhs = fem.Function(Vx)
    pi_mass_rhs = fem.Function(Vx)

    # LHS: mass + diffusion
    a_form = (
        (M_x / dt) * pi_mass_lhs * psihat * phi
        + (M_x * epsilon) * pi_diff_lhs * ufl.inner(ufl.grad(psihat), ufl.grad(phi))
    ) * dx_v

    # RHS: mass term
    L_form = ((M_x / dt) * pi_mass_rhs * phi) * dx_v

    gdim = Vx.mesh.geometry.dim
    grad_phi = ufl.grad(phi)

    u0 = u_vec[0] if gdim >= 1 else 0.0
    u1 = u_vec[1] if gdim >= 2 else 0.0
    L00, L01 = Lambda_x[0, 0], Lambda_x[0, 1]
    L10, L11 = Lambda_x[1, 0], Lambda_x[1, 1]
    ueff0 = L00 * u0 + L10 * u1
    ueff1 = L01 * u0 + L11 * u1

    scalar = ueff0 * grad_phi[0] + (ueff1 * grad_phi[1] if gdim >= 2 else 0.0)
    if gdim >= 3:
        scalar = scalar + 0.0 * grad_phi[2]  # keeps shapes aligned

    L_form += (M_x * scalar) * dx_v

    return psihat, phi, a_form, L_form, {
        "pi_mass_lhs": pi_mass_lhs,
        "pi_diff_lhs": pi_diff_lhs,
        "pi_mass_rhs": pi_mass_rhs,
        "dx_vertex": dx_v,
    }
