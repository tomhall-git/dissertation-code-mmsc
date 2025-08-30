# variational_formulations/variational_q.py
from __future__ import annotations
import ufl
from dolfinx import fem
from typing import Dict, Any, Optional

# Define q-space numerical scheme (4.2)
def q_space_form(
    Vq: fem.FunctionSpace,
    M_q: ufl.core.expr.Expr,          # scalar on Vq.mesh
    dt: float,
    Wi: float,
    *,
    Lambda_q: Optional[ufl.core.expr.Expr] = None,   
    S_u_const: Optional[fem.Constant] = None,        # 2×2 Constant for S(u)
) -> tuple[ufl.Argument, ufl.Argument, ufl.Form, ufl.Form, Dict[str, Any]]:
    
    psihat = ufl.TrialFunction(Vq)
    phi    = ufl.TestFunction(Vq)

    dx_v = ufl.Measure("dx", domain=Vq.mesh, metadata={"quadrature_rule": "vertex"})

    # Nodal coefficients
    pi_mass_lhs = fem.Function(Vq)
    pi_diff_lhs = fem.Function(Vq)
    pi_mass_rhs = fem.Function(Vq)

    # LHS: mass + q-diffusion
    a_form = (
        (M_q / dt) * pi_mass_lhs * psihat * phi
        + (M_q / (2.0 * Wi)) * pi_diff_lhs * ufl.inner(ufl.grad(psihat), ufl.grad(phi))
    ) * dx_v

    # RHS: mass term
    L_form = ((M_q / dt) * pi_mass_rhs * phi) * dx_v

    if (Lambda_q is not None) and (S_u_const is not None):
        gdim = Vq.mesh.geometry.dim
        grad_phi = ufl.grad(phi)

        # A(q) = S(u) q (2D)
        q_coord = ufl.SpatialCoordinate(Vq.mesh)
        q_vec   = ufl.as_vector((q_coord[0], q_coord[1]))  # 2D config space
        A_q     = ufl.dot(S_u_const, q_vec)                # (A0, A1)

        A0, A1 = A_q[0], A_q[1]
        L00, L01 = Lambda_q[0, 0], Lambda_q[0, 1]
        L10, L11 = Lambda_q[1, 0], Lambda_q[1, 1]
        Aeff0 = L00 * A0 + L10 * A1
        Aeff1 = L01 * A0 + L11 * A1
        
        scalar = Aeff0 * grad_phi[0] + (Aeff1 * grad_phi[1] if gdim >= 2 else 0.0)
        if gdim >= 3:
            scalar = scalar + 0.0 * grad_phi[2]

        L_form += (M_q * scalar) * dx_v

    return psihat, phi, a_form, L_form, {
        "pi_mass_lhs": pi_mass_lhs,
        "pi_diff_lhs": pi_diff_lhs,
        "pi_mass_rhs": pi_mass_rhs,
        "dx_vertex": dx_v,
    }
