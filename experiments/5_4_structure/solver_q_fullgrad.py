# experiments/5_4_structure/solver_q_fullgrad.py
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem import petsc
import numpy as np
from ...variational_formulations.variational_q import q_space_form

# Now including full grad u, not just skew-symmetric part
def q_step_fullgrad(psihat_table_old,
                    Vq,
                    M_q,
                    dt_sub,
                    Wi,
                    tilde_Lambda_q_raw,     
                    omega_q,
                    grad_u_at_xslice,           # returns full grad u
                    max_iter: int = 10,
                    tol: float = 1e-6,
                    verbose: bool = False,
                    relax_theta: float = 0.3,
                    psi_cap: float = 1e6):
    Nx = psihat_table_old.shape[0]
    psihat_table_new = np.zeros_like(psihat_table_old)

    Grad_const = fem.Constant(Vq.mesh, np.zeros((2, 2), dtype=float))
    psi_for_L = fem.Function(Vq)

    psi_for_L.x.array[:] = psihat_table_old[0, :]
    Lambda_holder = tilde_Lambda_q_raw(psi_for_L)

    _, _, a_form, L_form, coeffs = q_space_form(
        Vq=Vq, M_q=M_q, dt=dt_sub, Wi=Wi,
        Lambda_q=Lambda_holder,
        S_u_const=Grad_const,  # we need full gradient
    )

    ndofs_local = Vq.dofmap.index_map.size_local * Vq.dofmap.index_map_bs
    coeffs["pi_mass_lhs"].x.array[:ndofs_local] = 1.0
    coeffs["pi_diff_lhs"].x.array[:ndofs_local] = 1.0
    psihat_rhs = coeffs["pi_mass_rhs"]

    # Assemble A once; reuse KSP
    form_A = fem.form(a_form)
    A = petsc.assemble_matrix(form_A)
    A.assemble()
    ksp = PETSc.KSP().create(A.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("gamg")
    ksp.setTolerances(rtol=1e-8)
    ksp.setInitialGuessNonzero(True)
    ksp.setFromOptions()

    # RHS vector and solution
    form_L = fem.form(L_form)
    b = petsc.create_vector(form_L)
    psihat_new_q = fem.Function(Vq)

    for i in range(Nx):
        # RHS mass term
        psihat_rhs.x.array[:] = psihat_table_old[i, :]
        Gu_np = np.asarray(grad_u_at_xslice(i))
        if Gu_np.shape == (2, 2):
            G2 = Gu_np
        elif Gu_np.shape == (3, 3):
            G2 = Gu_np[:2, :2]
        else:
            G2 = np.array(Gu_np, dtype=float).reshape(-1)[:4].reshape(2, 2)
        Grad_const.value[:] = G2

        psihat_new_q.x.array[:] = psihat_table_old[i, :]

        converged = False
        for fp_iter in range(max_iter):
            psi_prev_iter = psihat_new_q.x.array.copy()

            psi_for_L.x.array[:] = psihat_new_q.x.array
            Lambda_new = tilde_Lambda_q_raw(psi_for_L)
            Lambda_holder.x.array[:] = Lambda_new.x.array
            Lambda_holder.x.scatter_forward()

            b.zeroEntries()
            fem.petsc.assemble_vector(b, form_L)
            b.assemble()

            # Solve A x = b
            ksp.solve(b, psihat_new_q.x.petsc_vec)
            if ksp.getConvergedReason() < 0:
                raise RuntimeError(f"KSP diverged in q-step (i={i}, fp_iter={fp_iter})")

            psihat_new_q.x.scatter_forward()

            lin_candidate = psihat_new_q.x.array.copy()
            damped = relax_theta * lin_candidate + (1.0 - relax_theta) * psi_prev_iter
            if np.isfinite(psi_cap):
                np.minimum(damped, psi_cap, out=damped)
            psihat_new_q.x.array[:] = damped

            if not np.isfinite(psihat_new_q.x.array).all():
                raise RuntimeError(f"Non-finite values in q-step solution (i={i}, fp_iter={fp_iter}).")

            diff = np.linalg.norm(psihat_new_q.x.array - psi_prev_iter)
            rel_diff = diff / (np.linalg.norm(psihat_new_q.x.array) + 1e-12)
            if rel_diff < tol:
                converged = True
                break

        if verbose:
            status = "✓" if converged else "!"
            print(f"[q_fullgrad] i={i:04d} {status} rel_tol={tol:g}")

        psihat_table_new[i, :] = psihat_new_q.x.array

    return psihat_table_new