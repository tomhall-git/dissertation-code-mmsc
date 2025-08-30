# solvers/solver_q.py
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem import petsc
import numpy as np
import ufl
from ..variational_formulations.variational_q import q_space_form

# q solver for full algorithm

def q_step(psihat_table_old, Vq, M_q, dt_sub, Wi,
           tilde_Lambda_q, omega_q, grad_u_at_xslice,
           delta_floor: float = 1e-8, max_iter: int = 10, tol: float = 1e-6, verbose: bool = False,
           relax_theta: float = 0.3, psi_cap: float = 1e6):

    Nx = psihat_table_old.shape[0]
    psihat_table_new = np.zeros_like(psihat_table_old)

    # 2×2 for S(u) (expected by q_space_form)
    S_const = fem.Constant(Vq.mesh, np.zeros((2, 2), dtype=float))

    # Setup functions for iteration
    psi_current = fem.Function(Vq)
    psi_for_L = fem.Function(Vq)
    psi_floor = max(delta_floor, 1e-12)
    psi_current.x.array[:] = psihat_table_old[0, :]
    psi_for_L.x.array[:] = np.clip(psi_current.x.array, psi_floor, psi_cap)
    Lambda_holder = tilde_Lambda_q(psi_for_L)  # DG0 tensor; updated in loop

    # Build forms once
    psihat_trial, phi, a_form, L_form, coeffs = q_space_form(
        Vq=Vq, M_q=M_q, dt=dt_sub, Wi=Wi,
        Lambda_q=Lambda_holder,
        S_u_const=S_const,
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
    pc = ksp.getPC()
    pc.setType("gamg")
    ksp.setTolerances(rtol=1e-8)
    ksp.setInitialGuessNonzero(True)  # use previous iterate as initial guess
    ksp.setFromOptions()

    # RHS vector and solution
    form_L = fem.form(L_form)
    b = petsc.create_vector(form_L)
    psihat_new_q = fem.Function(Vq)

    # Fixed-point iteration for each x-slice
    for i in range(Nx):
        # RHS mass term
        psihat_rhs.x.array[:] = psihat_table_old[i, :]

        # Update corotational S(u)(x_i): ensure 2×2
        Gu_np = np.asarray(grad_u_at_xslice(i))
        if Gu_np.shape == (2, 2):
            G2 = Gu_np
        elif Gu_np.shape == (3, 3):
            G2 = Gu_np[:2, :2]
        else:
            G2 = np.array(Gu_np, dtype=float).reshape(-1)[:4].reshape(2, 2)
        S2 = 0.5 * (G2 - G2.T)
        S_const.value[:] = S2

        # Init iterate with previous timestep
        psihat_new_q.x.array[:] = psihat_table_old[i, :]

        converged = False
        for fp_iter in range(max_iter):
            psi_prev_iter = psihat_new_q.x.array.copy()

            # Update lambda using current iterate (small negative clipping)
            psi_for_L.x.array[:] = np.clip(psihat_new_q.x.array, psi_floor, psi_cap)
            Lambda_new = tilde_Lambda_q(psi_for_L)
            Lambda_holder.x.array[:] = Lambda_new.x.array
            Lambda_holder.x.scatter_forward()

            # Reassemble RHS with updated lambda
            b.zeroEntries()
            fem.petsc.assemble_vector(b, form_L)
            b.assemble()

            # Solve A x = b (A is constant SPD)
            ksp.solve(b, psihat_new_q.x.petsc_vec)
            reason = ksp.getConvergedReason()
            if reason < 0:
                raise RuntimeError(f"KSP diverged in q-step (slice i={i}, fp_iter={fp_iter}), reason={reason}")

            psihat_new_q.x.scatter_forward()

            # FP relaxation (damping) and clipping
            lin_candidate = psihat_new_q.x.array.copy()
            damped = relax_theta * lin_candidate + (1.0 - relax_theta) * psi_prev_iter
            np.minimum(damped, psi_cap, out=damped)
            psihat_new_q.x.array[:] = damped

            # Safety
            if not np.isfinite(psihat_new_q.x.array).all():
                raise RuntimeError(f"Non-finite values in q-step solution (slice i={i}, fp_iter={fp_iter}).")

            # Convergence check
            diff = np.linalg.norm(psihat_new_q.x.array - psi_prev_iter)
            rel_diff = diff / (np.linalg.norm(psihat_new_q.x.array) + 1e-12)
            if rel_diff < tol:
                converged = True
                break

        # Store solution (damped + clipped)
        psihat_table_new[i, :] = psihat_new_q.x.array

    return psihat_table_new