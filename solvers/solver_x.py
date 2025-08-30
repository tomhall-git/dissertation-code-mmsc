# solvers/solver_x.py
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem import petsc
import numpy as np
import ufl
from ..variational_formulations.variational_x import x_space_form

# x solver step for full algorithm 

def x_step(psihat_table_old, Vx, M_x, dt_sub, eps,
           tilde_Lambda_x, omega_x, u=None, delta_floor: float = 1e-8,
           max_iter: int = 10, tol: float = 1e-6, verbose: bool = False,
           relax_theta: float = 0.3, psi_cap: float = 1e6):
    Nq = psihat_table_old.shape[1]
    psihat_table_new = np.zeros_like(psihat_table_old)

    # Ensure velocity has the right dimension (2D geom)
    gdim = Vx.mesh.geometry.dim
    if u is None:
        u_vec = ufl.as_vector((0.0,) * gdim)
    else:
        u_vec = ufl.as_vector(tuple(u[i] for i in range(gdim)))

    # Setup functions for iteration
    psi_current = fem.Function(Vx)
    psi_for_L = fem.Function(Vx)
    psi_current.x.array[:] = psihat_table_old[:, 0]
    psi_floor = max(delta_floor, 1e-12)
    psi_for_L.x.array[:] = np.clip(psi_current.x.array, psi_floor, psi_cap)
    Lambda_holder = tilde_Lambda_x(psi_for_L)  # DG0 tensor; updated in loop

    # Build forms once
    psihat_trial, phi, a_form, L_form, coeffs = x_space_form(
        Vx=Vx, M_x=M_x, dt=dt_sub, epsilon=eps,
        u_vec=u_vec,
        Lambda_x=Lambda_holder,
    )

    ndofs_local = Vx.dofmap.index_map.size_local * Vx.dofmap.index_map_bs
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
    psihat_new_x = fem.Function(Vx)

    # Fixed-point iteration for each q-slice
    for j in range(Nq):
        # Set RHS mass term and init iterate with previous timestep
        psihat_rhs.x.array[:] = psihat_table_old[:, j]
        psihat_new_x.x.array[:] = psihat_table_old[:, j]

        converged = False
        for fp_iter in range(max_iter):
            # Previous iterate (for relaxation + convergence test)
            psi_prev_iter = psihat_new_x.x.array.copy()

            # Update lambda using current iterate (small negative clipping)
            psi_for_L.x.array[:] = np.clip(psihat_new_x.x.array, psi_floor, psi_cap)
            Lambda_new = tilde_Lambda_x(psi_for_L)
            Lambda_holder.x.array[:] = Lambda_new.x.array
            Lambda_holder.x.scatter_forward()

            # Reassemble RHS with updated lambda
            b.zeroEntries()
            fem.petsc.assemble_vector(b, form_L)
            b.assemble()

            # Solve A x = b (A is constant SPD)
            ksp.solve(b, psihat_new_x.x.petsc_vec)
            reason = ksp.getConvergedReason()
            if reason < 0:
                raise RuntimeError(f"KSP diverged in x-step (slice j={j}, fp_iter={fp_iter}), reason={reason}")

            psihat_new_x.x.scatter_forward()

            # FP relaxation (damping) and clipping
            lin_candidate = psihat_new_x.x.array.copy()
            damped = relax_theta * lin_candidate + (1.0 - relax_theta) * psi_prev_iter
            np.minimum(damped, psi_cap, out=damped)
            psihat_new_x.x.array[:] = damped

            # Safety
            if not np.isfinite(psihat_new_x.x.array).all():
                raise RuntimeError(f"Non-finite values in x-step solution (slice j={j}, fp_iter={fp_iter}).")

            # Convergence check
            diff = np.linalg.norm(psihat_new_x.x.array - psi_prev_iter)
            rel_diff = diff / (np.linalg.norm(psihat_new_x.x.array) + 1e-12)
            if rel_diff < tol:
                converged = True
                break

        # Store solution (even if FP didn't fully converge; it's damped + clipped)
        psihat_table_new[:, j] = psihat_new_x.x.array

    return psihat_table_new