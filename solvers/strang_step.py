# solvers/strang_step.py
from .solver_q import q_step
from .solver_x import x_step
import time
import numpy as np

# Simple strang framework for full algorithm

def strang_step(psihat_table_old, Vx, Vq, M_x, M_q,
                dt, epsilon, Wi,
                tilde_Lambda_x, tilde_Lambda_q,
                omega_x, omega_q,
                u,
                grad_u_at_xslice,
                delta,  # used as delta_floor
                verbose=False,
                # Fixed-point controls
                max_fp_iter=10, fp_tol=1e-6, fp_verbose=False,
                # Stabilisation controls
                relax_theta: float = 0.3, psi_cap: float = 1e6):
    t0 = time.perf_counter()

    # Half q-step (implicit)
    psihat_half_q = q_step(
        psihat_table_old, Vq, M_q, 0.5 * dt, Wi,
        tilde_Lambda_q, omega_q,
        grad_u_at_xslice=grad_u_at_xslice,
        delta_floor=delta,
        max_iter=max_fp_iter, tol=fp_tol, verbose=fp_verbose,
        relax_theta=relax_theta, psi_cap=psi_cap
    )
    if verbose:
        print(f"[Strang] q(Δt/2) implicit done in {time.perf_counter()-t0:.3f}s", flush=True)

    # Full x-step (implicit)
    t1 = time.perf_counter()
    psihat_full_x = x_step(
        psihat_half_q, Vx, M_x, dt, epsilon,
        tilde_Lambda_x, omega_x,
        u=u,
        delta_floor=delta,
        max_iter=max_fp_iter, tol=fp_tol, verbose=fp_verbose,
        relax_theta=relax_theta, psi_cap=psi_cap
    )
    if verbose:
        print(f"[Strang] x(Δt) implicit done in {time.perf_counter()-t1:.3f}s", flush=True)

    # Half q-step (implicit)
    t2 = time.perf_counter()
    psihat_new = q_step(
        psihat_full_x, Vq, M_q, 0.5 * dt, Wi,
        tilde_Lambda_q, omega_q,
        grad_u_at_xslice=grad_u_at_xslice,
        delta_floor=delta,
        max_iter=max_fp_iter, tol=fp_tol, verbose=fp_verbose,
        relax_theta=relax_theta, psi_cap=psi_cap
    )
    if verbose:
        print(f"[Strang] q(Δt/2) implicit done in {time.perf_counter()-t2:.3f}s", flush=True)
        print(f"[Strang] one full implicit step took {time.perf_counter()-t0:.3f}s", flush=True)

    # Final safety check (paranoid)
    if not np.isfinite(psihat_new).all():
        raise RuntimeError("Non-finite values after Strang step. Consider reducing dt or tightening caps.")

    return psihat_new