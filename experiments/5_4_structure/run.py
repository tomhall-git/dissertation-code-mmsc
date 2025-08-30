# experiments/5_4_structure/run.py
import os
import csv  
import numpy as np
from mpi4py import MPI
from dolfinx import fem
import ufl

from .strang_step_exp import strang_step  

from .lambdas_raw import (
    tilde_Lambda_x_delta_raw,
    tilde_Lambda_q_delta_raw,
)

from ...utils.interpolant import lump_weights
from ...utils.parameters import epsilon, Wi, dt, num_steps, b, delta
from ...viz.results import ResultManager
from ...domain.mesh_x import get_domain_x
from ...domain.mesh_q import get_domain_q

# Experiment 4 code

def _taylor_green_u(mesh):
    x = ufl.SpatialCoordinate(mesh)
    pi = np.pi
    ux = pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    uy = -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    return ufl.as_vector((ux, uy, 0.0)) 

def run(
    lc_x: float = 0.25,
    lc_q: float = 0.75,
    max_fp_iter: int = 10,
    fp_tol: float = 1e-4,
    fp_verbose: bool = False,
    relax_theta: float = 0.3,
    psi_cap: float = 1e6,
    outdir: str | None = None,
    lambda_min: float | None = 1e-8,
    lambda_max: float | None = None,
):

    comm = MPI.COMM_WORLD
    if outdir is None:
        outdir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(outdir, exist_ok=True)

    print("Running 5_4 experiment (FULL ∇u + raw Λ̃ with λ-clipping)...", flush=True)

    mesh_x = get_domain_x(lc=lc_x)
    mesh_q = get_domain_q(lc=lc_q, radius=np.sqrt(b), b=b)
    Vx = fem.functionspace(mesh_x, ("CG", 1))
    Vq = fem.functionspace(mesh_q, ("CG", 1))

    u = _taylor_green_u(mesh_x)

    xxy = Vx.mesh.geometry.x[:, :2]
    pi = np.pi

    def grad_u_raw_at(i: int):
        x, y = xxy[i, 0], xxy[i, 1]
        cpx, spx = np.cos(pi * x), np.sin(pi * x)
        cpy, spy = np.cos(pi * y), np.sin(pi * y)

        return np.array(
            [[(pi**2) * cpx * cpy, -(pi**2) * spx * spy],
             [(pi**2) * spx * spy, -(pi**2) * cpx * cpy]],
            dtype=float,
        )

    def grad_norm_inf_estimate(grad_u_func, Nx_local: int):
        maxnorm = 0.0
        for i in range(Nx_local):
            G = np.asarray(grad_u_func(i))[:2, :2]
            smax = np.linalg.svd(G, compute_uv=False)[0]
            maxnorm = max(maxnorm, float(smax))
        return maxnorm

    Nx_loc = Vx.dofmap.index_map.size_local
    Gu_inf_raw = grad_norm_inf_estimate(grad_u_raw_at, Nx_loc)

    target = np.sqrt(max(1e-16, 1.0 / (Wi * b * dt)))
    vel_scale = min(1.0, 0.9 * target / (Gu_inf_raw + 1e-16))

    def grad_u_at_xslice(i: int):
        return vel_scale * grad_u_raw_at(i)

    u_scaled = vel_scale * u
    Gu_inf = vel_scale * Gu_inf_raw
    dt_max = 1.0 / (Wi * (Gu_inf**2) * b)

    if comm.rank == 0:
        print(f"[Scaling] vel_scale={vel_scale:.3f} ⇒ ||∇u||_∞≈{Gu_inf:.6f}", flush=True)
        print(
            f"[Stability] ||∇u||_∞≈{Gu_inf:.6f}, b={b:.3f} ⇒ dt_max≈{dt_max:.6f}, current dt={dt:.6f}",
            flush=True,
        )
        print("[q-step] FULL ∇u; Λ̃ uses RAW nodal values with λ-clipping.", flush=True)

        # Diagnostic: eigenvalues of grad u (violation if max>0, this is what we want)
        lam_max_local = -np.inf
        lam_min_local = np.inf
        for i in range(Nx_loc):
            G = grad_u_raw_at(i)
            S = 0.5 * (G + G.T)
            vals = np.linalg.eigvalsh(S)
            lam_max_local = max(lam_max_local, float(vals[-1]))
            lam_min_local = min(lam_min_local, float(vals[0]))
        lam_max = comm.allreduce(lam_max_local, op=MPI.MAX)
        lam_min = comm.allreduce(lam_min_local, op=MPI.MIN)
        print(f"[q·∇u·q] sym(∇u) eigenvalue range: [{lam_min:.3e}, {lam_max:.3e}].", flush=True)

    omega_x = lump_weights(Vx)
    omega_q = lump_weights(Vq)

    q = ufl.SpatialCoordinate(Vq.mesh)
    M_x = fem.Constant(Vx.mesh, 1.0)
    M_q = (1.0 - ufl.dot(q, q) / b) ** (b / 2)

    M_floor = 1e-12
    M_q_LHS = ufl.max_value(M_q, M_floor)

    Mq_fun = fem.Function(Vq)

    def Mq_callable(x):
        r2 = x[0] ** 2 + x[1] ** 2
        return np.maximum(1.0 - r2 / b, 0.0) ** (b / 2)

    Mq_fun.interpolate(Mq_callable)
    Mq_vals = Mq_fun.x.array.copy()

    Nx = Vx.dofmap.index_map.size_local
    Nq = Vq.dofmap.index_map.size_local
    psihat_table = np.ones((Nx, Nq), dtype=np.float64)

    xxy_arr = xxy
    qxy = Vq.mesh.geometry.x[:, :2]
    θx = np.arctan2(xxy_arr[:, 1], xxy_arr[:, 0])
    rx2 = xxy_arr[:, 0] ** 2 + xxy_arr[:, 1] ** 2
    θq = np.arctan2(qxy[:, 1], qxy[:, 0])
    rq = np.sqrt(qxy[:, 0] ** 2 + qxy[:, 1] ** 2)

    k, m = 4, 5
    Xmod = (1 + 0.4 * np.cos(k * θx)) * np.exp(-rx2 / 0.6)
    Qmod = (1 + 0.4 * np.sin(m * θq)) * np.exp(-(rq / np.sqrt(b)) ** 2 / 0.35)
    psihat_table += 0.25 * (Xmod[:, None] * Qmod[None, :])
    psihat_table = np.maximum(psihat_table, 0.0)

    W = np.multiply.outer(omega_x, omega_q)
    M_tbl = Mq_vals[None, :]
    M_target = float((W * (np.ones_like(psihat_table) * M_tbl)).sum())
    M_now = float((W * (psihat_table * M_tbl)).sum())
    if M_now > 0:
        psihat_table *= M_target / M_now

    def mass_local(tbl_hat): return float(np.sum(W * (tbl_hat * M_tbl)))
    def neg_mass_local(tbl_hat): return float(np.sum(W * (np.minimum(tbl_hat, 0.0) * M_tbl)))
    def min_local(tbl_hat): return float(tbl_hat.min())

    M0 = comm.allreduce(mass_local(psihat_table), op=MPI.SUM)
    if comm.rank == 0:
        print(f"[Checks] initial mass(ψ) = {M0:.6e}", flush=True)

    qxy_arr = qxy
    R_est = float(np.max(np.linalg.norm(qxy_arr, axis=1)))
    rfs = [0.25, 0.55, 0.80, 0.40]
    angs = [0.0, 1.2, 2.4, 3.6]
    q_targets = [np.array([rf * R_est * np.cos(a), rf * R_est * np.sin(a)]) for rf, a in zip(rfs, angs)]
    slice_js = []
    for t in q_targets:
        j = int(np.argmin(np.linalg.norm(qxy_arr - t, axis=1)))
        if j not in slice_js:
            slice_js.append(j)

    x_targets = [
        np.array([0.5, 0.0]),
        np.array([-0.3, 0.2]),
        np.array([0.0, -0.45]),
        np.array([0.25, 0.35]),
    ]
    slice_is = []
    for t in x_targets:
        i = int(np.argmin(np.linalg.norm(xxy_arr - t, axis=1)))
        if i not in slice_is:
            slice_is.append(i)

    writer = ResultManager(
        Vx, Vq,
        omega_x=omega_x, omega_q=omega_q,
        Mq_vals=Mq_vals,
        outdir=outdir,
        slice_js=slice_js,
        slice_is=slice_is,
        export="psihat",
    )
    writer.write(psihat_table, t=0.0)

    table_path = os.path.join(outdir, "timestep_table.csv")
    if comm.rank == 0:
        with open(table_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dt", "min_psihat", "neg_mass_psihat"])

    def adaptive_fp_tol(n: int, total: int, base_tol: float) -> float:
        frac = n / total
        if frac < 0.3:
            return base_tol * 10
        elif frac < 0.7:
            return base_tol
        else:
            return base_tol * 0.01

    print_every = max(1, num_steps // 100)
    delta_fp = max(delta, 1e-6) 

    for n in range(num_steps):
        tol_now = adaptive_fp_tol(n, num_steps, fp_tol)

        psihat_table = strang_step(
            psihat_table, Vx, Vq,
            M_x,
            M_q_LHS,
            dt, epsilon, Wi,
            lambda phi: tilde_Lambda_x_delta_raw(phi, delta_fp, lam_min=lambda_min, lam_max=lambda_max),
            lambda phi: tilde_Lambda_q_delta_raw(phi, delta_fp, lam_min=lambda_min, lam_max=lambda_max),
            omega_x, omega_q,
            u=u_scaled,
            grad_u_at_xslice=grad_u_at_xslice, 
            delta=delta_fp,
            verbose=(n % print_every == 0),
            max_fp_iter=max_fp_iter,
            fp_tol=tol_now,
            fp_verbose=fp_verbose and (n % print_every == 0),
            relax_theta=relax_theta,
            psi_cap=psi_cap,
        )

        M_now = comm.allreduce(mass_local(psihat_table), op=MPI.SUM)
        mmin = comm.allreduce(min_local(psihat_table), op=MPI.MIN)
        M_neg = comm.allreduce(neg_mass_local(psihat_table), op=MPI.SUM)
        drift = (M_now - M0) / M0 if M0 != 0.0 else 0.0

        if comm.rank == 0:
            with open(table_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([dt, mmin, M_neg])

        if comm.rank == 0 and (n % print_every == 0):
            print(
                f"[Checks] step {n+1}/{num_steps}: "
                f"mass(ψ)={M_now:.6e} (drift {drift:.2e}), "
                f"min ψ̂={mmin:.3e}, neg_mass(ψ̂)={M_neg:.3e}, "
                f"fp_tol_used={tol_now:.1e}",
                flush=True,
            )

        if (n + 1) % print_every == 0:
            writer.write(psihat_table, t=(n + 1) * dt)

    writer.close()
    if comm.rank == 0:
        print("5_4 experiment finished. Final ψ̂ shape:", psihat_table.shape)
        print("Results written to:", outdir)
        print(f"Timestep table written to: {table_path}")

    return psihat_table


if __name__ == "__main__":
    final = run(
        lc_x=0.35, lc_q=0.85,
        max_fp_iter=10,
        fp_tol=1e-3,
        fp_verbose=False,     # set True to see per-slice FP logs from q_fullgrad
        relax_theta=0.3,
        psi_cap=1e6,
        outdir=os.path.join(os.path.dirname(__file__), "results"),
        lambda_min=1e-8,    
        lambda_max=None,   
    )
    print("Final table (ψ̂) shape:", final.shape)