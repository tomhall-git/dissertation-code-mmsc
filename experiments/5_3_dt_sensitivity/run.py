# experiments/5_3_dt_sensitivity/run.py
import os
import csv
import numpy as np
from mpi4py import MPI
from dolfinx import fem
import ufl

from ...utils.interpolant import lump_weights
from ...utils.lambdas import tilde_Lambda_x_delta, tilde_Lambda_q_delta
from ...utils.parameters import epsilon, Wi, b, delta  # not importing num_steps anymore
from ...viz.results import ResultManager
from ...solvers.strang_step import strang_step
from ...domain.mesh_x import get_domain_x
from ...domain.mesh_q import get_domain_q

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Code for third experiment - deliberately violate dt constraint (main_solver.py with minor changes)

dt = 0.002
Tfinal = 1.0
num_steps = int(Tfinal / dt)

def _taylor_green_u(mesh):
    x = ufl.SpatialCoordinate(mesh)
    pi = np.pi
    ux = pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    uy = -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    return ufl.as_vector((ux, uy, 0.0))

def main_solver(
    lc_x: float = 0.25,
    lc_q: float = 0.75,
    max_fp_iter: int = 10,
    fp_tol: float = 1e-4,
    fp_verbose: bool = False,
    relax_theta: float = 0.3,
    psi_cap: float = 1e6,
    outdir: str = "results",
):
    print("Running implicit solver...", flush=True)
    comm = MPI.COMM_WORLD
    rank = comm.rank

    os.makedirs(outdir, exist_ok=True)

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
    Gu_inf = grad_norm_inf_estimate(grad_u_raw_at, Nx_loc)
    dt_max = 1.0 / (Wi * (Gu_inf**2) * b)

    if rank == 0:
        ratio = dt / dt_max
        print(
            f"[Stability] ||∇u||_∞≈{Gu_inf:.6f}, b={b:.3f}, Wi={Wi:.2f} "
            f"⇒ dt_max≈{dt_max:.6e}, chosen dt={dt:.6e} (ratio {ratio:.2f})",
            flush=True,
        )
        if dt >= dt_max:
            print("⚠️ [Stability] WARNING: dt exceeds the theoretical bound!", flush=True)
        print(f"[Time] Running for Tfinal={Tfinal}, num_steps={num_steps}", flush=True)

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

    qxy = Vq.mesh.geometry.x[:, :2]
    θx = np.arctan2(xxy[:, 1], xxy[:, 0])
    rx2 = xxy[:, 0] ** 2 + xxy[:, 1] ** 2
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
    def max_local(tbl_hat): return float(tbl_hat.max())

    M0 = comm.allreduce(mass_local(psihat_table), op=MPI.SUM)
    if rank == 0:
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
        i = int(np.argmin(np.linalg.norm(xxy - t, axis=1)))
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

    def adaptive_fp_tol(n: int, total: int, base_tol: float) -> float:
        frac = n / total
        if frac < 0.3:
            return base_tol * 10
        elif frac < 0.7:
            return base_tol
        else:
            return base_tol * 0.01

    print_every = max(1, num_steps // 50)
    delta_fp = max(delta, 1e-6)

    ts, mass_ts, drift_ts, min_ts, max_ts, negmass_ts = [], [], [], [], [], []

    for n in range(num_steps):
        tnow = (n + 1) * dt
        tol_now = adaptive_fp_tol(n, num_steps, fp_tol)

        psihat_table = strang_step(
            psihat_table, Vx, Vq,
            M_x,
            M_q_LHS,
            dt, epsilon, Wi,
            lambda phi: tilde_Lambda_x_delta(phi, delta_fp),
            lambda phi: tilde_Lambda_q_delta(phi, delta_fp),
            omega_x, omega_q,
            u=u,
            grad_u_at_xslice=grad_u_raw_at,
            delta=delta_fp,
            verbose=(n % print_every == 0),
            max_fp_iter=max_fp_iter,
            fp_tol=tol_now,
            fp_verbose=fp_verbose and (n % print_every == 0),
            relax_theta=relax_theta, psi_cap=psi_cap,
        )

        M_now = comm.allreduce(mass_local(psihat_table), op=MPI.SUM)
        mmin = comm.allreduce(min_local(psihat_table), op=MPI.MIN)
        mmax = comm.allreduce(max_local(psihat_table), op=MPI.MAX)
        M_neg = comm.allreduce(neg_mass_local(psihat_table), op=MPI.SUM)
        drift = (M_now - M0) / M0 if M0 != 0.0 else 0.0

        ts.append(tnow); mass_ts.append(M_now); drift_ts.append(drift)
        min_ts.append(mmin); max_ts.append(mmax); negmass_ts.append(M_neg)

        if rank == 0 and (n % print_every == 0):
            print(
                f"[Checks] step {n+1}/{num_steps}: mass={M_now:.3e} (drift {drift:+.2e}), "
                f"min={mmin:.3e}, max={mmax:.3e}, neg_mass={M_neg:.3e}, "
                f"fp_tol_used={tol_now:.1e}",
                flush=True,
            )
            if (np.isnan(M_now) or np.isinf(M_now) or abs(M_now) > 1e12):
                print("Numerical blow-up detected — stopping early.", flush=True)
                break

        if (n + 1) % print_every == 0:
            writer.write(psihat_table, t=tnow)

    writer.close()

    if rank == 0:
        # CSV dump
        csv_path = os.path.join(outdir, "stability_timeseries.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "mass_psi", "mass_drift", "min_psihat", "max_psihat", "negative_mass_psihat",
                        "dt", "dt_max", "dt_over_dtmax", "Wi", "b"])
            for i in range(len(ts)):
                w.writerow([ts[i], mass_ts[i], drift_ts[i], min_ts[i], max_ts[i], negmass_ts[i],
                            dt, dt_max, dt / dt_max, Wi, b])
        print(f"Saved diagnostics CSV: {csv_path}")

        fig1, ax1 = plt.subplots(figsize=(6.0, 3.8))
        ax1.plot(ts, mass_ts, label="mass(ψ)")
        ax1.set_xlabel("time")
        ax1.set_ylabel("mass(ψ)")
        ax1.grid(True, linewidth=0.4, alpha=0.5)
        ax1r = ax1.twinx()
        ax1r.plot(ts, negmass_ts, linestyle="--", label="neg_mass(ψ̂)")
        ax1r.set_ylabel("neg_mass(ψ̂)")
        ax1r.grid(False)
        ax1.set_title(f"Mass & Negativity vs Time (dt/dt_max ≈ {dt/dt_max:.2f})")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1r.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, frameon=False)
        fig1.tight_layout()
        fig1_path = os.path.join(outdir, "mass_and_negativity.png")
        fig1.savefig(fig1_path, dpi=220)
        plt.close(fig1)
        print(f"Saved figure: {fig1_path}")

        fig2, ax2 = plt.subplots(figsize=(6.0, 3.8))
        ax2.semilogy(ts, np.maximum(np.abs(min_ts), 1e-16), label="|min ψ̂|")
        ax2.semilogy(ts, np.maximum(np.abs(max_ts), 1e-16), label="|max ψ̂|")
        ax2.set_xlabel("time")
        ax2.set_ylabel("extrema magnitude (log scale)")
        ax2.grid(True, linewidth=0.4, alpha=0.5)
        ax2.set_title("Extrema of ψ̂ vs Time (log scale)")
        ax2.legend(frameon=False)
        fig2.tight_layout()
        fig2_path = os.path.join(outdir, "extrema_psihat.png")
        fig2.savefig(fig2_path, dpi=220)
        plt.close(fig2)
        print(f"Saved figure: {fig2_path}")

    return psihat_table


if __name__ == "__main__":
    final_table = main_solver(
        lc_x=0.35, lc_q=0.95,
        max_fp_iter=6,
        fp_tol=1e-3,
        fp_verbose=True,
        relax_theta=0.3,
        psi_cap=1e6,
        outdir="results",
    )
    print("Implicit solver finished. Final table (ψ̂) shape:", final_table.shape)