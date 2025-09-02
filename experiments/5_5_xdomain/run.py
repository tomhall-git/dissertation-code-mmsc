# experiments/5_1_baseline/run_lshape_zero_u.py
import os, csv, numpy as np
from mpi4py import MPI
from dolfinx import fem
import ufl
from dolfinx.io import XDMFFile

from ...utils.interpolant import lump_weights
from ...utils.lambdas import tilde_Lambda_x_delta, tilde_Lambda_q_delta
from ...utils.parameters import epsilon, Wi, dt, num_steps, b, delta
from ...viz.results import ResultManager
from ...solvers.strang_step import strang_step
from .mesh_x_lshape import get_domain_x       # L-shaped mesh
from ...domain.mesh_q import get_domain_q


def run():
    outdir = os.path.join(os.path.dirname(__file__), "results_lshape_zero_u")
    os.makedirs(outdir, exist_ok=True)

    _this_dir = os.path.dirname(__file__)
    _experiments_dir = os.path.dirname(_this_dir)
    xheat_dir = os.path.join(_experiments_dir, "5_5_xdomain", "results")
    os.makedirs(xheat_dir, exist_ok=True)
    xheat_path = os.path.join(xheat_dir, "x_heatmap_timeseries.xdmf")

    comm = MPI.COMM_WORLD
    rank0 = (comm.rank == 0)

    csv_path = os.path.join(outdir, "mass_timeseries.csv")
    if rank0:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "time", "mass", "mass_drift", "min_psihat", "neg_mass", "fp_tol_used"])

    lc_x, lc_q = 0.25, 0.85
    max_fp_iter, fp_tol_base = 6, 1e-3
    fp_verbose, relax_theta, psi_cap = False, 0.3, 1e6

    print("Running implicit solver on L-shape with incompressible, no-flux, corotational u...", flush=True)

    mesh_x = get_domain_x(lc=lc_x)          
    mesh_q = get_domain_q(lc=lc_q, radius=np.sqrt(b), b=b)
    Vx = fem.functionspace(mesh_x, ("CG", 1))
    Vq = fem.functionspace(mesh_q, ("CG", 1))

    xcoord = ufl.SpatialCoordinate(Vx.mesh)
    X, Y = xcoord[0], xcoord[1]

    A = X * (2.0 - X) * Y * (1.0 - Y)            
    B = (X - 1.0) ** 2 * (Y - 0.25) ** 2       
    psi = A * B

    vel_scale = 4.0
    gpsi = ufl.grad(psi)
    ux = gpsi[1]
    uy = -gpsi[0]
    u = ufl.as_vector((vel_scale * ux, vel_scale * uy, 0.0))

    xxy = Vx.mesh.geometry.x[:, :2]  

    def _grad_u_xy(x, y):
        A     = x*(2.0 - x) * y*(1.0 - y)
        A_x   = (2.0 - 2.0*x) * y*(1.0 - y)
        A_y   = x*(2.0 - x) * (1.0 - 2.0*y)
        A_xx  = -2.0 * y*(1.0 - y)
        A_yy  = -2.0 * x*(2.0 - x)
        A_xy  = (2.0 - 2.0*x) * (1.0 - 2.0*y)

        dx = (x - 1.0)
        dy = (y - 0.25)
        B     = (dx*dx) * (dy*dy)
        B_x   = 2.0*dx * (dy*dy)
        B_y   = 2.0*dy * (dx*dx)
        B_xx  = 2.0 * (dy*dy)
        B_yy  = 2.0 * (dx*dx)
        B_xy  = 4.0 * dx * dy

        psi_xx = A_xx*B + 2.0*A_x*B_x + A*B_xx
        psi_yy = A_yy*B + 2.0*A_y*B_y + A*B_yy
        psi_xy = A_xy*B + A_x*B_y + A_y*B_x + A*B_xy  # = ψ_yx

        return np.array([[psi_xy,  psi_yy],
                         [-psi_xx, -psi_xy]], dtype=float) * vel_scale

    def grad_u_at_xslice(i: int):
        x, y = float(xxy[i, 0]), float(xxy[i, 1])
        return _grad_u_xy(x, y)

    gnorm = 0.0
    for xy in xxy:
        G = _grad_u_xy(float(xy[0]), float(xy[1]))
        gnorm = max(gnorm, float(np.linalg.norm(G)))
    print(f"[Scaling] vel_scale={vel_scale} ⇒ approx. ||∇u||_∞≈{gnorm:.3f}", flush=True)

    omega_x = lump_weights(Vx); omega_q = lump_weights(Vq)

    q = ufl.SpatialCoordinate(Vq.mesh)
    M_x = fem.Constant(Vx.mesh, 1.0)
    M_q = (1.0 - ufl.dot(q, q) / b) ** (b / 2)
    M_q_LHS = ufl.max_value(M_q, 1e-12)

    Mq_fun = fem.Function(Vq)
    def Mq_callable(x):
        r2 = x[0]**2 + x[1]**2
        return np.maximum(1.0 - r2 / b, 0.0) ** (b / 2)
    Mq_fun.interpolate(Mq_callable)
    Mq_vals = Mq_fun.x.array.copy()

    Nx = Vx.dofmap.index_map.size_local
    Nq = Vq.dofmap.index_map.size_local
    psihat_table = np.ones((Nx, Nq), dtype=np.float64)

    xxy = Vx.mesh.geometry.x[:, :2]
    qxy = Vq.mesh.geometry.x[:, :2]
        # --- Corner-localised perturbation at (1, 0.25) in x, mild modulation in q ---
    corner = np.array([1.0, 0.25])
    dx = xxy[:, 0] - corner[0]
    dy = xxy[:, 1] - corner[1]

    # Anisotropic Gaussian aligned with the two legs of the L
    # (narrow across the corner, longer along the legs)
    sigma_x = 0.18   # along x (rightward leg)
    sigma_y = 0.12   # along y (upward leg)
    Xbump = np.exp(- (dx/sigma_x)**2 - (dy/sigma_y)**2)

    # Optional: add a tiny tangential ripple to break symmetry near the corner (can omit)
    Xmod = 1.0 + 0.45 * Xbump

    # Keep your q-mod as before (or set Qmod = 1.0 for pure x-local perturbation)
    θq = np.arctan2(qxy[:, 1], qxy[:, 0])
    rq = np.sqrt(qxy[:, 0]**2 + qxy[:, 1]**2)
    m = 5
    Qmod = (1 + 0.3*np.sin(m*θq)) * np.exp(-(rq/np.sqrt(b))**2 / 0.35)

    # Apply separable perturbation and enforce nonnegativity
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
    if rank0:
        print(f"[Checks] initial mass(ψ) = {M0:.6e}", flush=True)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([0, 0.0, M0, 0.0, min_local(psihat_table), neg_mass_local(psihat_table), "n/a"])

    def nearest_i(pt):
        return int(np.argmin(np.linalg.norm(xxy - np.asarray(pt), axis=1)))
    slice_is = []
    for pt in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.60, 0.40)]:
        i = nearest_i(pt)
        if i not in slice_is:
            slice_is.append(i)

    qxy_arr = qxy
    R_est = float(np.max(np.linalg.norm(qxy_arr, axis=1)))
    rfs, angs = [0.25, 0.55, 0.80, 0.40], [0.0, 1.2, 2.4, 3.6]
    slice_js = []
    for rf, a in zip(rfs, angs):
        tgt = np.array([rf * R_est * np.cos(a), rf * R_est * np.sin(a)])
        j = int(np.argmin(np.linalg.norm(qxy_arr - tgt, axis=1)))
        if j not in slice_js:
            slice_js.append(j)

    writer = ResultManager(Vx, Vq,
                           omega_x=omega_x, omega_q=omega_q,
                           Mq_vals=Mq_vals, outdir=outdir,
                           slice_js=slice_js, slice_is=slice_is,
                           export="psihat")
    writer.write(psihat_table, t=0.0)

    x_heat = fem.Function(Vx, name="psi_x_marginal")
    def update_x_heat_from_table(tbl):
        wqM = omega_q * Mq_vals
        x_heat.x.array[:] = tbl @ wqM 

    if rank0:
        print(f"[XDMF] writing x heatmap to: {xheat_path}")
    xdmf_x = XDMFFile(Vx.mesh.comm, xheat_path, "w")
    xdmf_x.write_mesh(Vx.mesh)
    update_x_heat_from_table(psihat_table)
    xdmf_x.write_function(x_heat, t=0.0)

    def adaptive_fp_tol(n, total, base):
        f = n / total
        return base*10 if f < 0.3 else base if f < 0.7 else base*0.01

    print_every = max(1, num_steps // 100)
    delta_fp = max(delta, 1e-6)

    for n in range(num_steps):
        tol_now = adaptive_fp_tol(n, num_steps, fp_tol_base)

        psihat_table = strang_step(
            psihat_table, Vx, Vq,
            M_x, M_q_LHS,
            dt, epsilon, Wi,
            lambda phi: tilde_Lambda_x_delta(phi, delta_fp),
            lambda phi: tilde_Lambda_q_delta(phi, delta_fp),
            omega_x, omega_q,
            u=u,                                  
            grad_u_at_xslice=grad_u_at_xslice,   
            delta=delta_fp,
            verbose=(n % print_every == 0),
            max_fp_iter=max_fp_iter,
            fp_tol=tol_now,
            fp_verbose=fp_verbose and (n % print_every == 0),
            relax_theta=relax_theta, psi_cap=psi_cap,
        )

        M_now = comm.allreduce(mass_local(psihat_table), op=MPI.SUM)
        mmin  = comm.allreduce(min_local(psihat_table), op=MPI.MIN)
        M_neg = comm.allreduce(neg_mass_local(psihat_table), op=MPI.SUM)
        drift = (M_now - M0) / M0 if M0 != 0 else 0.0

        if rank0 and (n % print_every == 0):
            print(f"[Checks] step {n+1}/{num_steps}: mass={M_now:.6e} (drift {drift:.2e}), "
                  f"min ψ̂={mmin:.3e}, neg_mass={M_neg:.3e}, fp_tol={tol_now:.1e}", flush=True)

        if rank0:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([n+1, (n+1)*dt,
                    f"{M_now:.16e}", f"{drift:.16e}", f"{mmin:.16e}", f"{M_neg:.16e}", f"{tol_now:.1e}"])

        if (n + 1) % print_every == 0:
            writer.write(psihat_table, t=(n + 1) * dt)

            update_x_heat_from_table(psihat_table)
            xdmf_x.write_function(x_heat, t=(n + 1) * dt)

    writer.close()
    xdmf_x.close()

    if rank0:
        print("L-shape run finished. Final ψ̂ shape:", psihat_table.shape)
        print("Mass/negativity time-series:", csv_path)
        print("X heatmap XDMF:", xheat_path)
        print("Results:", outdir)

if __name__ == "__main__":
    run()