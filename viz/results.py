# viz/results.py
from __future__ import annotations
import os
import numpy as np
from dolfinx import fem
from dolfinx.io import XDMFFile

from ..utils.interpolant import as_Vx_function, as_Vq_function

# Handling of XDMF time series for visualisation
class ResultManager:

    def __init__(self, Vx, Vq, omega_x, omega_q, Mq_vals,
                 outdir="results", slice_js=None, slice_is=None,
                 export: str = "psihat"):  # "psi" or "psihat"
        self.Vx, self.Vq = Vx, Vq
        self.ox, self.oq = np.asarray(omega_x), np.asarray(omega_q)
        self.Mq = np.asarray(Mq_vals)
        assert export in ("psi", "psihat")
        self.export = export

        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir

        # Marginal writers
        self.xdmf_rho_x = XDMFFile(Vx.mesh.comm, os.path.join(outdir, "rho_x.xdmf"), "w")
        self.xdmf_rho_x.write_mesh(Vx.mesh)
        self.xdmf_rho_q = XDMFFile(Vq.mesh.comm, os.path.join(outdir, "rho_q.xdmf"), "w")
        self.xdmf_rho_q.write_mesh(Vq.mesh)

        # Choose slice indices
        Nx = Vx.dofmap.index_map.size_local
        Nq = Vq.dofmap.index_map.size_local

        if slice_js is None:
            Q = Vq.mesh.geometry.x[:, :2]
            R = float(np.max(np.linalg.norm(Q, axis=1)))
            rfs  = [0.25, 0.55, 0.80, 0.40]
            angs = [0.3, 1.4, 2.5, 3.6]  # avoid 0 (x-axis) and symmetry
            targets = [np.array([rf*R*np.cos(a), rf*R*np.sin(a)]) for rf, a in zip(rfs, angs)]
            slice_js = []
            for t in targets:
                j = int(np.argmin(np.linalg.norm(Q - t, axis=1)))
                if j not in slice_js:
                    slice_js.append(j)
            if len(slice_js) < 4 and Nq > 0:
                cand = int(np.argmax(self.Mq))
                if cand not in slice_js:
                    slice_js.append(cand)

        if slice_is is None:
            X = Vx.mesh.geometry.x[:, :2]
            targets = [np.array([0.4, 0.0]),
                       np.array([-0.3, 0.25]),
                       np.array([0.0, -0.35]),
                       np.array([0.25, 0.35])]
            slice_is = []
            for t in targets:
                i = int(np.argmin(np.linalg.norm(X - t, axis=1)))
                if i not in slice_is:
                    slice_is.append(i)
            # Fallback: include centre node
            if len(slice_is) < 4 and Nx > 0:
                cand = Nx // 2
                if cand not in slice_is:
                    slice_is.append(cand)

        self.slice_js = slice_js
        self.slice_is = slice_is

        # XDMF writers
        self.xdmf_x_slices = []
        for j in self.slice_js:
            path = os.path.join(outdir, f"{'psi' if export=='psi' else 'psihat'}_x__q{j}.xdmf")
            f = XDMFFile(Vx.mesh.comm, path, "w")
            f.write_mesh(Vx.mesh)
            self.xdmf_x_slices.append(f)

        self.xdmf_q_slices = []
        for i in self.slice_is:
            path = os.path.join(outdir, f"{'psi' if export=='psi' else 'psihat'}_q__x{i}.xdmf")
            f = XDMFFile(Vq.mesh.comm, path, "w")
            f.write_mesh(Vq.mesh)
            self.xdmf_q_slices.append(f)

        # Reusable Functions
        self._rho_x_fun = fem.Function(Vx)
        self._rho_q_fun = fem.Function(Vq)
        self._x_slice_fun = fem.Function(Vx)
        self._q_slice_fun = fem.Function(Vq)

    def write(self, psi_hat_table: np.ndarray, t: float):
        rho_x_vals = psi_hat_table @ (self.oq * self.Mq)  # (Nx,)
        self._rho_x_fun.x.array[:] = rho_x_vals
        self.xdmf_rho_x.write_function(self._rho_x_fun, t)

        rho_q_vals = (self.ox @ psi_hat_table) * self.Mq  # (Nq,)
        self._rho_q_fun.x.array[:] = rho_q_vals
        self.xdmf_rho_q.write_function(self._rho_q_fun, t)

        if self.export == "psi":
            psi_table = psi_hat_table * self.Mq[None, :]  # along x
        else:
            psi_table = psi_hat_table

        for f, j in zip(self.xdmf_x_slices, self.slice_js):
            self._x_slice_fun.x.array[:] = psi_table[:, j]
            f.write_function(self._x_slice_fun, t)

        if self.export == "psi":
            for f, i in zip(self.xdmf_q_slices, self.slice_is):
                self._q_slice_fun.x.array[:] = psi_hat_table[i, :] * self.Mq
                f.write_function(self._q_slice_fun, t)
        else:
            for f, i in zip(self.xdmf_q_slices, self.slice_is):
                self._q_slice_fun.x.array[:] = psi_hat_table[i, :]
                f.write_function(self._q_slice_fun, t)

    def close(self):
        self.xdmf_rho_x.close()
        self.xdmf_rho_q.close()
        for f in self.xdmf_x_slices:
            f.close()
        for f in self.xdmf_q_slices:
            f.close()
