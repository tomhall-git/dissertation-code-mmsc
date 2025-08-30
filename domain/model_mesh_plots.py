#domain/model_mesh_plots.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dolfinx.io import XDMFFile

from .mesh_x import get_domain_x
from .mesh_q import get_domain_q
from ..utils.parameters import b

# Model mesh plots for Appendix

LC_X = 0.15
LC_Q = 0.55
OUTDIR = os.path.join(os.path.dirname(__file__), "mesh_figs")


def _triangles_from_mesh(mesh):

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)
    n_local = mesh.topology.index_map(tdim).size_local
    tri = np.empty((n_local, 3), dtype=np.int64)
    for c in range(n_local):
        verts = c2v.links(c)
        # Expect triangles
        tri[c, :] = verts[:3]
    return tri


def _save_mesh_xdmf(mesh, path):
    with XDMFFile(mesh.comm, path, "w") as xdmf:
        xdmf.write_mesh(mesh)


def _save_mesh_png(mesh, path, title):

    coords = mesh.geometry.x[:, :2]
    triangles = _triangles_from_mesh(mesh)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=12)

    # Draw triangle edges
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)
    ax.triplot(tri, linewidth=0.6)

    # Clean look
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # x-mesh (square)
    mesh_x = get_domain_x(lc=LC_X)
    xdmf_x = os.path.join(OUTDIR, f"model_mesh_x_lc{LC_X:.2f}.xdmf")
    png_x  = os.path.join(OUTDIR, f"model_mesh_x_lc{LC_X:.2f}.png")
    _save_mesh_xdmf(mesh_x, xdmf_x)
    _save_mesh_png(mesh_x, png_x, title=f"x-mesh (lc={LC_X:.2f})")
    print(f"[x-mesh] wrote {xdmf_x} and {png_x}")

    # q-mesh (disk)
    radius = float(np.sqrt(b))
    mesh_q = get_domain_q(lc=LC_Q, radius=radius, b=b)
    xdmf_q = os.path.join(OUTDIR, f"model_mesh_q_lc{LC_Q:.2f}.xdmf")
    png_q  = os.path.join(OUTDIR, f"model_mesh_q_lc{LC_Q:.2f}.png")
    _save_mesh_xdmf(mesh_q, xdmf_q)
    _save_mesh_png(mesh_q, png_q, title=f"q-mesh (lc={LC_Q:.2f}, R=√b≈{radius:.2f})")
    print(f"[q-mesh] wrote {xdmf_q} and {png_q}")


if __name__ == "__main__":
    main()
