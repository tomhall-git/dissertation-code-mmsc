# domain/mesh_x.py
import gmsh
import numpy as np
from dolfinx.io import gmshio, XDMFFile
from dolfinx.mesh import Mesh
from mpi4py import MPI

# --------- Box mesh [0,1]^2 with boundary tagging ---------

def _count_obtuse(mesh: Mesh, deg_tol: float = 1e-12):
    mesh.topology.create_connectivity(2, 0)
    c2v = mesh.topology.connectivity(2, 0)
    X = mesh.geometry.x
    n_local = mesh.topology.index_map(2).size_local
    obtuse = 0
    max_angle = 0.0
    for c in range(n_local):
        verts = c2v.links(c)
        tri = X[verts]
        a = np.linalg.norm(tri[1] - tri[2])
        b = np.linalg.norm(tri[0] - tri[2])
        cL = np.linalg.norm(tri[0] - tri[1])

        def ang(L, A, B):
            cosA = (A*A + B*B - L*L) / (2*A*B)
            cosA = np.clip(cosA, -1.0, 1.0)
            return np.degrees(np.arccos(cosA))

        a1, a2, a3 = ang(a, b, cL), ang(b, a, cL), ang(cL, a, b)
        local_max = max(a1, a2, a3)
        if local_max > 90.0 + deg_tol:
            obtuse += 1
            max_angle = max(max_angle, float(local_max))
    comm = mesh.comm
    return (
        comm.allreduce(obtuse, op=MPI.SUM),
        comm.allreduce(max_angle, op=MPI.MAX),
        comm.allreduce(mesh.topology.index_map(2).size_local, op=MPI.SUM),
    )

def build_domain_x_box(lx: float = 1.0, ly: float = 1.0, lc: float = 0.25,
                       ensure_weakly_acute: bool = True,
                       max_retries: int = 2, shrink: float = 0.75):
    """
    Build a 2D box mesh [0,lx]×[0,ly] with target size lc.
    Boundary tags (1: left, 2: right, 3: bottom, 4: top), cells tag 11.
    """
    gmsh.initialize()
    try:
        attempt = 0
        cur_lc = float(lc)
        while True:
            gmsh.model.add("box_x")
            p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
            p2 = gmsh.model.occ.addPoint(lx, 0.0, 0.0)
            p3 = gmsh.model.occ.addPoint(lx, ly, 0.0)
            p4 = gmsh.model.occ.addPoint(0.0, ly, 0.0)
            l1 = gmsh.model.occ.addLine(p1, p2)  # bottom
            l2 = gmsh.model.occ.addLine(p2, p3)  # right
            l3 = gmsh.model.occ.addLine(p3, p4)  # top
            l4 = gmsh.model.occ.addLine(p4, p1)  # left
            cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
            s = gmsh.model.occ.addPlaneSurface([cl])
            gmsh.model.occ.synchronize()

            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cur_lc)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cur_lc)

            # Physical groups: domain and each side
            pg_cells = gmsh.model.addPhysicalGroup(2, [s], tag=11)
            gmsh.model.setPhysicalName(2, pg_cells, "Ω_box")
            pg_left  = gmsh.model.addPhysicalGroup(1, [l4], tag=1); gmsh.model.setPhysicalName(1, pg_left , "left")
            pg_right = gmsh.model.addPhysicalGroup(1, [l2], tag=2); gmsh.model.setPhysicalName(1, pg_right, "right")
            pg_bot   = gmsh.model.addPhysicalGroup(1, [l1], tag=3); gmsh.model.setPhysicalName(1, pg_bot  , "bottom")
            pg_top   = gmsh.model.addPhysicalGroup(1, [l3], tag=4); gmsh.model.setPhysicalName(1, pg_top  , "top")

            gmsh.model.mesh.generate(2)

            mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
                gmsh.model, comm=MPI.COMM_WORLD, rank=0
            )

            obtuse, max_ang, n_cells = _count_obtuse(mesh)
            if mesh.comm.rank == 0:
                if obtuse == 0:
                    print(f"[mesh_x box] weakly acute ✓  (cells={n_cells}, lc={cur_lc})")
                else:
                    print(f"[mesh_x box] obtuse={obtuse} (worst {max_ang:.2f}°), cells={n_cells}, lc={cur_lc}")

            if not ensure_weakly_acute or obtuse == 0 or attempt >= max_retries:
                return mesh, cell_tags, facet_tags

            attempt += 1
            cur_lc *= shrink
            gmsh.clear()
    finally:
        gmsh.finalize()

# --------- Parameterised getters (no singletons) ---------

_mesh_cache_x = {}

def get_domain_x(lc: float = 0.25, lx: float = 1.0, ly: float = 1.0,
                 ensure_weakly_acute: bool = True):
    """
    Return Mesh for given charlen `lc` (cached). Most callers only need the Mesh.
    """
    key = (lx, ly, lc, ensure_weakly_acute)
    if key not in _mesh_cache_x:
        _mesh_cache_x[key] = build_domain_x_box(lx=lx, ly=ly, lc=lc,
                                                ensure_weakly_acute=ensure_weakly_acute)
    return _mesh_cache_x[key][0]

def get_domain_x_full(lc: float = 0.25, lx: float = 1.0, ly: float = 1.0,
                      ensure_weakly_acute: bool = True):
    """
    Return (mesh, cell_tags, facet_tags) for given `lc`.
    """
    key = (lx, ly, lc, ensure_weakly_acute)
    if key not in _mesh_cache_x:
        _mesh_cache_x[key] = build_domain_x_box(lx=lx, ly=ly, lc=lc,
                                                ensure_weakly_acute=ensure_weakly_acute)
    return _mesh_cache_x[key]

if __name__ == "__main__":
    m = get_domain_x(lc=0.25)
    print("mesh dim:", m.topology.dim, "| cells:",
          m.topology.index_map(m.topology.dim).size_local)
    with XDMFFile(m.comm, "domain/physical_mesh_box.xdmf", "w") as xdmf:
        xdmf.write_mesh(m)
