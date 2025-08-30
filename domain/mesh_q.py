# domain/mesh_q.py
import gmsh
import numpy as np
from dolfinx.io import gmshio, XDMFFile
from dolfinx.mesh import Mesh, locate_entities_boundary, entities_to_geometry
from mpi4py import MPI

# Circular mesh with weak acuteness check and automatic refinement

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
            if local_max > max_angle:
                max_angle = float(local_max)
    comm = mesh.comm
    return (comm.allreduce(obtuse, op=MPI.SUM),
            comm.allreduce(max_angle, op=MPI.MAX),
            comm.allreduce(mesh.topology.index_map(2).size_local, op=MPI.SUM))

# Check M value on dD

def _maxwellian(q, b):
    r2 = float(q[0]*q[0] + q[1]*q[1])
    return (1.0 - r2 / b) ** (b / 2) if r2 < b else 0.0

def _report_boundary_M(mesh: Mesh, b: float, tol: float = 1e-10):
    bfacets = locate_entities_boundary(mesh, dim=1,
                                       marker=lambda x: np.full(x.shape[1], True))
    bverts = np.unique(entities_to_geometry(mesh, 1, bfacets))
    X = mesh.geometry.x
    Mvals = np.array([_maxwellian(X[v], b) for v in bverts]) if bverts.size else np.array([0.0])
    maxM = float(Mvals.max())
    if mesh.comm.rank == 0:
        print(f"[mesh_q] max M on ∂D (vertices): {maxM:.3e}"
              + ("" if maxM <= tol else "  (polygonal boundary ⇒ not exactly zero)"))

def build_domain_q_disk(radius: float = np.sqrt(30), lc: float = 0.75,
                        ensure_weakly_acute: bool = True,
                        max_retries: int = 3, shrink: float = 0.8,
                        check_boundary_M: bool = True, b: float = 30.0):
    gmsh.initialize()
    try:
        attempt = 0
        cur_lc = float(lc)
        mesh = cell_tags = facet_tags = None

        while True:
            gmsh.model.add("disk_q")
            surf_tag = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, radius, radius)
            gmsh.model.occ.synchronize()

            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cur_lc)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cur_lc)
            gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
            gmsh.model.mesh.generate(2)

            mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
                gmsh.model, comm=MPI.COMM_WORLD, rank=0
            )

            obtuse, max_ang, n_cells = _count_obtuse(mesh)
            if mesh.comm.rank == 0:
                if obtuse == 0:
                    print(f"[mesh_q] weakly acute ✓  (cells={n_cells}, lc={cur_lc})")
                else:
                    print(f"[mesh_q] obtuse={obtuse} (worst {max_ang:.2f}°), "
                          f"cells={n_cells}, lc={cur_lc}")

            if check_boundary_M:
                _report_boundary_M(mesh, b)

            if not ensure_weakly_acute or obtuse == 0 or attempt >= max_retries:
                break

            attempt += 1
            cur_lc *= shrink
            gmsh.clear()

        return mesh, cell_tags, facet_tags
    finally:
        gmsh.finalize()

_mesh_cache_q = {}

def get_domain_q(lc: float = 0.75, radius: float = np.sqrt(30),
                 ensure_weakly_acute: bool = True, b: float = 30.0,
                 check_boundary_M: bool = True):
    key = (radius, lc, ensure_weakly_acute, b, check_boundary_M)
    if key not in _mesh_cache_q:
        _mesh_cache_q[key] = build_domain_q_disk(
            radius=radius, lc=lc, ensure_weakly_acute=ensure_weakly_acute,
            b=b, check_boundary_M=check_boundary_M
        )
    return _mesh_cache_q[key][0]

def get_domain_q_full(lc: float = 0.75, radius: float = np.sqrt(30),
                      ensure_weakly_acute: bool = True, b: float = 30.0,
                      check_boundary_M: bool = True):
    key = (radius, lc, ensure_weakly_acute, b, check_boundary_M)
    if key not in _mesh_cache_q:
        _mesh_cache_q[key] = build_domain_q_disk(
            radius=radius, lc=lc, ensure_weakly_acute=ensure_weakly_acute,
            b=b, check_boundary_M=check_boundary_M
        )
    return _mesh_cache_q[key]

if __name__ == "__main__":
    m = get_domain_q(lc=0.75, radius=np.sqrt(30), b=30.0)
    print("mesh dim:", m.topology.dim, "| cells:",
          m.topology.index_map(m.topology.dim).size_local)
    with XDMFFile(m.comm, "domain/config_mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(m)
