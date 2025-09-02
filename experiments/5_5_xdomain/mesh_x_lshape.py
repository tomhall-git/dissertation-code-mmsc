# experiments/5_5_xdomain/mesh_x_lshape.py
from mpi4py import MPI
import gmsh, os, numpy as np
from dolfinx.io import gmshio, XDMFFile
from dolfinx.mesh import Mesh

# Output directory
EXP_DIR = os.path.dirname(__file__)
MESH_DIR = os.path.join(EXP_DIR, "meshes")
os.makedirs(MESH_DIR, exist_ok=True)


def _count_obtuse(mesh: Mesh, deg_tol: float = 1e-12):
    if mesh.topology.dim != 2:
        raise ValueError("Weak-acuteness checker assumes a 2D topological mesh.")
    mesh.topology.create_connectivity(2, 0)
    c2v = mesh.topology.connectivity(2, 0)
    X = mesh.geometry.x  # (N, gdim); z=0 if gdim=3

    def side_len(P, Q):
        d = P[:2] - Q[:2]
        return float(np.sqrt(np.dot(d, d)))

    def angle_deg(L, A, B):
        cosA = (A*A + B*B - L*L) / (2.0 * A * B)
        cosA = np.clip(cosA, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosA)))

    n_local = mesh.topology.index_map(2).size_local
    obtuse = 0
    max_angle = 0.0
    for c in range(n_local):
        verts = c2v.links(c)
        tri = X[verts]
        a = side_len(tri[1], tri[2])
        b = side_len(tri[0], tri[2])
        cL = side_len(tri[0], tri[1])
        a1 = angle_deg(a, b, cL)
        a2 = angle_deg(b, a, cL)
        a3 = angle_deg(cL, a, b)
        mA = max(a1, a2, a3)
        if mA > 90.0 + deg_tol:
            obtuse += 1
            if mA > max_angle:
                max_angle = mA

    comm = mesh.comm
    return (
        comm.allreduce(obtuse, op=MPI.SUM),
        comm.allreduce(max_angle, op=MPI.MAX),
        comm.allreduce(mesh.topology.index_map(2).size_local, op=MPI.SUM),
    )

# Build L-shaped mesh

def get_domain_x(lc: float = 0.25,
                 ensure_weakly_acute: bool = True,
                 lc_min_factor: float = 0.30,   
                 r_in: float = 0.35,       
                 r_out: float = 1.20,         
                 max_retries: int = 0) -> Mesh:
    gmsh.initialize()
    try:
        for attempt in range(max_retries + 1):
            gmsh.model.add("Lshape_2x1_notch")

            occ = gmsh.model.occ
            outer = occ.addRectangle(0.0, 0.0, 0.0, 2.0, 1.0)
            notch = occ.addRectangle(1.0, 0.25, 0.0, 1.0, 0.75)
            occ.synchronize()
            (cut, _) = occ.cut([(2, outer)], [(2, notch)],
                               removeObject=True, removeTool=True)
            occ.synchronize()

            corner_pt = None
            bb = gmsh.model.getEntitiesInBoundingBox(
                1.0 - 1e-9, 0.25 - 1e-9, -1e-9,
                1.0 + 1e-9, 0.25 + 1e-9,  1e-9, dim=0
            )
            if bb:
                corner_pt = bb[0][1]
            else:
                for _, tag in gmsh.model.getEntities(0):
                    x, y, _ = gmsh.model.getValue(0, tag, [])
                    if abs(x-1.0) < 1e-12 and abs(y-0.25) < 1e-12:
                        corner_pt = tag
                        break
            if corner_pt is None:
                raise RuntimeError("Could not find re-entrant corner point.")

            inner_vert = None  
            inner_horz = None  
            for _, tag in gmsh.model.getEntities(1):
                xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, tag)
                if abs(xmin - xmax) < 1e-12 and abs(xmin - 1.0) < 1e-8 and ymin >= -1e-12 and ymax <= 0.25 + 1e-8:
                    inner_vert = tag
                if abs(ymin - ymax) < 1e-12 and abs(ymin - 0.25) < 1e-8 and xmin >= 1.0 - 1e-8 and xmax <= 2.0 + 1e-8:
                    inner_horz = tag
            inner_edges = [t for t in [inner_vert, inner_horz] if t is not None]

            for _, tag in gmsh.model.getEntities(0):
                gmsh.model.mesh.setSize([(0, tag)], lc)

            f_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", [corner_pt])
            if inner_edges:
                gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", inner_edges)

            f_thr = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
            gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", lc * lc_min_factor)
            gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", lc)
            gmsh.model.mesh.field.setNumber(f_thr, "DistMin", r_in)
            gmsh.model.mesh.field.setNumber(f_thr, "DistMax", r_out)
            gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)

            if inner_vert is not None:
                n_v = max(3, int(np.ceil(0.25 / (lc * 0.5))))
                gmsh.model.mesh.setTransfiniteCurve(inner_vert, n_v)
            if inner_horz is not None:
                n_h = max(3, int(np.ceil(1.0 / (lc * 0.8))))
                gmsh.model.mesh.setTransfiniteCurve(inner_horz, n_h)

            # Meshing weakly acute is harder here - try different options
            gmsh.model.mesh.setOrder(1)
            gmsh.option.setNumber("Mesh.Algorithm", 5)    
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)   
            gmsh.option.setNumber("Mesh.Smoothing", 10)     
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

            gmsh.model.mesh.generate(2)

            try:
                gmsh.model.mesh.optimize("Netgen")
                gmsh.model.mesh.optimize("Netgen")
            except Exception:
                pass

            surf_tags = [t for d, t in cut if d == 2]
            if surf_tags:
                ps = gmsh.model.addPhysicalGroup(2, surf_tags, tag=401)
                gmsh.model.setPhysicalName(2, ps, "Ω_L_2x1")
            edge_tags = [t for d, t in gmsh.model.getEntities(1)]
            if edge_tags:
                pb = gmsh.model.addPhysicalGroup(1, edge_tags, tag=402)
                gmsh.model.setPhysicalName(1, pb, "∂Ω_L_2x1")

            mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
                gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=3
            )

            obtuse, max_ang, n_cells = _count_obtuse(mesh)
            if mesh.comm.rank == 0:
                if obtuse == 0:
                    print(f"[mesh_x L-2x1] weakly acute ✓  (cells={n_cells}, lc={lc})")
                else:
                    print(f"[mesh_x L-2x1] obtuse={obtuse} (worst {max_ang:.2f}°), "
                          f"cells={n_cells}, lc={lc}")
                out = os.path.join(MESH_DIR, f"Lshape_2x1_notch_lc{lc:.4f}_try{attempt}.xdmf")
                with XDMFFile(mesh.comm, out, "w") as xdmf:
                    xdmf.write_mesh(mesh)
                print(f"[mesh_x L-2x1] wrote {out}")

            if obtuse == 0 or not ensure_weakly_acute or attempt == max_retries:
                return mesh

            lc_min_factor *= 0.8 
            gmsh.clear()

        return mesh  

    finally:
        gmsh.finalize()

if __name__ == "__main__":
    m = get_domain_x(lc=0.25, ensure_weakly_acute=True)
    if m.comm.rank == 0:
        print("mesh dim:", m.topology.dim,
              "| cells:", m.topology.index_map(m.topology.dim).size_local)
