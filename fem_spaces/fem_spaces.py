# fem_spaces/fem_spaces.py
from dolfinx import fem
from dolfinx.mesh import Mesh

# Scalar CG spaces for x and q meshes

def Vx_from_mesh(mesh_x: Mesh, degree: int = 1):
    return fem.functionspace(mesh_x, ("CG", degree))

def Vq_from_mesh(mesh_q: Mesh, degree: int = 1):
    return fem.functionspace(mesh_q, ("CG", degree))
