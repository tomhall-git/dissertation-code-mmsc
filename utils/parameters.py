# utils/parameters.py
from __future__ import annotations
import numpy as np
import ufl

# Global parameters for experiments 

epsilon = 0.1
final_time = 1.0
num_steps = 250
dt = final_time / num_steps
fe_degree: int = 1
Wi = 1.0
delta = 1e-8
b = 30.0
R = np.sqrt(b)

# Full velocity field (Taylor-Green) and skew-grad

def taylor_green_u(mesh) -> ufl.core.expr.Expr:
    x = ufl.SpatialCoordinate(mesh)
    pi = np.pi
    ux = pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    uy = -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    return ufl.as_vector((ux, uy, 0.0))

def skew_grad_2x2(u_vec3: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    J = ufl.grad(u_vec3) 
    J2 = ufl.as_matrix(((J[0, 0], J[0, 1]),
                        (J[1, 0], J[1, 1])))
    return 0.5 * (J2 - ufl.transpose(J2))

