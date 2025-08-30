# utils/function.py
import math
import ufl

# Define F_delta and derivatives in ufl

def F_delta(s, delta):

    branch_lo = (s**2 - delta**2) / (2.0 * delta) + (ufl.ln(delta) - 1.0) * s + 1.0
    branch_hi = (ufl.ln(s) - 1.0) * s + 1.0
    return ufl.conditional(ufl.le(s, delta), branch_lo, branch_hi)

def Fp_delta(s, delta):
    branch_lo = s / delta + ufl.ln(delta) - 1.0
    branch_hi = ufl.ln(s)
    return ufl.conditional(ufl.le(s, delta), branch_lo, branch_hi)

def Fpp_delta(s, delta):
    branch_lo = 1.0 / delta
    branch_hi = 1.0 / s
    return ufl.conditional(ufl.le(s, delta), branch_lo, branch_hi)

# Numerical equivalents

def F_delta_num(s: float, delta: float) -> float:
    if s <= delta:
        return (s * s - delta * delta) / (2.0 * delta) + (math.log(delta) - 1.0) * s + 1.0
    # s > delta => s > 0 so log is defined
    return (math.log(s) - 1.0) * s + 1.0

def Fp_delta_num(s: float, delta: float) -> float:
    if s <= delta:
        return s / delta + math.log(delta) - 1.0
    return math.log(s)

def Fpp_delta_num(s: float, delta: float) -> float:
    if s <= delta:
        return 1.0 / delta
    # s > delta > 0
    return 1.0 / s
