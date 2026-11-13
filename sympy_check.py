r"""
Symbolic checks (SymPy) of two algebraic statements used in the paper.

1. Hessian formula, eq. (8):  for one link (x -> y) with U arbitrary,
       F(omega) = - Re Tr[ exp(i w_x.T) U exp(-i w_y.T) ],
   its quadratic Taylor coefficient equals
       1/2 [ w_x^T Mxx w_x + w_y^T Myy w_y + 2 w_x^T Mxy w_y ]
   with Mxx_ab = Re Tr({Ta,Tb}/2 U), Myy_ab = Re Tr(U {Ta,Tb}/2),
   Mxy_ab = - Re Tr(Ta U Tb).  Done exactly for SU(2) (T = sigma/2), using the
   closed form exp(i w.sigma/2) = cos(r) + i sin(r)/r (w.sigma/2), r = |w|/2,
   with no truncated series in the group element.  The identity is linear in U,
   so a fully general 2x2 complex U is used.

2. Proposition 1:  -d_mu (D_mu xi) = Delta_A xi + [A_mu, D_mu xi],
   for su(2)-valued fields depending on four coordinates, with no gauge
   condition imposed (the identity is purely algebraic).
"""
import sympy as sp
from sympy import I, Matrix, Rational, symbols, cos, sin, sqrt, simplify, expand, re, im, conjugate

sig = [Matrix([[0, 1], [1, 0]]), Matrix([[0, -I], [I, 0]]), Matrix([[1, 0], [0, -1]])]
Tm = [s / 2 for s in sig]


def real_trace(M):
    t = M.trace()
    return sp.expand((t + sp.conjugate(t)) / 2)


# -------------------------------------------------------------- check 1
p = symbols('p1:9', real=True)
U = Matrix([[p[0] + I * p[1], p[2] + I * p[3]], [p[4] + I * p[5], p[6] + I * p[7]]])
x = symbols('x1:4', real=True)
y = symbols('y1:4', real=True)
eps = symbols('epsilon', real=True)
rho = symbols('rho_x rho_y', positive=True)


def su2_exp(w, r, sign):
    A = sum((w[a] * sig[a] for a in range(3)), sp.zeros(2)) / 2
    return cos(eps * r) * sp.eye(2) + sign * I * sin(eps * r) / r * A


gx = su2_exp(x, rho[0], +1)
gy_dag = su2_exp(y, rho[1], -1)            # exp(-i w_y.T) = (exp(i w_y.T))^+
F = -real_trace(gx * U * gy_dag)
# second-order Taylor coefficient in epsilon, with rho -> |w|/2 afterwards
F2 = sp.series(F, eps, 0, 3).removeO().coeff(eps, 2)
F2 = F2.subs({rho[0]: sqrt(sum(xi ** 2 for xi in x)) / 2, rho[1]: sqrt(sum(yi ** 2 for yi in y)) / 2})

sym = lambda a, b: (Tm[a] * Tm[b] + Tm[b] * Tm[a]) / 2
Mxx = Matrix(3, 3, lambda a, b: real_trace(sym(a, b) * U))
Myy = Matrix(3, 3, lambda a, b: real_trace(U * sym(a, b)))
Mxy = Matrix(3, 3, lambda a, b: -real_trace(Tm[a] * U * Tm[b]))
X, Y = Matrix(x), Matrix(y)
quad = Rational(1, 2) * ((X.T * Mxx * X)[0] + (Y.T * Myy * Y)[0] + 2 * (X.T * Mxy * Y)[0])
diff1 = simplify(expand(F2 - quad))
print("check 1 (Hessian formula, exact SU(2)):", "difference =", diff1)
assert diff1 == 0

# the first-order term must equal the gradient formula  Im Tr(T_a U) for x, -Im Tr(T_a U) for y
F1 = sp.series(F, eps, 0, 3).removeO().coeff(eps, 1)
F1 = F1.subs({rho[0]: sqrt(sum(xi ** 2 for xi in x)) / 2, rho[1]: sqrt(sum(yi ** 2 for yi in y)) / 2})
grad_x = [sp.im(sp.expand((Tm[a] * U).trace())) for a in range(3)]
grad_y = [-g for g in grad_x]
lin = sum(grad_x[a] * x[a] + grad_y[a] * y[a] for a in range(3))
diff0 = simplify(expand(F1 - lin))
print("check 1b (gradient formula, exact SU(2)):", "difference =", diff0)
assert diff0 == 0

# -------------------------------------------------------------- check 2
X4 = symbols('x0:4', real=True)
fa = [sp.Function(f'xi{a}')(*X4) for a in range(3)]
Aa = [[sp.Function(f'A{m}_{a}')(*X4) for a in range(3)] for m in range(4)]
xi = sum((fa[a] * (-I) * sig[a] / 2 for a in range(3)), sp.zeros(2))        # anti-Hermitian
Am = [sum((Aa[m][a] * (-I) * sig[a] / 2 for a in range(3)), sp.zeros(2)) for m in range(4)]
comm = lambda P, Q: P * Q - Q * P
D = lambda m, Z: Z.diff(X4[m]) + comm(Am[m], Z)
lhs = -sum((D_m for D_m in [(D(m, xi)).diff(X4[m]) for m in range(4)]), sp.zeros(2))
Delta = -sum((D(m, D(m, xi)) for m in range(4)), sp.zeros(2))
rhs = Delta + sum((comm(Am[m], D(m, xi)) for m in range(4)), sp.zeros(2))
d2 = (lhs - rhs).applyfunc(lambda e: sp.simplify(sp.expand(e)))
print("check 2 (M_A = Delta_A + [A_mu, D_mu .]):", "difference matrix =", d2.tolist())
assert d2 == sp.zeros(2)
print("ALL SYMBOLIC CHECKS PASSED")
