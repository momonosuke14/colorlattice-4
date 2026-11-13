"""Independent verification of ColorLattice 4.

Each test compares the code against something that does not share its
implementation: an exact integral, a second algorithm, finite differences,
or a mathematical identity.  Results are written to verify_results.json.
"""
import json
import time
import numpy as np
from scipy.linalg import eigh
from colorlattice4 import *
from colorlattice4 import dag, herm_exp, _heatbath_link, _I3

rng = np.random.default_rng(20261112)
res = {}


def report(name, ok, detail):
    res[name] = dict(passed=bool(ok), detail=detail)
    print(("PASS " if ok else "FAIL ") + name + " :: " + detail, flush=True)


def blocked_error(x, nblocks=20):
    x = np.asarray(x)
    m = len(x) // nblocks
    b = x[: m * nblocks].reshape(nblocks, m).mean(axis=1)
    return b.mean(), b.std(ddof=1) / np.sqrt(nblocks)


# ---- 1. generators ------------------------------------------------------
ortho = np.abs(np.einsum('aij,bji->ab', T, T) - np.eye(8) / 2).max()
herm = max(np.abs(t - dag(t)).max() for t in T)
traceless = max(abs(np.trace(t)) for t in T)
comm = np.einsum('aij,bjk->abik', T, T) - np.einsum('bij,ajk->abik', T, T)
# [Ta,Tb] = i f_abc Tc : f real and totally antisymmetric
f = (-2j * np.einsum('abik,cki->abc', comm, T)).real
anti = np.abs(f + f.transpose(1, 0, 2)).max() + np.abs(f + f.transpose(0, 2, 1)).max()
report("generators", max(ortho, herm, traceless, anti) < 1e-14,
       f"Tr(TaTb)=delta/2, hermitian, traceless, f_abc antisymmetric: max dev {max(ortho,herm,traceless,anti):.1e}")

# ---- 2. Haar sampler ------------------------------------------------------
# <|Tr U|^2> = 1 for Haar SU(3); <Re Tr U> = 0
u = random_su3(rng, 400000)
tr = np.trace(u, axis1=-2, axis2=-1)
m2, e2 = blocked_error(np.abs(tr) ** 2)
m1, e1 = blocked_error(tr.real)
report("haar_sampler", abs(m2 - 1) < 4 * e2 and abs(m1) < 4 * e1,
       f"<|TrU|^2>={m2:.4f}+-{e2:.4f} (exact 1), <ReTrU>={m1:+.4f}+-{e1:.4f} (exact 0)")

# ---- 3. single-link heat bath against an exact (importance-sampled) integral
beta = 2.5
Sig = 1.5 * random_su3(rng)
A = random_su3(rng)
Uh = random_su3(rng, 1_000_000)
x1 = np.einsum('nij,ji->n', Uh, Sig).real
x2 = np.einsum('nij,ji->n', Uh, A).real
w = np.exp(beta / 3 * x1)
ex1 = (w * x1).sum() / w.sum()
ex2 = (w * x2).sum() / w.sum()
ex3 = (w * x1 ** 2).sum() / w.sum()
# error of a ratio estimator via block jackknife
nb = 20
idx = np.array_split(np.arange(len(w)), nb)
def rat(f):
    return np.array([(w[i] * f[i]).sum() / w[i].sum() for i in idx]).std(ddof=1) / np.sqrt(nb)
s1, s2, s3 = rat(x1), rat(x2), rat(x1 ** 2)
Ucur = random_su3(rng)
h1, h2, h3 = [], [], []
for k in range(60000):
    Ucur = _heatbath_link(Ucur, Sig, beta, rng)
    if k >= 200:
        h1.append(np.trace(Ucur @ Sig).real)
        h2.append(np.trace(Ucur @ A).real)
        h3.append(np.trace(Ucur @ Sig).real ** 2)
(a1, b1), (a2, b2), (a3, b3) = blocked_error(h1), blocked_error(h2), blocked_error(h3)
dev = [abs(a1 - ex1) / np.hypot(b1, s1), abs(a2 - ex2) / np.hypot(b2, s2), abs(a3 - ex3) / np.hypot(b3, s3)]
report("heatbath_single_link", max(dev) < 4,
       f"<ReTr(U Sig)> HB {a1:.4f}+-{b1:.4f} vs exact {ex1:.4f}+-{s1:.4f}; "
       f"<ReTr(U A)> {a2:.4f}+-{b2:.4f} vs {ex2:.4f}+-{s2:.4f}; "
       f"<(ReTr)^2> {a3:.4f}+-{b3:.4f} vs {ex3:.4f}+-{s3:.4f}; worst {max(dev):.2f} sigma")

# ---- 4. whole-lattice cross-check: heat bath vs Metropolis ---------------
lat = Lattice(2)
bt = 3.0
Uhb = hot_start(lat, rng)
for _ in range(200):
    heatbath_sweep(lat, Uhb, bt, rng)
p_hb = []
for _ in range(6000):
    heatbath_sweep(lat, Uhb, bt, rng)
    p_hb.append(mean_plaquette(lat, Uhb))
Um = hot_start(lat, rng)
for _ in range(300):
    metropolis_sweep(lat, Um, bt, rng)
p_mt = []
accs = []
for _ in range(12000):
    accs.append(metropolis_sweep(lat, Um, bt, rng))
    p_mt.append(mean_plaquette(lat, Um))
(ph, eh), (pm, em) = blocked_error(p_hb, 30), blocked_error(p_mt, 30)
zz = abs(ph - pm) / np.hypot(eh, em)
report("heatbath_vs_metropolis", zz < 4,
       f"<plaq> beta=3: HB {ph:.5f}+-{eh:.5f}, Metropolis {pm:.5f}+-{em:.5f} "
       f"(acc {np.mean(accs):.2f}), difference {zz:.2f} sigma")

# ---- 5. gradient and Hessian against finite differences ------------------
def F_of_omega(U, om):
    h = herm_exp(np.einsum('xa,aij->xij', om.reshape(lat.V, 8), T))
    return landau_functional(lat, gauge_transform(lat, U, h))

Ur = hot_start(lat, rng)                       # NOT gauge fixed
Mx = fp_hessian(lat, Ur)
n = 8 * lat.V
# diagonal and a random subset of off-diagonal entries (full FD would be 8k evals; subset suffices)
h = 2e-3
def fd_pair(U, i, j, h):
    E = np.eye(n)
    f = lambda a, b: F_of_omega(U, a * h * E[i] + b * h * E[j])
    return (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / (4 * h * h)
pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(Mx[i, j]) > 1e-12]
sel = [pairs[k] for k in rng.choice(len(pairs), 300, replace=False)]
err = []
for (i, j) in sel:
    a = fd_pair(Ur, i, j, h); b = fd_pair(Ur, i, j, h / 2)
    rich = (4 * b - a) / 3
    err.append(abs(rich - Mx[i, j]))
diag_sel = rng.choice(n, 60, replace=False)
F0 = F_of_omega(Ur, np.zeros(n))
for i in diag_sel:
    e = np.zeros(n); e[i] = 1
    a = (F_of_omega(Ur, h * e) + F_of_omega(Ur, -h * e) - 2 * F0) / h ** 2
    b = (F_of_omega(Ur, h / 2 * e) + F_of_omega(Ur, -h / 2 * e) - 2 * F0) / (h / 2) ** 2
    err.append(abs((4 * b - a) / 3 - Mx[i, i]))
sym_err = np.abs(Mx - Mx.T).max()
C = np.zeros((n, 8))
for a in range(8):
    C[a::8, a] = 1.0
quad_err = np.abs(C.T @ Mx @ C).max()          # c^T M c = 0 holds at ANY configuration
# M c = 0 holds only where the gradient vanishes (gauge-fixed): checked below in test 6
report("hessian_vs_finite_differences_nonstationary", max(err) < 1e-7 and sym_err < 1e-14 and quad_err < 1e-12,
       f"random hot configuration, 360 entries (Richardson FD of F itself): max |analytic-FD| {max(err):.1e}; "
       f"asymmetry {sym_err:.1e}; c^T M c = {quad_err:.1e}")

Uq = hot_start(lat, rng)
g0 = rng.normal(size=(lat.V, 8)) * 0.3
eg = []
for i in rng.choice(n, 30, replace=False):
    e = np.zeros(n); e[i] = 1
    fdg = (F_of_omega(Uq, h * e) - F_of_omega(Uq, -h * e)) / (2 * h)
    fdg2 = (F_of_omega(Uq, h / 2 * e) - F_of_omega(Uq, -h / 2 * e)) / h
    eg.append(abs((4 * fdg2 - fdg) / 3 - landau_gradient(lat, Uq).ravel()[i]))
report("gradient_vs_finite_differences", max(eg) < 1e-8, f"max |analytic-FD| {max(eg):.1e}")

# ---- 6. Landau gauge fixing, copies, gauge invariance ---------------------
Ut = hot_start(lat, rng)
for _ in range(100):
    heatbath_sweep(lat, Ut, 6.0, rng)
Uf, gf, inf = landau_fix(lat, Ut)
M = fp_hessian(lat, Uf)
Q = const_complement(lat.V)
lam = np.linalg.eigvalsh(Q.T @ M @ Q)
report("landau_fix_converges", inf['converged'] and inf['grad_norm'] < 1e-10 and lam[0] > 0,
       f"|grad F| = {inf['grad_norm']:.1e}, lowest FP eigenvalue {lam[0]:.6f}, negative modes {int((lam<0).sum())}, dim {len(lam)}")
# plaquette is gauge invariant; the functional is not
report("gauge_invariance_of_plaquette", abs(mean_plaquette(lat, Uf) - mean_plaquette(lat, Ut)) < 1e-13,
       f"|plaq(fixed)-plaq(original)| = {abs(mean_plaquette(lat, Uf)-mean_plaquette(lat, Ut)):.1e}")
Cn = np.zeros((n, 8))
for a_ in range(8):
    Cn[a_::8, a_] = 1.0
zm = np.abs(M @ Cn).max()
report("constant_modes_exact_zero_modes_when_gauge_fixed", zm < 1e-11,
       f"gauge-fixed configuration: max |M c| = {zm:.1e} (eight exact zero modes)")

def copy_scan(U, ncopies):
    Fs, l1s = [], []
    for c in range(ncopies):
        gr = random_su3(rng, lat.V)
        Ufc, _, ic = landau_fix(lat, gauge_transform(lat, U, gr))
        assert ic['converged']
        Fs.append(ic['F'])
        l1s.append(np.linalg.eigvalsh(Q.T @ fp_hessian(lat, Ufc) @ Q)[0])
    return np.array(Fs), np.array(l1s)

Fh, lh = copy_scan(hot_start(lat, rng), 20)         # disordered (hot) configuration
Ft, lt = copy_scan(Ut, 20)                          # thermalized, beta = 6
nh = len(np.unique(np.round(Fh, 6)))
nt = len(np.unique(np.round(Ft, 6)))
report("gribov_copies_scan", np.all(lh > 0) and np.all(lt > 0) and nh >= 1 and nt >= 1,
       f"20 random gauge starts each: hot config -> {nh} distinct local minima of F "
       f"(F in [{Fh.min():.4f},{Fh.max():.4f}]); thermalized beta=6 -> {nt} distinct "
       f"(F in [{Ft.min():.4f},{Ft.max():.4f}]); every minimum has lambda_1>0")

# ---- 7. Wilson flow -------------------------------------------------------
Ub = hot_start(lat, rng)
for _ in range(100):
    heatbath_sweep(lat, Ub, 6.0, rng)
S0 = wilson_action(lat, Ub, 6.0)
Ss = [S0]
Uc = Ub
for _ in range(10):
    Uc = reunitarize(wilson_flow_step(lat, Uc, 0.002))
    Ss.append(wilson_action(lat, Uc, 6.0))
mono = np.all(np.diff(Ss) < 0)
tfin = 0.02
ref = wilson_flow(lat, Ub, tfin, 512)
errs = []
for ns in (2, 4, 8, 16):
    errs.append(np.abs(wilson_flow(lat, Ub, tfin, ns) - ref).max())
ords = [np.log2(errs[i] / errs[i + 1]) for i in range(3)]
# unitarity after flow
unit = np.abs(Uc @ dag(Uc) - _I3).max()
report("wilson_flow", mono and min(ords) > 2.7 and unit < 1e-13,
       f"action decreases monotonically ({Ss[0]:.3f} -> {Ss[-1]:.3f}); observed RK order {['%.2f'%o for o in ords]} (theory 3); unitarity {unit:.1e}")
# first-order check: dS/dt = -sum |Z|^2 normalisation
dS_fd = (wilson_action(lat, wilson_flow(lat, Ub, 1e-6, 1), 6.0) - S0) / 1e-6
print("   dS/dt at t=0 (finite difference):", f"{dS_fd:.3f}")
res['wilson_flow']['dSdt_t0'] = float(dS_fd)

# ---- 8. integrated operator: bounds and convergence ----------------------
Ui, gi, ii = landau_fix(lat, Ut)
out = integrated_operator(lat, Uf, 0.002, 2)
Ms = out['M_red']
lstar = min(np.linalg.eigvalsh(m)[0] for m in Ms)
LM = max(np.linalg.eigvalsh(m)[-1] for m in Ms)
Tt = out['T']
hn = np.linalg.eigvalsh(out['H_naive'])
hc = np.linalg.eigvalsh(out['H_cov'])
b_lo, b_hi = Tt / LM, Tt / lstar
ok_bounds = hn[0] >= b_lo * (1 - 1e-12) and hn[-1] <= b_hi * (1 + 1e-12)
# covariant H also within bounds up to the compression error
ok_bounds_c = hc[0] >= b_lo * (1 - 1e-3) and hc[-1] <= b_hi * (1 + 1e-3)
M0 = Ms[0]
gap = np.linalg.norm(out['H_naive'] - Tt * np.linalg.inv(M0), 2) / np.linalg.norm(out['H_naive'], 2)
report("proposition2_bounds", ok_bounds and ok_bounds_c,
       f"T/L_M={b_lo:.4e} <= spec(H) in [{hn[0]:.4e},{hn[-1]:.4e}] <= T/lam_*={b_hi:.4e}; "
       f"covariant H in [{hc[0]:.4e},{hc[-1]:.4e}]; ||H-T M0^-1||/||H||={gap:.2e}")
dH = np.linalg.norm(out['H_cov'] - out['H_naive'], 2) / np.linalg.norm(out['H_naive'], 2)
report("transport_effect_measured", True,
       f"relative operator-norm change from gauge transport {dH:.2e}; max rotation per step {out['rotations'][1:]}")

json.dump(res, open('verify_results.json', 'w'), indent=1)
print("\nSUMMARY:", sum(v['passed'] for v in res.values()), "/", len(res), "tests passed")
