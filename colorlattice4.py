r"""
ColorLattice 4
==============

Pure SU(3) lattice gauge theory on a periodic hypercubic lattice, built to
study the Faddeev-Popov (FP) operator in minimal Landau gauge on
*thermalized* configurations.

What is in here
---------------
* ``Lattice``            -- geometry, plaquette list, staples.
* ``heatbath_sweep``     -- Cabibbo-Marinari heat bath with Kennedy-Pendleton
                            SU(2) sampling (the thermal bath).
* ``metropolis_sweep``   -- an independent Metropolis update, used only to
                            cross-check the heat bath.
* ``landau_fix``         -- minimal Landau gauge: cooling + Newton polish.
* ``fp_hessian``         -- exact Hessian of the Landau functional (eq. (8)
                            of the paper) and its reduction to the space
                            without constant modes.
* ``wilson_flow``        -- Luescher's third-order Runge-Kutta gradient flow.
* ``integrated_operator``-- the operator H = int M^{-1} dt, with and without
                            gauge transport between frames.

Conventions
-----------
``T[a]`` are the Gell-Mann matrices over two, so ``Tr(T_a T_b) = delta_ab/2``.
Links are stored as ``U[x, mu]`` (3x3 complex), ``x`` a flat site index.
The Wilson action is ``S = beta * sum_p (1 - Re Tr P / 3)``.

Everything is double precision.  No statement here concerns the continuum
theory; this is a finite-lattice diagnostic tool.

Examples
--------
>>> import numpy as np
>>> np.allclose(np.einsum('aij,bji->ab', T, T), np.eye(8) / 2)
True
>>> lat = Lattice(2)
>>> (lat.V, lat.nlinks, len(lat.plaq))
(16, 64, 96)
"""
from __future__ import annotations

import itertools
import numpy as np

__all__ = [
    "T", "Lattice", "random_su3", "reunitarize", "su3_exp", "cold_start",
    "hot_start", "wilson_action", "mean_plaquette", "heatbath_sweep",
    "metropolis_sweep", "landau_functional", "landau_gradient",
    "landau_fix", "fp_hessian", "const_complement", "wilson_flow_step",
    "wilson_flow", "adjoint_rotation", "integrated_operator", "gauge_transform",
]

# --------------------------------------------------------------------------
# Group and algebra
# --------------------------------------------------------------------------


def _gell_mann_over_two() -> np.ndarray:
    lam = np.zeros((8, 3, 3), dtype=complex)
    lam[0][0, 1] = lam[0][1, 0] = 1
    lam[1][0, 1] = -1j
    lam[1][1, 0] = 1j
    lam[2][0, 0] = 1
    lam[2][1, 1] = -1
    lam[3][0, 2] = lam[3][2, 0] = 1
    lam[4][0, 2] = -1j
    lam[4][2, 0] = 1j
    lam[5][1, 2] = lam[5][2, 1] = 1
    lam[6][1, 2] = -1j
    lam[6][2, 1] = 1j
    lam[7] = np.diag([1, 1, -2]) / np.sqrt(3)
    return lam / 2


T = _gell_mann_over_two()
_I3 = np.eye(3, dtype=complex)
_SUBGROUPS = ((0, 1), (0, 2), (1, 2))


def dag(a: np.ndarray) -> np.ndarray:
    """Hermitian conjugate over the last two axes."""
    return np.swapaxes(a.conj(), -1, -2)


def random_su3(rng: np.random.Generator, size=()) -> np.ndarray:
    """Haar-distributed SU(3) matrices.

    A QR decomposition of a complex Gaussian matrix gives Haar U(3); dividing
    by a cube root of the determinant commutes with left multiplication by
    SU(3), so the image measure is Haar on SU(3).

    >>> rng = np.random.default_rng(1)
    >>> u = random_su3(rng, (5,))
    >>> bool(np.allclose(u @ dag(u), np.eye(3)) and np.allclose(np.linalg.det(u), 1))
    True
    """
    shape = tuple(np.atleast_1d(size)) if size != () else ()
    z = rng.normal(size=shape + (3, 3)) + 1j * rng.normal(size=shape + (3, 3))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r, axis1=-2, axis2=-1)
    q = q * (d / np.abs(d))[..., None, :]
    det = np.linalg.det(q)
    return q / (det ** (1 / 3))[..., None, None]


def reunitarize(u: np.ndarray) -> np.ndarray:
    """Project near-SU(3) matrices back onto SU(3) (Gram-Schmidt + phase).

    >>> rng = np.random.default_rng(2)
    >>> u = random_su3(rng) + 1e-9
    >>> v = reunitarize(u)
    >>> bool(np.allclose(v @ dag(v), np.eye(3), atol=1e-14))
    True
    """
    u = np.array(u, dtype=complex)
    r0 = u[..., 0, :]
    r0 = r0 / np.linalg.norm(r0, axis=-1, keepdims=True)
    r1 = u[..., 1, :]
    r1 = r1 - np.sum(r0.conj() * r1, axis=-1, keepdims=True) * r0
    r1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
    r2 = np.conj(np.cross(r0, r1))
    return np.stack([r0, r1, r2], axis=-2)


def su3_exp(z: np.ndarray) -> np.ndarray:
    """exp(Z) for anti-Hermitian Z (batched), via the Hermitian matrix iZ.

    >>> z = 1j * T[2] * 0.7
    >>> bool(np.allclose(su3_exp(z), np.diag(np.exp(0.7j * np.array([.5, -.5, 0])))))
    True
    """
    w, v = np.linalg.eigh(1j * z)
    return (v * np.exp(-1j * w)[..., None, :]) @ dag(v)


def herm_exp(h: np.ndarray, sign: float = 1.0) -> np.ndarray:
    """exp(sign * i * H) for Hermitian H (batched)."""
    w, v = np.linalg.eigh(h)
    return (v * np.exp(1j * sign * w)[..., None, :]) @ dag(v)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


class Lattice:
    """Periodic hypercubic lattice with ``L**4`` sites.

    Attributes
    ----------
    fwd, bwd : (V, 4) int arrays
        Neighbor of ``x`` in direction ``+mu`` and ``-mu``.
    plaq : list of 4-tuples ``((site, dir, dagger), ...)``
        P(x, mu, nu) = U_mu(x) U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+.
    staples : ``staples[x][mu]`` -> list of (factors, dagger_result)
        so that  Re Tr P = Re Tr (U_{x,mu} * staple).

    Notes
    -----
    For L = 2 the "forward" and "backward" neighbors coincide as sites, but
    the links are still distinct labelled variables; every plaquette contains
    four distinct links, which ``check_distinct_links`` verifies.
    """

    def __init__(self, L: int = 2):
        if L < 2:
            raise ValueError("L must be >= 2")
        self.L = L
        self.V = L ** 4
        self.nlinks = 4 * self.V
        self.coords = np.array(list(itertools.product(range(L), repeat=4)))

        def idx(c):
            return ((c[..., 0] * L + c[..., 1]) * L + c[..., 2]) * L + c[..., 3]

        self.fwd = np.zeros((self.V, 4), dtype=int)
        self.bwd = np.zeros((self.V, 4), dtype=int)
        for mu in range(4):
            s = self.coords.copy()
            s[:, mu] = (s[:, mu] + 1) % L
            self.fwd[:, mu] = idx(s)
            s = self.coords.copy()
            s[:, mu] = (s[:, mu] - 1) % L
            self.bwd[:, mu] = idx(s)

        plaq = []
        for x in range(self.V):
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    plaq.append((
                        (x, mu, False),
                        (self.fwd[x, mu], nu, False),
                        (self.fwd[x, nu], mu, True),
                        (x, nu, True),
                    ))
        self.plaq = plaq
        self.check_distinct_links()

        self._ps = np.array([[f[0] for f in p] for p in plaq])
        self._pd = np.array([[f[1] for f in p] for p in plaq])
        self._pg = np.array([[f[2] for f in p] for p in plaq])

        self.staples = [[[] for _ in range(4)] for _ in range(self.V)]
        for p in plaq:
            for k in range(4):
                s, d, g = p[k]
                others = tuple(p[(k + j) % 4] for j in (1, 2, 3))
                self.staples[s][d].append((others, g))

    def check_distinct_links(self) -> None:
        """Every plaquette must contain four different link variables."""
        for p in self.plaq:
            assert len({(f[0], f[1]) for f in p}) == 4

    # -- vectorised helpers ------------------------------------------------
    def plaquette_matrices(self, U: np.ndarray) -> np.ndarray:
        """All oriented plaquette matrices, shape (Np, 3, 3)."""
        m = U[self._ps, self._pd]
        m = np.where(self._pg[..., None, None], dag(m), m)
        return m[:, 0] @ m[:, 1] @ m[:, 2] @ m[:, 3]

    def staple_sum(self, U: np.ndarray, x: int, mu: int) -> np.ndarray:
        """Sum of staples for one link."""
        S = np.zeros((3, 3), dtype=complex)
        for others, g in self.staples[x][mu]:
            B = _I3
            for (s, d, gg) in others:
                B = B @ (dag(U[s, d]) if gg else U[s, d])
            S += dag(B) if g else B
        return S

    def all_staples(self, U: np.ndarray) -> np.ndarray:
        """Staple sums for all links at once, shape (V, 4, 3, 3)."""
        Sig = np.zeros((self.V, 4, 3, 3), dtype=complex)
        m = U[self._ps, self._pd]
        m = np.where(self._pg[..., None, None], dag(m), m)
        for k in range(4):
            B = m[:, (k + 1) % 4] @ m[:, (k + 2) % 4] @ m[:, (k + 3) % 4]
            B = np.where(self._pg[:, k][:, None, None], dag(B), B)
            np.add.at(Sig, (self._ps[:, k], self._pd[:, k]), B)
        return Sig


# --------------------------------------------------------------------------
# Configurations, action
# --------------------------------------------------------------------------


def cold_start(lat: Lattice) -> np.ndarray:
    return np.tile(_I3, (lat.V, 4, 1, 1))


def hot_start(lat: Lattice, rng: np.random.Generator) -> np.ndarray:
    return random_su3(rng, (lat.V, 4))


def mean_plaquette(lat: Lattice, U: np.ndarray) -> float:
    """<(1/3) Re Tr P> over all plaquettes.

    >>> lat = Lattice(2)
    >>> mean_plaquette(lat, cold_start(lat))
    1.0
    """
    tr = np.trace(lat.plaquette_matrices(U), axis1=-2, axis2=-1).real
    return float(tr.mean() / 3)


def wilson_action(lat: Lattice, U: np.ndarray, beta: float) -> float:
    """S = beta * sum_p (1 - Re Tr P / 3).

    >>> lat = Lattice(2); wilson_action(lat, cold_start(lat), 6.0)
    0.0
    """
    return float(beta * len(lat.plaq) * (1.0 - mean_plaquette(lat, U)))


def gauge_transform(lat: Lattice, U: np.ndarray, g: np.ndarray) -> np.ndarray:
    """U'_mu(x) = g(x) U_mu(x) g(x+mu)^+."""
    out = np.empty_like(U)
    for mu in range(4):
        out[:, mu] = g @ U[:, mu] @ dag(g[lat.fwd[:, mu]])
    return out


# --------------------------------------------------------------------------
# Thermal bath: Cabibbo-Marinari heat bath, Kennedy-Pendleton SU(2)
# --------------------------------------------------------------------------


def _quaternion_part(X: np.ndarray):
    """Real-quaternion part (a0, a1, a2, a3) of a 2x2 complex matrix."""
    a, b, c, d = X[0, 0], X[0, 1], X[1, 0], X[1, 1]
    a0 = 0.5 * (a + d).real
    a3 = 0.5 * (a - d).imag
    a1 = 0.5 * (b + c).imag
    a2 = 0.5 * (b - c).real
    return a0, np.array([a1, a2, a3])


def _su2_from_quaternion(a0: float, a: np.ndarray) -> np.ndarray:
    return np.array([[a0 + 1j * a[2], a[1] + 1j * a[0]],
                     [-a[1] + 1j * a[0], a0 - 1j * a[2]]])


def _sample_a0(alpha: float, rng: np.random.Generator) -> float:
    """Sample a0 with density ~ sqrt(1-a0^2) exp(alpha a0) on [-1, 1].

    Kennedy-Pendleton for alpha >= 1, plain rejection from Haar below.
    """
    if alpha < 1.0:
        while True:
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            if rng.random() <= np.exp(alpha * (q[0] - 1.0)):
                return float(q[0])
    while True:
        r1, r2, r3, r4 = 1.0 - rng.random(4)
        delta = -(np.log(r1) + np.cos(2 * np.pi * r2) ** 2 * np.log(r3)) / alpha
        if r4 ** 2 <= 1.0 - 0.5 * delta:
            return 1.0 - delta


def _heatbath_link(U: np.ndarray, Sig: np.ndarray, beta: float,
                   rng: np.random.Generator) -> np.ndarray:
    """One Cabibbo-Marinari update of a single link given its staple sum.

    The conditional density is proportional to exp((beta/3) Re Tr(U Sig)).
    """
    for (i, j) in _SUBGROUPS:
        W = U @ Sig
        X = W[np.ix_((i, j), (i, j))]
        a0, a = _quaternion_part(X)
        k = np.sqrt(a0 * a0 + a @ a)
        if k < 1e-12:
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            r = _su2_from_quaternion(q[0], q[1:])
        else:
            Vinv = _su2_from_quaternion(a0 / k, -a / k)  # V^+
            b0 = _sample_a0(2.0 * beta * k / 3.0, rng)
            n = rng.normal(size=3)
            n /= np.linalg.norm(n)
            b = np.sqrt(max(0.0, 1.0 - b0 * b0)) * n
            r = _su2_from_quaternion(b0, b) @ Vinv
        R = _I3.copy()
        R[np.ix_((i, j), (i, j))] = r
        U = R @ U
    return U


def heatbath_sweep(lat: Lattice, U: np.ndarray, beta: float,
                   rng: np.random.Generator) -> None:
    """One full heat-bath sweep over all links (in place), then reunitarize."""
    for x in range(lat.V):
        for mu in range(4):
            Sig = lat.staple_sum(U, x, mu)
            U[x, mu] = _heatbath_link(U[x, mu], Sig, beta, rng)
    U[:] = reunitarize(U)


def metropolis_sweep(lat: Lattice, U: np.ndarray, beta: float,
                     rng: np.random.Generator, eps: float = 0.35,
                     hits: int = 3) -> float:
    """Independent Metropolis sweep used as a cross-check; returns acceptance.

    The proposal U -> exp(i eps c.T) U with c uniform in a cube is symmetric
    (c -> -c is the inverse move), so plain Metropolis is exact.
    """
    acc = tot = 0
    for x in range(lat.V):
        for mu in range(4):
            Sig = lat.staple_sum(U, x, mu)
            for _ in range(hits):
                c = rng.uniform(-1, 1, size=8)
                Uh = herm_exp(np.einsum('a,aij->ij', c * eps, T))
                new = Uh @ U[x, mu]
                dS = -(beta / 3.0) * (np.trace(new @ Sig) - np.trace(U[x, mu] @ Sig)).real
                tot += 1
                if dS <= 0 or rng.random() < np.exp(-dS):
                    U[x, mu] = new
                    acc += 1
    U[:] = reunitarize(U)
    return acc / tot


# --------------------------------------------------------------------------
# Landau gauge
# --------------------------------------------------------------------------


def landau_functional(lat: Lattice, U: np.ndarray) -> float:
    """F = - sum_{x,mu} Re Tr U_mu(x)  (U already in the gauge frame)."""
    return float(-np.trace(U, axis1=-2, axis2=-1).real.sum())


def landau_gradient(lat: Lattice, U: np.ndarray) -> np.ndarray:
    """dF/d omega_x^a at omega = 0, shape (V, 8).

    grad_x^a = sum_mu Im Tr( T_a [U_mu(x) - U_mu(x-mu)] ).
    """
    out = np.zeros((lat.V, 8))
    for mu in range(4):
        D = U[:, mu] - U[lat.bwd[:, mu], mu]
        out += np.einsum('aij,xji->xa', T, D).imag
    return out


def _apply_frame(lat: Lattice, U: np.ndarray, h: np.ndarray) -> np.ndarray:
    return gauge_transform(lat, U, h)


def const_complement(V: int) -> np.ndarray:
    """Orthonormal basis (128 x 120 for V=16) of the complement of constants."""
    n = V * 8
    C = np.zeros((n, 8))
    for a in range(8):
        C[a::8, a] = 1.0 / np.sqrt(V)
    P = np.eye(n) - C @ C.T
    w, v = np.linalg.eigh(P)
    return v[:, w > 0.5]


def fp_hessian(lat: Lattice, U: np.ndarray) -> np.ndarray:
    """Exact Hessian of F[exp(i omega.T)] at omega = 0 (eq. (8) of the paper).

    For a link (x -> y) with U = U_mu(x) the second-order expansion is
        F2 = 1/2 ReTr(A_x^2 U) + 1/2 ReTr(U A_y^2) - ReTr(A_x U A_y),
    with A = omega^a T_a.  The matrix is returned on the full (8V)x(8V) space,
    index = 8*x + a.
    """
    n = 8 * lat.V
    M = np.zeros((n, n))
    sym = 0.5 * (T[:, None] @ T[None, :] + T[None, :] @ T[:, None])  # {Ta,Tb}/2
    for mu in range(4):
        for x in range(lat.V):
            y = lat.fwd[x, mu]
            Ul = U[x, mu]
            dxx = np.einsum('abij,ji->ab', sym, Ul).real      # Re Tr({Ta,Tb}/2 U)
            dyy = dxx                                         # Re Tr(U {Ta,Tb}/2): same by cyclicity
            dxy = -np.einsum('aij,jk,bki->ab', T, Ul, T).real  # -Re Tr(Ta U Tb)
            xs, ys = slice(8 * x, 8 * x + 8), slice(8 * y, 8 * y + 8)
            M[xs, xs] += dxx
            M[ys, ys] += dyy
            M[xs, ys] += dxy
            M[ys, xs] += dxy.T
    return M


def _newton_polish(lat, U, g, tol, max_iter=30):
    Q = const_complement(lat.V)
    for _ in range(max_iter):
        grad = landau_gradient(lat, U).ravel()
        gn = np.linalg.norm(grad)
        if gn < tol:
            return U, g, gn, True
        Mr = Q.T @ fp_hessian(lat, U) @ Q
        w = np.linalg.eigvalsh(Mr)
        if w[0] <= 1e-10:
            return U, g, gn, False
        om = -(Q @ np.linalg.solve(Mr, Q.T @ grad)).reshape(lat.V, 8)
        F0 = landau_functional(lat, U)
        step = 1.0
        ok = False
        for _ in range(30):
            h = herm_exp(np.einsum('xa,aij->xij', om * step, T))
            Un = gauge_transform(lat, U, h)
            if landau_functional(lat, Un) <= F0 + 1e-14 and \
                    np.linalg.norm(landau_gradient(lat, Un)) < gn * (1 + 1e-9):
                ok = True
                break
            step *= 0.5
        if not ok:
            return U, g, gn, False
        U, g = Un, h @ g
    gn = np.linalg.norm(landau_gradient(lat, U))
    return U, g, gn, gn < tol


def landau_fix(lat: Lattice, U: np.ndarray, tol: float = 1e-11,
               max_sweeps: int = 20000):
    """Minimal-Landau-gauge fixing of one configuration.

    Stage 1 is Cabibbo-Marinari *cooling* of F (each local SU(2) maximisation
    is exact, so F decreases monotonically).  Stage 2 is a damped Newton
    iteration that uses the exact Hessian; it converges quadratically once
    the Hessian is positive definite.

    Returns ``(U_fixed, g, info)`` with ``U_fixed = g U g^+`` and ``info`` a
    dict with the final gradient norm, the functional, and ``converged``.
    """
    U = U.copy()
    g = np.tile(_I3, (lat.V, 1, 1))
    gn = np.inf
    converged = False
    for sweep in range(max_sweeps):
        for x in range(lat.V):
            W = np.zeros((3, 3), dtype=complex)
            for mu in range(4):
                W += U[x, mu] + dag(U[lat.bwd[x, mu], mu])
            h = _I3
            for (i, j) in _SUBGROUPS:
                X = (h @ W)[np.ix_((i, j), (i, j))]
                a0, a = _quaternion_part(X)
                k = np.sqrt(a0 * a0 + a @ a)
                if k < 1e-14:
                    continue
                r = _su2_from_quaternion(a0 / k, -a / k)
                R = _I3.copy()
                R[np.ix_((i, j), (i, j))] = r
                h = R @ h
            for mu in range(4):
                U[x, mu] = h @ U[x, mu]
                y = lat.bwd[x, mu]
                U[y, mu] = U[y, mu] @ dag(h)
            g[x] = h @ g[x]
        if sweep % 10 == 9:
            gn = np.linalg.norm(landau_gradient(lat, U))
            if gn < 1e-3:
                U2, g2, gn2, ok = _newton_polish(lat, U, g, tol)
                if ok:
                    U, g, gn, converged = U2, g2, gn2, True
                    break
    U = reunitarize(U)
    g = reunitarize(g)
    gn = np.linalg.norm(landau_gradient(lat, U))
    info = dict(grad_norm=float(gn), F=landau_functional(lat, U),
                converged=bool(converged and gn < tol * 10), sweeps=sweep + 1)
    return U, g, info


# --------------------------------------------------------------------------
# Wilson (gradient) flow: Luescher, JHEP 08 (2010) 071, appendix C
# --------------------------------------------------------------------------


def _ta_project(Y: np.ndarray) -> np.ndarray:
    """Traceless anti-Hermitian part of Y: (Y-Y^+)/2 - Tr(Y-Y^+)/6."""
    A = 0.5 * (Y - dag(Y))
    tr = np.trace(A, axis1=-2, axis2=-1)
    return A - (tr / 3.0)[..., None, None] * _I3


def _flow_Z(lat: Lattice, U: np.ndarray) -> np.ndarray:
    """Z(U) = -P_ta(U Sigma) so that dU/dt = Z U (with g0^2 = 6/beta)."""
    return -_ta_project(U @ lat.all_staples(U))


def wilson_flow_step(lat: Lattice, U: np.ndarray, eps: float) -> np.ndarray:
    """One RK3 step of size eps for dU/dt = Z(U) U."""
    Z0 = eps * _flow_Z(lat, U)
    W1 = su3_exp(0.25 * Z0) @ U
    Z1 = eps * _flow_Z(lat, W1)
    W2 = su3_exp(8.0 / 9.0 * Z1 - 17.0 / 36.0 * Z0) @ W1
    Z2 = eps * _flow_Z(lat, W2)
    return su3_exp(0.75 * Z2 - 8.0 / 9.0 * Z1 + 17.0 / 36.0 * Z0) @ W2


def wilson_flow(lat: Lattice, U: np.ndarray, t: float, nsteps: int) -> np.ndarray:
    """Flow to time t in nsteps RK3 steps (returns a new array)."""
    V = U.copy()
    for _ in range(nsteps):
        V = reunitarize(wilson_flow_step(lat, V, t / nsteps))
    return V


# --------------------------------------------------------------------------
# Integrated operator H, with gauge transport
# --------------------------------------------------------------------------


def adjoint_rotation(h: np.ndarray) -> np.ndarray:
    """Block-diagonal Ad(h): (xa),(xb) -> 2 Tr(T_a h(x) T_b h(x)^+)."""
    V = h.shape[0]
    R = np.zeros((8 * V, 8 * V))
    for x in range(V):
        blk = 2.0 * np.einsum('aij,jk,bkl,li->ab', T, h[x], T, dag(h[x])).real
        R[8 * x:8 * x + 8, 8 * x:8 * x + 8] = blk
    return R


def _pinv_const_free(M: np.ndarray, V: int):
    Q = const_complement(V)
    Mr = Q.T @ M @ Q
    return Q @ np.linalg.inv(Mr) @ Q.T


def integrated_operator(lat: Lattice, U0: np.ndarray, T_total: float, n: int,
                        flow_steps_per_interval: int = 4, tol: float = 1e-11):
    """H = sum_i dt/2 (M_i^{-1} + M_{i+1}^{-1}) along a short Wilson flow.

    The configuration at t_i is re-fixed to Landau gauge, starting from the
    flowed field (so the gauge rotation between frames is small).  Two
    versions of H are returned on the 120-dimensional space of frame 0:

    ``H_naive``  adds the matrices as they are (what the paper did);
    ``H_cov``    first transports each inverse to frame 0 with the cumulative
                 adjoint rotation, then compresses to the frame-0 space.

    Returns a dict with both, the list of reduced spectra and the measured
    gauge rotations ``max_x ||g(x)-1||_F`` per step.
    """
    dt = T_total / n
    Q0 = const_complement(lat.V)
    frames = [U0]
    hs_cum = [np.tile(_I3, (lat.V, 1, 1))]
    rot = [0.0]
    cur = U0
    cum = hs_cum[0]
    for _ in range(n):
        fl = wilson_flow(lat, cur, dt, flow_steps_per_interval)
        fixed, h, info = landau_fix(lat, fl, tol=tol)
        if not info['converged']:
            raise RuntimeError("gauge fixing did not converge along the flow")
        cum = h @ cum
        frames.append(fixed)
        hs_cum.append(cum.copy())
        rot.append(float(np.max(np.linalg.norm(h - _I3, axis=(-2, -1)))))
        cur = fixed
    Ms = [fp_hessian(lat, f) for f in frames]
    Minv = [_pinv_const_free(M, lat.V) for M in Ms]
    w = np.full(n + 1, dt)
    w[0] = w[-1] = dt / 2.0
    Hn = sum(wi * Mi for wi, Mi in zip(w, Minv))
    Hc = 0
    for wi, Mi, hc in zip(w, Minv, hs_cum):
        R = adjoint_rotation(hc)
        Hc = Hc + wi * (R.T @ Mi @ R)
    out = dict(
        H_naive=Q0.T @ Hn @ Q0,
        H_cov=Q0.T @ ((Hc + Hc.T) / 2) @ Q0,
        M_red=[Q0.T @ M @ Q0 for M in Ms],
        rotations=rot, frames=frames, T=T_total, n=n,
    )
    return out
