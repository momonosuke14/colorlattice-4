"""Production run: thermalized SU(3) ensembles on a 2^4 lattice.

For every beta:
  1. thermalize with the heat bath (hot start), check against a cold start;
  2. save a configuration every ``SEP`` sweeps, estimate the autocorrelation
     time of the plaquette;
  3. gauge-fix each configuration from several random gauge starts (Gribov
     copies), keep the copy with the lowest F, record the FP spectrum;
  4. along a short Wilson flow build H (naive and gauge-transported) and
     check the two-sided bounds of Proposition 2.

Output: results.npz and results.json
"""
import json
import time
import numpy as np
from colorlattice4 import *
from colorlattice4 import _I3

SEED = 20261112
rng = np.random.default_rng(SEED)
lat = Lattice(2)
Q = const_complement(lat.V)
BETAS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0]
NTHERM, SEP, NCFG, NCOPIES = 400, 10, 60, 4
FLOW = [(0.002, 2)]                    # same (T, n) as the paper
t0 = time.time()
log = {}


def tau_int(x, c=6.0):
    """Integrated autocorrelation time (Madras-Sokal windowing)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0]
    tau = 0.5
    for t in range(1, n):
        tau += ac[t]
        if t >= c * tau:
            break
    return float(tau)


def blocked(x, nb=20):
    x = np.asarray(x)
    m = len(x) // nb
    b = x[:m * nb].reshape(nb, m).mean(1)
    return float(b.mean()), float(b.std(ddof=1) / np.sqrt(nb))


# ---- exact check: cold configuration -------------------------------------
Uc = cold_start(lat)
Uc_f, _, ic = landau_fix(lat, Uc)
ev_cold = np.linalg.eigvalsh(Q.T @ fp_hessian(lat, Uc_f) @ Q)
vals, counts = np.unique(np.round(ev_cold, 8), return_counts=True)
log['cold_limit'] = dict(values=vals.tolist(), multiplicities=counts.tolist(),
                         expected={'2': 32, '4': 48, '6': 32, '8': 8})
print("cold-limit spectrum (should be 2x32, 4x48, 6x32, 8x8):", dict(zip(vals, counts)), flush=True)

# ---- thermalization check: hot vs cold start ------------------------------
check = {}
for b in (3.0, 6.0):
    U1 = hot_start(lat, rng)
    U2 = cold_start(lat)
    for _ in range(NTHERM):
        heatbath_sweep(lat, U1, b, rng)
        heatbath_sweep(lat, U2, b, rng)
    p1, p2 = [], []
    for _ in range(2000):
        heatbath_sweep(lat, U1, b, rng); p1.append(mean_plaquette(lat, U1))
        heatbath_sweep(lat, U2, b, rng); p2.append(mean_plaquette(lat, U2))
    (m1, e1), (m2, e2) = blocked(p1), blocked(p2)
    check[str(b)] = dict(hot=[m1, e1], cold=[m2, e2], sigma=abs(m1 - m2) / np.hypot(e1, e2))
    print(f"beta={b}: <plaq> from hot start {m1:.5f}+-{e1:.5f}, from cold start {m2:.5f}+-{e2:.5f} "
          f"({check[str(b)]['sigma']:.2f} sigma)", flush=True)
log['hot_vs_cold'] = check

# ---- ensembles ---------------------------------------------------------------
store = dict(betas=np.array(BETAS))
summary = []
for b in BETAS:
    U = hot_start(lat, rng)
    for _ in range(NTHERM):
        heatbath_sweep(lat, U, b, rng)
    plaq_series, cfgs = [], []
    for k in range(NCFG * SEP):
        heatbath_sweep(lat, U, b, rng)
        plaq_series.append(mean_plaquette(lat, U))
        if (k + 1) % SEP == 0:
            cfgs.append(U.copy())
    tau = tau_int(plaq_series)
    pm, pe = blocked(plaq_series)
    spec, l1_best, l1_spread, Fspread, ncopies_distinct = [], [], [], [], []
    Hn_ev, Hc_ev, gaps, dH, bounds_ok, rots, lstars, LMs = [], [], [], [], [], [], [], []
    for U in cfgs:
        bestF, bestU, bestev = None, None, None
        l1s, Fs = [], []
        for c in range(NCOPIES):
            gtr = random_su3(rng, lat.V)
            Uf, g, info = landau_fix(lat, gauge_transform(lat, U, gtr))
            assert info['converged'], (b, info)
            ev = np.linalg.eigvalsh(Q.T @ fp_hessian(lat, Uf) @ Q)
            assert ev[0] > 0
            l1s.append(ev[0]); Fs.append(info['F'])
            if bestF is None or info['F'] < bestF:
                bestF, bestU, bestev = info['F'], Uf, ev
        spec.append(bestev); l1_best.append(bestev[0])
        l1_spread.append(max(l1s) - min(l1s)); Fspread.append(max(Fs) - min(Fs))
        ncopies_distinct.append(len(np.unique(np.round(Fs, 6))))
        for (Tt, n) in FLOW:
            o = integrated_operator(lat, bestU, Tt, n)
            Ms = o['M_red']
            ls = min(np.linalg.eigvalsh(m)[0] for m in Ms)
            LM = max(np.linalg.eigvalsh(m)[-1] for m in Ms)
            hn, hc = np.linalg.eigvalsh(o['H_naive']), np.linalg.eigvalsh(o['H_cov'])
            Hn_ev.append(hn); Hc_ev.append(hc)
            bounds_ok.append(bool(hn[0] >= Tt / LM * (1 - 1e-12) and hn[-1] <= Tt / ls * (1 + 1e-12)))
            gaps.append(np.linalg.norm(o['H_naive'] - Tt * np.linalg.inv(Ms[0]), 2) / np.linalg.norm(o['H_naive'], 2))
            dH.append(np.linalg.norm(o['H_cov'] - o['H_naive'], 2) / np.linalg.norm(o['H_naive'], 2))
            rots.append(max(o['rotations'][1:])); lstars.append(ls); LMs.append(LM)
    store[f'plaq_{b}'] = np.array(plaq_series)
    store[f'spec_{b}'] = np.array(spec)
    store[f'Hn_{b}'] = np.array(Hn_ev)
    store[f'Hc_{b}'] = np.array(Hc_ev)
    row = dict(beta=b, plaq=pm, plaq_err=pe, tau_int=tau,
               lam1_mean=float(np.mean(l1_best)), lam1_std=float(np.std(l1_best, ddof=1)),
               lam1_min=float(np.min(l1_best)),
               copies_mean_distinct=float(np.mean(ncopies_distinct)),
               copies_lam1_spread_mean=float(np.mean(l1_spread)),
               bounds_ok=bool(all(bounds_ok)), cor1_gap_mean=float(np.mean(gaps)),
               transport_rel_change_mean=float(np.mean(dH)), transport_rel_change_max=float(np.max(dH)),
               rotation_max=float(np.max(rots)), lam_max_mean=float(np.mean(LMs)))
    summary.append(row)
    log['summary'] = summary
    log['settings'] = dict(seed=SEED, L=2, ntherm=NTHERM, sep=SEP, ncfg=NCFG, ncopies=NCOPIES, flow=FLOW)
    np.savez('results.npz', **store)
    json.dump(log, open('results.json', 'w'), indent=1)
    print(f"beta={b:5.1f} plaq={pm:.4f}+-{pe:.4f} tau={tau:5.1f}  lam1={row['lam1_mean']:.4f}+-{row['lam1_std']:.4f} "
          f"(min {row['lam1_min']:.4f})  copies/cfg={row['copies_mean_distinct']:.2f}  bounds_ok={row['bounds_ok']} "
          f"gap={row['cor1_gap_mean']:.2e} transport={row['transport_rel_change_mean']:.1e}  [{time.time()-t0:.0f}s]", flush=True)

log['summary'] = summary
log['settings'] = dict(seed=SEED, L=2, ntherm=NTHERM, sep=SEP, ncfg=NCFG, ncopies=NCOPIES, flow=FLOW)
np.savez('results.npz', **store)
json.dump(log, open('results.json', 'w'), indent=1)
print("done", time.time() - t0, "s", flush=True)
