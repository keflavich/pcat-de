"""Speed of the PCAT engines on a synthetic crowded field.

Draws a power-law population of stars with a Moffat ePSF on white noise,
starts every engine from the same perturbed truth catalogue and reports, per
engine: time per proposal, mean N after burn-in, and the effective sample size
of N and logL per CPU-second (one chain, so ESS values scatter by ~20-30%).

    python benchmarks/bench_synthetic.py [--size 96] [--nstar 400] [--out bench.json]

The engines:

* ``python``     PCATSampler (C kernel through ctypes, numpy region bookkeeping);
* ``numba``      FastPCATSampler, same target and proposals as ``python``;
* ``numba_st``   + flux-adaptive stamps (stamp_tol 0.05);
* ``numba_k1``   + one star moved per region per star move (move_nstar 1).
"""
import argparse
import json
import platform
import time

import numpy as np

from pcat_fast.fast_sampler import ess_geyer
from pcat_fast.mocks import TruthRenderer, make_mock, power_law_fluxes, psf_snr_factor
from pcat_fast.pcat_sampler import PCATConfig, run_pcat

ENGINES = {
    'python': dict(engine='python'),
    'numba': dict(engine='numba'),
    'numba_st': dict(engine='numba', stamp_tol=0.05),
    'numba_k1': dict(engine='numba', stamp_tol=0.05, move_nstar=1),
}


def moffat_epsf(oversample=5, size=205, fwhm=2.0, beta=2.5):
    """Pixel-integrated Moffat ePSF in the ePSF convention of :mod:`pcat_fast.psf`
    (centre at index size // 2, every ``oversample``-th node = flux fraction per
    detector pixel).  The power-law wings make the stamp truncation (stamp_tol)
    matter, as for JWST PSFs; a Gaussian has none."""
    from scipy.ndimage import uniform_filter
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    u = (np.arange(size) - size // 2) / oversample
    r2 = u[:, None] ** 2 + u[None, :] ** 2
    p = uniform_filter((1 + r2 / alpha ** 2) ** -beta, size=oversample, mode='constant')
    c = size // 2
    return p / p[c % oversample::oversample, c % oversample::oversample].sum()


def scene(size, nstar, sigma, seed):
    epsf, os_ = moffat_epsf(), 5
    rng = np.random.default_rng(seed)
    fmin = 3.0 * sigma / psf_snr_factor(epsf, os_)   # S/N 3 for a known position
    x = rng.uniform(-0.5, size - 0.5, nstar)
    y = rng.uniform(-0.5, size - 0.5, nstar)
    f = power_law_fluxes(nstar, fmin, 2.0, rng, fmax=3000 * fmin)
    img, w = make_mock((size, size), x, y, f, TruthRenderer(epsf, os_), sigma, bkg=5.0, rng=rng)
    keep = f > 2 * fmin
    # near-truth start, so that the chains measure mixing rather than burn-in
    init = dict(x=x[keep] + rng.normal(0, 0.05, keep.sum()), y=y[keep] + rng.normal(0, 0.05, keep.sum()),
                f=f[keep] * np.exp(rng.normal(0, 0.03, keep.sum())))
    return epsf, os_, img, w, init, fmin


def cpu_model():
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--size', type=int, default=96)
    ap.add_argument('--nstar', type=int, default=400)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--nloop', type=int, default=500)
    ap.add_argument('--nsamp', type=int, default=400, help='samples for the numba engines')
    ap.add_argument('--nsamp-python', type=int, default=100)
    ap.add_argument('--engines', nargs='+', default=list(ENGINES))
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=None, help='JSON output')
    args = ap.parse_args()

    epsf, os_, img, w, init, fmin = scene(args.size, args.nstar, args.sigma, args.seed)
    results = {}
    for name in args.engines:
        nsamp = args.nsamp_python if name == 'python' else args.nsamp
        cfg = PCATConfig(trueminf=fmin, flux_slope=2.0, nsamp=nsamp, nloop=args.nloop, regsize=24,
                         float_bkg=True, fourier_order=1, bkg=[5.0], init_catalog=init, seed=args.seed,
                         burnin_frac=0.5, store_models=False, verbose=0, **ENGINES[name])
        t0 = time.process_time()
        res = run_pcat(img, w, epsf, os_, cfg, nc=21)
        cpu = time.process_time() - t0
        pb = res.post_burnin()
        n, logl = res.samples['n'][pb], res.samples['logL'][pb]
        t_samp = cpu * pb.size / max(1, res.samples['n'].size)
        r = dict(nsamp=int(res.samples['n'].size), cpu_s=cpu,
                 us_per_proposal=1e6 * cpu / (res.samples['n'].size * args.nloop),
                 n_mean=float(n.mean()), n_sd=float(n.std()), logL_mean=float(logl.mean()),
                 ess_n=ess_geyer(n), ess_logL=ess_geyer(logl))
        r['ess_n_per_cpu_s'] = r['ess_n'] / t_samp
        r['ess_logL_per_cpu_s'] = r['ess_logL'] / t_samp
        results[name] = r
        print(f'{name:10s} {r["us_per_proposal"]:8.1f} us/prop  N {r["n_mean"]:6.1f} +- {r["n_sd"]:4.1f}  '
              f'logL {r["logL_mean"]:10.1f}  ESS(N)/cpu-s {r["ess_n_per_cpu_s"]:7.3f}  '
              f'ESS(logL)/cpu-s {r["ess_logL_per_cpu_s"]:7.3f}', flush=True)
    if 'python' in results:
        ref = results['python']
        for name, r in results.items():
            r['speedup_per_proposal'] = ref['us_per_proposal'] / r['us_per_proposal']
            r['speedup_ess_n'] = r['ess_n_per_cpu_s'] / ref['ess_n_per_cpu_s']
            print(f'{name:10s} speed-up per proposal {r["speedup_per_proposal"]:5.1f}x, '
                  f'ESS(N) per CPU-s {r["speedup_ess_n"]:5.1f}x')
    if args.out:
        meta = dict(args=vars(args), ntruth=args.nstar, ninit=int(init['x'].size), fmin=fmin,
                    machine=platform.machine(), cpu=cpu_model(), node=platform.node(),
                    python=platform.python_version())
        with open(args.out, 'w') as fh:
            json.dump(dict(meta=meta, results=results), fh, indent=1)


if __name__ == '__main__':
    main()
