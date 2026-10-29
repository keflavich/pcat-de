# pcat-de
Probabilistic Cataloger (in the presence of) Diffuse Emission (PCAT-DE). This code builds off the probabilistic cataloging code [Lion](https://github.com/stephenportillo/lion) [(Portillo+2017)](https://iopscience.iop.org/article/10.3847/1538-3881/aa8565/pdf), and the implementation of PCAT-DE is described in Feder+(2022). 

Certain datasets may require a consistent treatment of both point-like and diffuse structured signals in order to maximize the information content (and recover appropriate uncertainties) of the individual components. This code incorporates both explicit spatial templates and a set of non-parametric Fourier component templates as additional ingredients in PCAT's forward model.

The majority of the code is implemented in Python, with the exception of the image model/likelihood evaluations which are written in C. The C code optionally uses Intel's [Math Kernel Library](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html) (MKL), but OpenBLAS and BLAS are available options as well if running on a non-Intel processor.

Documentation can be found on [Read the Docs](https://pcat-de.readthedocs.io).


## This fork: `pcat_fast`

This fork ([keflavich/pcat-de](https://github.com/keflavich/pcat-de)) adds `pcat_fast`, an
importable Python package with a PCAT sampler for single-band JWST images.  It was developed in
the `photometry.transdimensional` module of
[jwst-gc-pipeline](https://github.com/keflavich/jwst-gc-pipeline) and is kept in sync with it.
The PCAT-DE scripts in `pcat-de/` are unchanged.

`pcat_fast` has two engines for the same posterior (catalogue of point sources with a power-law
flux prior, a parsimony penalty and a Fourier-component background):

* `engine='python'` (`PCATSampler`): the PCAT proposal scheme of Portillo et al. (2017) and
  Feder et al. (2020) re-implemented in numpy, with the PCAT-DE / Lion C kernel
  (`pcat_fast/_vendor/pcat_lion_kernel.c`, patched as listed in its header) for the model and
  likelihood;
* `engine='numba'` (`FastPCATSampler`): a compiled sampler.  Sources live in fixed-capacity slot
  arrays with per-region index lists, so a proposal touches only the stamps it changes; the
  likelihood change is summed over the touched pixels; the Fourier background is updated by a
  Gibbs step that uses its separable (y, x) factors.

Options of the numba engine (`PCATConfig`), all off by default so that the default reproduces
the Python engine's proposals:

| option | effect |
| --- | --- |
| `stamp_tol` | per-source stamp size from a flux ladder: the smallest stamp whose truncated wing changes chi^2 by less than `stamp_tol` |
| `move_nstar` | stars moved per region in a star move (0: all, as PCAT); 1 gives the largest gain in effective samples per CPU-hour in our tests |
| `burnin` | burn-in in samples (instead of `burnin_frac`) |
| `burnin_auto` | burn-in = max(`burnin`, 2 t_eq), with t_eq the first sample at which logL and N reach their second-half ranges |
| `target_ess` | stop each chain once min(ESS(N), ESS(logL)) after burn-in reaches this value |
| `birth_resid_frac` | experimental: fraction of births placed in proportion to the squared positive residual S/N of a pixel (the rest uniform); the death acceptance carries the density of the reverse birth, so the target is unchanged. At 0.6, on three 240 x 240 px F182M frames, it raised the ESS of N per CPU-hour 1.9-2.4 times, cost 18-25% more time per sample and lowered the per-star ESS of ln flux per CPU-hour by 14-20% (equal-length chains) |

`condense()` (the condensed catalogue of Portillo et al. 2017, Sect. 5) uses compiled grid
searches when numba is installed and returns the same catalogue as its cKDTree path.

### Install and test

```bash
pip install -e ".[test]"        # numpy, scipy, astropy; numba and photutils for the tests
python -m pytest
```

The C kernel is compiled on first use with `$CC` (default `gcc`) into `$PCAT_FAST_LIBDIR`
(default `pcat_fast/_lib`).  It links OpenBLAS when it finds `libcblas` in the active conda
environment or a system `libopenblas`, and otherwise builds a BLAS-free variant.  Set
`PCAT_FAST_TEST_EPSF` to an oversampled ePSF FITS file (2D, header `OVERSAMP`) to run the tests
that need a real PSF; without it they use a Gaussian ePSF or skip.

### Usage

```python
from pcat_fast.pcat_sampler import PCATConfig, run_pcat
from pcat_fast.condense import condense

# image, inverse_variance: 2D arrays; epsf: oversampled ePSF array (or a FITS path)
# fmin: flux floor of the prior, e.g. S/N 3 for a point source
cfg = PCATConfig(engine='numba', stamp_tol=0.05, move_nstar=1, burnin=100, burnin_auto=True,
                 target_ess=250, nsamp=6000, nloop=1000, trueminf=fmin, fourier_order=2)
res = run_pcat(image, inverse_variance, epsf, oversample, cfg, nc=31)
pb = res.post_burnin()
cat = condense(res.samples['x'][pb], res.samples['y'][pb], res.samples['f'][pb])
```

### Speed

`benchmarks/bench_synthetic.py` runs every engine on a 96 x 96 px synthetic field (400 stars,
Moffat ePSF) from a near-truth start and reports the time per proposal and the effective sample
size of N per CPU-second.  On one AMD EPYC 75F3 core (a shared HiPerGator interactive node) with
the default arguments (`benchmarks/results/bench_synthetic_hipergator.json`):

| engine | us per proposal | ESS(N) per CPU-s |
| --- | ---: | ---: |
| python | 419 | 0.52 |
| numba, same proposals | 45 | 6.7 |
| + `stamp_tol=0.05` | 30 | 6.0 |
| + `move_nstar=1` | 9.0 | 29.7 |

The ESS values come from one chain each (the Python one has 50 post-burn-in samples), so they
scatter by tens of per cent between runs.

### Validation on JWST data

The configuration in the usage example above (one star per region move, `stamp_tol=0.05`, ESS
stop 250, adaptive burn-in) was compared with the Python engine (6000 samples, 2 or 4 chains)
on 16 160 x 160 px JWST/NIRCam F182M cutouts of the Brick (G0.253+0.016): 8 real frames, 2
mocks and 6 frames with injected artificial stars, with 51 px stamps, 52 px regions, four chains
on four CPUs (AMD EPYC 75F3).  Per frame:

| quantity | range over 16 frames (median) |
| --- | --- |
| wall time, compiled (4 chains + condense) | 1.9-3.2 min |
| wall time, Python engine | 79-89 min |
| ESS(N) per CPU-hour, compiled / Python | 27-49 (35) |
| summed chain CPU time, Python / compiled | 20-53 (28) |
| Gelman-Rubin R-hat(N), compiled | 1.000-1.004 |
| Python-engine catalogue sources matched (prob >= 0.5, 0.5 px) | >= 98.9% |
| median ln flux ratio, S/N >= 10 | within +-0.0009 |
| rms position difference, S/N >= 10 | <= 0.0067 px |
| mean N, compiled - Python | -1.25 to +0.18 |

Both engines recover all 280 injected and mock stars above S/N 5.  The mean N is lower on 14
of the 16 frames; on the frame where each change was tested separately, `stamp_tol=0.05`
lowered it by 0.8 sources (0.12 posterior standard deviations), which is the main source of
the shift.  With the same proposals as the Python engine (no `stamp_tol`, all stars moved) the
compiled engine is 4.7 times faster per proposal and reproduces the mean N to 0.1 source.  The
wall-time ratio is larger than the CPU-time ratio on 13 frames because the Python-engine runs
there used 2 chains and the compiled runs 4.  The scripts and per-frame numbers are in the
jwst-gc-pipeline astrometry-paper workspace (`runs/pcat_fast/`); they are not part of this
repository.

### Licence

PCAT-DE (`pcat-de/`, and the vendored copies in `pcat_fast/_vendor/`) is MIT-licensed
(`LICENSE`, `pcat_fast/_vendor/LICENSE_PCAT-DE.txt`).  The rest of `pcat_fast` is
BSD-3-Clause (`pcat_fast/LICENSE.rst`).
