"""Shared fixtures for the transdimensional (PCAT) tests."""
import os
from pathlib import Path

import numpy as np
import pytest

# a real oversampled ePSF (FITS, 2D, header OVERSAMP) for the tests that need one;
# without it the tests use a synthetic Gaussian ePSF or skip
EPSF_PATH = Path(os.environ.get('PCAT_FAST_TEST_EPSF', '/nonexistent/epsf.fits'))
# the PCAT-DE scripts directory of this repository
PCATDE_DIR = Path(__file__).resolve().parents[2] / 'pcat-de'


def lib_dir():
    """Kernel directory (:func:`..kernel.default_lib_dir`)."""
    from ..kernel import default_lib_dir
    return default_lib_dir()


def synthetic_epsf(oversample=5, size=205, fwhm=1.95):
    """Pixel-integrated Gaussian ePSF in the common-ePSF convention (centre at
    index size // 2, values = flux fraction per detector pixel)."""
    from scipy.special import erf
    s = fwhm / 2.3548
    u = (np.arange(size) - size // 2) / oversample

    def integ(u):
        return 0.5 * (erf((u + 0.5) / (np.sqrt(2) * s)) - erf((u - 0.5) / (np.sqrt(2) * s)))
    return np.outer(integ(u), integ(u))


def get_epsf(require_real=False):
    """``(epsf, oversample, is_real)``: the common F182M ePSF when present,
    else a synthetic Gaussian ePSF (or skip when ``require_real``)."""
    if EPSF_PATH.exists():
        from ..psf_poly import load_epsf
        e, os_, _ = load_epsf(EPSF_PATH)
        return e, os_, True
    if require_real:
        pytest.skip(f'{EPSF_PATH} not available')
    return synthetic_epsf(), 5, False


def have_compiler():
    import shutil
    return shutil.which(os.environ.get('CC', 'gcc')) is not None
