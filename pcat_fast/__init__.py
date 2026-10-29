"""pcat_fast: a compiled (numba) PCAT sampler with a JWST-oriented interface.

Entry points: :class:`.pcat_sampler.PCATConfig` and :func:`.pcat_sampler.run_pcat`
(``PCATConfig(engine='numba')`` selects :class:`.fast_sampler.FastPCATSampler`),
:func:`.condense.condense` for the catalogue, and :mod:`.pcat_io` for output files.
Developed in the jwst_gc_pipeline ``photometry.transdimensional`` module; see the
README for the provenance and the validation.
"""
__version__ = '0.1.0'
