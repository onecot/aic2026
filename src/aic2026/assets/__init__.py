"""Offline asset-production toolchain.

Everything required to rebuild  raw -> keyframes -> embeddings -> index,
plus deterministic worker split/merge/verification helpers. Nothing here is
imported by the runtime API path; these modules are driven by the scripts in
``scripts/build_*.py`` (and were historically run as distributed workers).
"""
