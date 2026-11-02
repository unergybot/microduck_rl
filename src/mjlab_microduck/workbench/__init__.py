"""Portable, CPU-only ROM recipe discovery and immutable configuration resolution.

This package does not import task registration, GPU libraries, UI code or a
scheduler. Resources snapshot the existing task factories; consuming a catalog
never grants execution capability. See resources/CONTRACT.md for hash rules.
"""
from .configuration import canonical_json, catalog, resolve_experiment, verify_experiment

__all__ = ['catalog', 'resolve_experiment', 'verify_experiment', 'canonical_json']
