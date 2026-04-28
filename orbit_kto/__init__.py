"""Orbit-KTO package entrypoint.

Keeps post-training code isolated from the base training path.
"""

from .orbit_kto import train

__all__ = ["train"]
