"""Geometry adapters backed by official VGGT-Ω utilities."""

from e3a.geometry.entity_pooling import pool_entity_geometry
from e3a.geometry.vggt_geometry import decode_and_unproject

__all__ = ["decode_and_unproject", "pool_entity_geometry"]
