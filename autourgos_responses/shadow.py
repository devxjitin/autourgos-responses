"""
Shadow-mode dual dispatch helpers for autourgos-responses.

Re-exports the (API-agnostic) similarity helper from autourgos-openaichat
instead of duplicating it.
"""

from __future__ import annotations

from autourgos_openaichat.shadow import compute_similarity

__all__ = ["compute_similarity"]
