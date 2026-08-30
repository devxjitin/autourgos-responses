"""
Local call ledger for autourgos-responses.

Re-exports the SQLite-backed ledger implementation from autourgos-openaichat,
which owns the canonical (API-agnostic) implementation. This keeps
`from autourgos_responses.ledger import open_ledger` working while avoiding
a duplicate copy to maintain.
"""

from __future__ import annotations

from autourgos_openaichat.ledger import (
    close_ledger,
    open_ledger,
    write_ledger_entry,
    write_shadow_ledger_entry,
)

__all__ = [
    "open_ledger",
    "write_ledger_entry",
    "write_shadow_ledger_entry",
    "close_ledger",
]
