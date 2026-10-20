"""Backward-compatible re-export of sigvar from pyhorace.data.sigvar."""

from ..data.sigvar import SigVar, sigvar

__all__ = ["sigvar", "SigVar"]
