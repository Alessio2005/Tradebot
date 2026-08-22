# src/tradebot/reporting/__init__.py
"""Reporting sub-package — AFML-style tearsheet generation."""
from __future__ import annotations

from .tearsheet import build_tearsheet, print_tearsheet, save_tearsheet

__all__ = ["build_tearsheet", "print_tearsheet", "save_tearsheet"]
