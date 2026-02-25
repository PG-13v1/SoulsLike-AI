"""Package exposing the dashboard helpers.

This allows you to import via ``from visualizations import dashboard`` or
``from visualizations.dashboard import ...``.  The real implementation lives
in :mod:`visualizations.dashboard` so we just expose it at package level.
"""

from . import dashboard

__all__ = ["dashboard"]
