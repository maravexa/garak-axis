"""
axis_store.py
-------------
Thread-local buffer for axis displacement scalars produced by AxisAwareGenerator.
Keeps activation data decoupled from Garak's normal output pipeline.
"""

import threading
from typing import List, Optional


class AxisStore:
    """Thread-local buffer accumulating per-turn axis displacement scalars."""

    _local = threading.local()

    def record(self, value: float) -> None:
        """Append a displacement scalar for the current thread."""
        if not hasattr(self._local, "values"):
            self._local.values = []
        self._local.values.append(value)

    def flush(self) -> List[float]:
        """Return all recorded values and clear the buffer."""
        vals = list(getattr(self._local, "values", []))
        self._local.values = []
        return vals

    def latest(self) -> Optional[float]:
        """Peek at the most recent value without clearing."""
        vals = getattr(self._local, "values", [])
        return vals[-1] if vals else None

    def depth(self) -> int:
        """Number of values currently buffered."""
        return len(getattr(self._local, "values", []))
