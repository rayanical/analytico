"""Thread-safe, byte- and entry-bounded TTL/LRU cache for in-process results."""

from __future__ import annotations

import copy
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


@dataclass(slots=True)
class _Entry(Generic[V]):
    value: V
    size_bytes: int
    expires_at: float


class BoundedTTLCache(Generic[K, V]):
    """An LRU cache that bounds both serialized bytes and entry count.

    Values are copied when inserted and retrieved, so callers cannot mutate a
    cached result through an object they already hold. Values that cannot be
    copied or sized are simply not cached.
    """

    def __init__(self, *, max_entries: int, max_bytes: int, ttl_seconds: float):
        if max_entries <= 0 or max_bytes <= 0 or ttl_seconds <= 0:
            raise ValueError("Cache limits and TTL must be positive.")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[K, _Entry[V]] = OrderedDict()
        self._size_bytes = 0
        self._lock = RLock()

    def get(self, key: K) -> V | None:
        """Return a detached value, or ``None`` on a miss or expired entry."""
        now = time.monotonic()
        with self._lock:
            self._remove_expired(now)
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            try:
                return copy.deepcopy(entry.value)
            except Exception:
                self._remove(key)
                return None

    def set(self, key: K, value: V) -> bool:
        """Store a detached value if it fits the configured bounds."""
        try:
            detached = copy.deepcopy(value)
            size_bytes = len(pickle.dumps((key, detached), protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return False
        if size_bytes > self.max_bytes:
            return False

        with self._lock:
            self._remove_expired(time.monotonic())
            self._remove(key)
            self._entries[key] = _Entry(
                value=detached,
                size_bytes=size_bytes,
                expires_at=time.monotonic() + self.ttl_seconds,
            )
            self._size_bytes += size_bytes
            while len(self._entries) > self.max_entries or self._size_bytes > self.max_bytes:
                oldest_key = next(iter(self._entries))
                self._remove(oldest_key)
        return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._size_bytes = 0

    def __len__(self) -> int:
        with self._lock:
            self._remove_expired(time.monotonic())
            return len(self._entries)

    @property
    def size_bytes(self) -> int:
        with self._lock:
            self._remove_expired(time.monotonic())
            return self._size_bytes

    def _remove_expired(self, now: float) -> None:
        for key, entry in list(self._entries.items()):
            if entry.expires_at <= now:
                self._remove(key)

    def _remove(self, key: K) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._size_bytes -= entry.size_bytes
