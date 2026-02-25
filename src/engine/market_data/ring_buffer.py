"""High-throughput OHLCV ring buffer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class OhlcvRing:
    """Fixed-capacity OHLCV ring with O(1) append and O(1) latest-window slicing."""

    capacity: int
    head: int
    size: int
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @classmethod
    def create(cls, capacity: int) -> OhlcvRing:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        zeros_i64 = np.zeros(capacity, dtype=np.int64)
        zeros_f64 = np.zeros(capacity, dtype=np.float64)
        return cls(
            capacity=capacity,
            head=0,
            size=0,
            ts=zeros_i64.copy(),
            open=zeros_f64.copy(),
            high=zeros_f64.copy(),
            low=zeros_f64.copy(),
            close=zeros_f64.copy(),
            volume=zeros_f64.copy(),
        )

    def append(
        self,
        *,
        ts_ms: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        idx = self.head
        self.ts[idx] = int(ts_ms)
        self.open[idx] = float(open_)
        self.high[idx] = float(high)
        self.low[idx] = float(low)
        self.close[idx] = float(close)
        self.volume[idx] = float(volume)

        self.head = (self.head + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def _ordered_indices(self) -> np.ndarray:
        if self.size == 0:
            return np.array([], dtype=np.int64)
        start = (self.head - self.size) % self.capacity
        if start + self.size <= self.capacity:
            return np.arange(start, start + self.size, dtype=np.int64)
        left = np.arange(start, self.capacity, dtype=np.int64)
        right = np.arange(0, (start + self.size) % self.capacity, dtype=np.int64)
        return np.concatenate((left, right))

    def latest(self, n: int) -> dict[str, np.ndarray]:
        if n <= 0:
            return {
                "ts": np.array([], dtype=np.int64),
                "open": np.array([], dtype=np.float64),
                "high": np.array([], dtype=np.float64),
                "low": np.array([], dtype=np.float64),
                "close": np.array([], dtype=np.float64),
                "volume": np.array([], dtype=np.float64),
            }
        if self.size == 0:
            return self.latest(0)

        count = min(n, self.size)
        ordered = self._ordered_indices()
        subset = ordered[-count:]
        return {
            "ts": self.ts[subset].copy(),
            "open": self.open[subset].copy(),
            "high": self.high[subset].copy(),
            "low": self.low[subset].copy(),
            "close": self.close[subset].copy(),
            "volume": self.volume[subset].copy(),
        }

    def latest_timestamp(self) -> int | None:
        if self.size == 0:
            return None
        idx = (self.head - 1) % self.capacity
        return int(self.ts[idx])
