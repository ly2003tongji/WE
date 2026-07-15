"""Shared traffic-future pack protocol for Replay / IDM / Nexus sidecars."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Provenance:
    """Lightweight provenance; raw arrays stay outside git."""

    source: str
    scene_id: str
    cutoff: int
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "scene_id": self.scene_id,
            "cutoff": self.cutoff,
            **self.extras,
        }


@dataclass
class TrafficFuturePack:
    """WorldEngine-consumable agent futures (horizon frames @ 2 Hz).

    Schema mirrors ``frozen_state_lib`` packs:
    ``futures[token] = {type, position, heading, velocity, valid, source}``.
    """

    source: str
    cutoff: int
    horizon: int
    frequency_hz: float
    dt_s: float
    futures: Dict[str, Any]
    coverage: Dict[str, List[str]]
    future_hash: str
    provenance: Provenance
    n_agents: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "cutoff": self.cutoff,
            "horizon": self.horizon,
            "frequency_hz": self.frequency_hz,
            "dt_s": self.dt_s,
            "n_agents": self.n_agents if self.n_agents is not None else len(self.futures),
            "futures": self.futures,
            "coverage": self.coverage,
            "future_hash": self.future_hash,
            "provenance": self.provenance.to_dict(),
        }
