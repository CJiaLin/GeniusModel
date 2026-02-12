"""OpenEvolve adapter.

By default this project ships with a placeholder `openevolve/engine.py` so the
end-to-end flow is runnable. Replace that package with your real OpenEvolve
implementation to enable production evolution logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvolutionResult:
    metrics: dict[str, float]
    top_features: list[str]
    notes: list[str]


class OpenEvolveAdapter:
    def __init__(self) -> None:
        from openevolve.engine import OpenEvolveEngine

        self._engine = OpenEvolveEngine()

    def evolve(self, optimize_target: str, generations: int, budget_minutes: int) -> EvolutionResult:
        result = self._engine.evolve(
            optimize_target=optimize_target,
            generations=generations,
            budget_minutes=budget_minutes,
        )
        return EvolutionResult(
            metrics=result["metrics"],
            top_features=result["top_features"],
            notes=result["notes"],
        )
