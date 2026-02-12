"""OpenEvolve engine placeholder.

You can replace this file with your real OpenEvolve implementation.
Current implementation provides a minimal compatible interface so backend can run.
"""

from __future__ import annotations


class OpenEvolveEngine:
    def evolve(self, optimize_target: str, generations: int, budget_minutes: int) -> dict:
        base_auc = 0.79
        growth = min(generations * 0.002, 0.07)
        return {
            "metrics": {
                "AUC": round(base_auc + growth, 4),
                "KS": round(0.32 + growth * 1.5, 4),
                "Recall@Top5%": round(0.46 + growth * 1.2, 4),
                "LatencyMs": round(max(20.0, 40 - generations * 0.4), 2),
            },
            "top_features": [
                "txn_amt_7d_std",
                "night_txn_ratio",
                "device_switch_cnt",
            ],
            "notes": [
                f"OpenEvolve placeholder finished with target={optimize_target}.",
                f"Budget={budget_minutes}min, generations={generations}.",
            ],
        }
