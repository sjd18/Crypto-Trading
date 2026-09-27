"""Point-in-time creator history and statistical creator-quality scoring.

A creator's past launches become informative only once their outcome is *resolved*
(``discovery.resolution_horizon_s`` after launch, or at migration). Until then a launch counts
toward experience but not toward success/rug rates, so a backtest can never use a token's
future outcome to score its own creator.

Scoring (0-100)
    p_success = Beta posterior mean  (a_s + successes) / (a_s + b_s + resolved)
    p_rug     = Beta posterior mean  (a_r + rugs)      / (a_r + b_r + resolved)
    experience = min(launches, saturation) / saturation
    score = 100 * (w_s * p_success + w_r * (1 - p_rug) + w_e * experience) / (w_s + w_r + w_e)

Wilson lower bounds are reported alongside for conservative filters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import polars as pl


def wilson_lower(successes: int, n: int, z: float = 1.2816) -> float:
    """Wilson score lower bound (default z = 90 % one-sided)."""
    if n == 0:
        return 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


@dataclass(slots=True)
class CreatorRecord:
    creator: str
    launches: int = 0
    resolved: int = 0
    successes: int = 0
    rugs: int = 0
    migrations: int = 0
    sum_ath_multiple: float = 0.0
    last_launch_ms: int = 0

    @property
    def avg_ath_multiple(self) -> float:
        return self.sum_ath_multiple / self.resolved if self.resolved else 0.0

    @property
    def win_rate(self) -> float:
        return self.successes / self.resolved if self.resolved else 0.0

    @property
    def rug_rate(self) -> float:
        return self.rugs / self.resolved if self.resolved else 0.0


@dataclass(slots=True)
class CreatorScore:
    creator: str
    score: float
    p_success: float
    p_rug: float
    launches: int
    resolved: int
    successes: int
    rugs: int
    migrations: int
    avg_ath_multiple: float
    win_rate_lb: float
    rug_rate_lb: float


class CreatorBook:
    """Creator statistics updated only with resolved outcomes (no look-ahead)."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.records: dict[str, CreatorRecord] = {}
        w = cfg.score_weights
        self.w_s, self.w_r, self.w_e = w.get("success", 0.45), w.get("rug_avoid", 0.35), w.get("experience", 0.1)

    def record_launch(self, creator: str, ts_ms: int) -> None:
        r = self.records.setdefault(creator, CreatorRecord(creator))
        r.launches += 1
        r.last_launch_ms = ts_ms

    def record_outcome(self, creator: str, success: bool, rug: bool, migrated: bool, ath_multiple: float) -> None:
        r = self.records.setdefault(creator, CreatorRecord(creator))
        r.resolved += 1
        r.successes += int(success)
        r.rugs += int(rug)
        r.migrations += int(migrated)
        r.sum_ath_multiple += float(min(ath_multiple, 1000.0))

    def score(self, creator: str | None, exclude_current_launch: bool = True) -> CreatorScore:
        """Score as of now. ``exclude_current_launch`` discounts the launch being evaluated."""
        c = self.cfg
        r = self.records.get(creator or "") or CreatorRecord(creator or "")
        launches = max(0, r.launches - (1 if exclude_current_launch and r.launches else 0))
        p_s = (c.creator_prior_alpha + r.successes) / (c.creator_prior_alpha + c.creator_prior_beta + r.resolved)
        p_r = (c.rug_prior_alpha + r.rugs) / (c.rug_prior_alpha + c.rug_prior_beta + r.resolved)
        exp_ = min(launches, c.experience_saturation) / max(1, c.experience_saturation)
        total_w = self.w_s + self.w_r + self.w_e
        score = 100.0 * (self.w_s * p_s + self.w_r * (1.0 - p_r) + self.w_e * exp_) / total_w
        return CreatorScore(r.creator, score, p_s, p_r, launches, r.resolved, r.successes, r.rugs, r.migrations,
                            r.avg_ath_multiple, wilson_lower(r.successes, r.resolved), wilson_lower(r.rugs, r.resolved))

    def to_frame(self, now_ms: int = 0) -> pl.DataFrame:
        rows = []
        for creator in self.records:
            s = self.score(creator, exclude_current_launch=False)
            rows.append({"creator": creator, "launches": s.launches, "resolved": s.resolved, "successes": s.successes,
                         "rugs": s.rugs, "migrations": s.migrations, "avg_ath_multiple": s.avg_ath_multiple,
                         "p_success": s.p_success, "p_rug": s.p_rug, "score": s.score, "updated_ms": now_ms})
        return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"creator": pl.Utf8})
