"""Metric aggregation, CSV logging, and the console table.

Two kinds of number get logged and they are easy to confuse:

* *Episode* metrics (lifespan, deaths, berries) come from episodes that finished
  during an update. Early in training an update may finish none at all, so these
  are reported as blanks rather than zeros -- a zero here would read as "every
  agent died instantly", which is the opposite of what a blank means.
* *Optimisation* metrics (losses, entropy, explained variance) come from the
  update itself and are always present.
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .world import EpisodeStats


@dataclass
class UpdateMetrics:
    """One row of the training log."""

    update: int = 0
    agent_steps: int = 0
    elapsed_s: float = 0.0
    steps_per_s: float = 0.0
    lr: float = 0.0

    # episode metrics (blank when no episode finished during this update)
    episodes: int = 0
    mean_lifespan: float | None = None
    deaths_per_episode: float | None = None
    berries_gathered: float | None = None
    mean_final_hunger: float | None = None
    survivors: float | None = None
    steals: float | None = None
    contests_lost: float | None = None
    wood_gathered: float | None = None
    stone_gathered: float | None = None
    builds: float | None = None
    shelters: float | None = None
    night_sheltered_frac: float | None = None

    # optimisation metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    mean_reward: float = 0.0

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def summarise_episodes(episodes: Iterable[EpisodeStats]) -> dict[str, float | None]:
    episodes = list(episodes)
    if not episodes:
        return {
            "episodes": 0, "mean_lifespan": None, "deaths_per_episode": None,
            "berries_gathered": None, "mean_final_hunger": None, "survivors": None,
            "steals": None, "contests_lost": None,
            "wood_gathered": None, "stone_gathered": None, "builds": None,
            "shelters": None, "night_sheltered_frac": None,
        }
    return {
        "episodes": len(episodes),
        "mean_lifespan": float(np.mean([e.mean_lifespan for e in episodes])),
        "deaths_per_episode": float(np.mean([e.deaths for e in episodes])),
        "berries_gathered": float(np.mean([e.berries_gathered for e in episodes])),
        "mean_final_hunger": float(np.mean([e.mean_final_hunger for e in episodes])),
        "survivors": float(np.mean([e.survivors for e in episodes])),
        "steals": float(np.mean([e.steals for e in episodes])),
        "contests_lost": float(np.mean([e.contests_lost for e in episodes])),
        "wood_gathered": float(np.mean([e.wood_gathered for e in episodes])),
        "stone_gathered": float(np.mean([e.stone_gathered for e in episodes])),
        "builds": float(np.mean([e.builds for e in episodes])),
        "shelters": float(np.mean([e.shelters_completed for e in episodes])),
        "night_sheltered_frac": float(
            sum(e.night_ticks_sheltered for e in episodes)
            / max(sum(e.night_ticks_sheltered + e.night_ticks_exposed for e in episodes), 1)
        ),
    }


def explained_variance(predictions: np.ndarray, targets: np.ndarray) -> float:
    """1 - Var(target - prediction) / Var(target).

    The single most useful number for spotting a broken value head: ~0 means the
    critic is no better than predicting the mean, <0 means it is actively worse.
    """
    targets = np.asarray(targets, dtype=np.float64).ravel()
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    var = targets.var()
    if var < 1e-12:
        return float("nan")
    return float(1.0 - (targets - predictions).var() / var)


class MetricsLogger:
    """Appends rows to a CSV and prints an aligned console table."""

    COLUMNS = [
        ("update", "upd", "{:>5d}"),
        ("agent_steps", "steps", "{:>9,d}"),
        ("steps_per_s", "step/s", "{:>7,.0f}"),
        ("mean_lifespan", "lifespan", "{:>8.1f}"),
        ("deaths_per_episode", "deaths", "{:>6.2f}"),
        ("berries_gathered", "berries", "{:>7.1f}"),
        ("steals", "steals", "{:>6.1f}"),
        ("shelters", "shelt", "{:>5.1f}"),
        ("night_sheltered_frac", "night%", "{:>6.2f}"),
        ("mean_final_hunger", "hunger", "{:>6.1f}"),
        ("mean_reward", "rew/step", "{:>8.4f}"),
        ("entropy", "entropy", "{:>7.3f}"),
        ("value_loss", "v_loss", "{:>7.3f}"),
        ("explained_variance", "exp_var", "{:>7.3f}"),
        ("approx_kl", "kl", "{:>6.4f}"),
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: csv.DictWriter | None = None
        self._fh = None
        self._rows_printed = 0
        self.rows: list[dict[str, Any]] = []

    def _ensure_open(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row.keys()))
            self._writer.writeheader()

    def log(self, metrics: UpdateMetrics, echo: bool = True) -> None:
        row = metrics.to_row()
        self._ensure_open(row)
        self._writer.writerow(row)
        self._fh.flush()
        self.rows.append(row)
        if echo:
            self._print(row)

    def _print(self, row: dict[str, Any]) -> None:
        if self._rows_printed % 20 == 0:
            header = " ".join(f"{label:>{max(len(label), _width(fmt))}}"
                              for _, label, fmt in self.COLUMNS)
            print(header)
            print("-" * len(header))
        cells = []
        for key, label, fmt in self.COLUMNS:
            value = row.get(key)
            width = max(len(label), _width(fmt))
            cells.append(f"{'':>{width}}" if value is None else f"{fmt.format(value):>{width}}")
        print(" ".join(cells))
        self._rows_printed += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None


def _width(fmt: str) -> int:
    """Field width out of a format spec like ``{:>7,.0f}``.

    Must not scoop up the precision digits -- a naive digit filter reads that
    spec as width 70 and prints a table nobody can follow.
    """
    match = re.match(r"\{:[<>^=]?[+\- ]?0?(\d+)", fmt)
    return int(match.group(1)) if match else 8


@dataclass
class EvalResult:
    """Aggregate over evaluation episodes, used for baselines and checkpoints."""

    label: str
    episodes: int
    mean_lifespan: float
    lifespan_std: float
    deaths_per_episode: float
    berries_gathered: float
    mean_final_hunger: float
    survivors: float
    survival_rate: float
    steals: float = 0.0
    shelters: float = 0.0
    night_sheltered_frac: float = 0.0

    @staticmethod
    def from_episodes(label: str, episodes: list[EpisodeStats], max_ticks: int) -> "EvalResult":
        lifespans = np.array([e.mean_lifespan for e in episodes], dtype=np.float64)
        return EvalResult(
            label=label,
            episodes=len(episodes),
            mean_lifespan=float(lifespans.mean()),
            lifespan_std=float(lifespans.std()),
            deaths_per_episode=float(np.mean([e.deaths for e in episodes])),
            berries_gathered=float(np.mean([e.berries_gathered for e in episodes])),
            mean_final_hunger=float(np.mean([e.mean_final_hunger for e in episodes])),
            survivors=float(np.mean([e.survivors for e in episodes])),
            survival_rate=float(lifespans.mean() / max_ticks),
            steals=float(np.mean([e.steals for e in episodes])),
            shelters=float(np.mean([e.shelters_completed for e in episodes])),
            night_sheltered_frac=float(
                sum(e.night_ticks_sheltered for e in episodes)
                / max(sum(e.night_ticks_sheltered + e.night_ticks_exposed for e in episodes), 1)
            ),
        )

    def line(self) -> str:
        return (
            f"{self.label:<22} lifespan {self.mean_lifespan:7.1f} +-{self.lifespan_std:5.1f}  "
            f"({self.survival_rate * 100:5.1f}% of episode)  deaths {self.deaths_per_episode:4.2f}  "
            f"berries {self.berries_gathered:6.1f}  final hunger {self.mean_final_hunger:5.1f}"
            + (f"  steals {self.steals:5.1f}" if self.steals else "")
            + (f"  shelters {self.shelters:3.1f} ({self.night_sheltered_frac * 100:3.0f}% nights in)"
               if self.shelters or self.night_sheltered_frac else "")
        )
