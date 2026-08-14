"""Metric aggregation and the CSV/console logger."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from sim.metrics import EvalResult, MetricsLogger, UpdateMetrics, _width, summarise_episodes
from sim.world import EpisodeStats


def test_format_width_ignores_precision_digits():
    """`{:>7,.0f}` is seven wide, not seventy: a naive digit scan reads the
    precision as part of the width and the table becomes unreadable."""
    assert _width("{:>5d}") == 5
    assert _width("{:>9,d}") == 9
    assert _width("{:>7,.0f}") == 7
    assert _width("{:>8.4f}") == 8
    assert _width("{:>6.2f}") == 6


def test_all_console_columns_have_sane_widths():
    for _, label, fmt in MetricsLogger.COLUMNS:
        assert 3 <= _width(fmt) <= 12, f"{label} -> {_width(fmt)}"


def test_summarise_empty_episodes_reports_blanks_not_zeros():
    """A blank means "no episode finished"; a zero would mean "everyone died
    instantly". They must not be confused."""
    s = summarise_episodes([])
    assert s["episodes"] == 0
    assert s["mean_lifespan"] is None
    assert s["deaths_per_episode"] is None


def test_summarise_averages_over_episodes():
    episodes = [
        EpisodeStats(ticks=100, deaths=2, berries_gathered=10, mean_lifespan=80.0,
                     mean_final_hunger=30.0, survivors=4),
        EpisodeStats(ticks=100, deaths=4, berries_gathered=20, mean_lifespan=60.0,
                     mean_final_hunger=10.0, survivors=2),
    ]
    s = summarise_episodes(episodes)
    assert s["episodes"] == 2
    assert s["mean_lifespan"] == pytest.approx(70.0)
    assert s["deaths_per_episode"] == pytest.approx(3.0)
    assert s["berries_gathered"] == pytest.approx(15.0)


def test_logger_writes_a_csv_with_blank_cells_for_missing_episodes(tmp_path, capsys):
    logger = MetricsLogger(tmp_path / "metrics.csv")
    logger.log(UpdateMetrics(update=0, agent_steps=100, entropy=2.3))
    logger.log(UpdateMetrics(update=1, agent_steps=200, entropy=2.1,
                             **summarise_episodes([
                                 EpisodeStats(ticks=50, deaths=1, berries_gathered=3,
                                              mean_lifespan=45.0, mean_final_hunger=20.0,
                                              survivors=5)])))
    logger.close()

    rows = list(csv.DictReader(open(tmp_path / "metrics.csv")))
    assert len(rows) == 2
    assert rows[0]["mean_lifespan"] == ""
    assert float(rows[1]["mean_lifespan"]) == pytest.approx(45.0)
    assert float(rows[1]["entropy"]) == pytest.approx(2.1)

    printed = capsys.readouterr().out.splitlines()
    assert "lifespan" in printed[0]
    assert len(set(len(line) for line in printed[1:])) == 1  # aligned columns


def test_eval_result_line_is_one_row(cfg):
    episodes = [EpisodeStats(ticks=600, deaths=1, berries_gathered=40, mean_lifespan=500.0,
                             mean_final_hunger=50.0, survivors=5) for _ in range(3)]
    r = EvalResult.from_episodes("test", episodes, cfg.world.max_ticks)
    assert r.mean_lifespan == pytest.approx(500.0)
    assert r.survival_rate == pytest.approx(500.0 / cfg.world.max_ticks)
    assert "\n" not in r.line()
    assert "test" in r.line()
