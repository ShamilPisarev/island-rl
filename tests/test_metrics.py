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


def _episodes(lifespans: list[float]) -> list[EpisodeStats]:
    return [EpisodeStats(ticks=600, mean_lifespan=v) for v in lifespans]


def test_paired_lines_difference_is_per_island_not_of_the_means():
    """The whole point of pairing: island noise cancels, so a consistent small
    edge is visible even when the two spreads overlap completely.

    Both policies below range over 200 ticks and their means differ by 10, but
    the learned one is better on every island by exactly 10 -- unpaired that is
    invisible, paired it has zero standard error.
    """
    from sim.evaluate import paired_lines

    islands = [300.0, 400.0, 500.0]
    runs = [("scripted forager", _episodes(islands)),
            ("learned policy", _episodes([v + 10 for v in islands]))]
    lines = paired_lines(runs, "learned policy")
    body = "\n".join(lines)
    assert "3 islands" in lines[0]
    assert "+10.0" in body and "3/3 islands" in body
    assert "scripted forager" in body and "learned policy" not in body[body.index("vs"):]


def test_paired_lines_counts_wins_not_just_the_mean():
    """One huge island must not read as a general win."""
    from sim.evaluate import paired_lines

    runs = [("scripted forager", _episodes([500.0, 500.0, 500.0, 500.0])),
            ("learned policy", _episodes([490.0, 490.0, 490.0, 800.0]))]
    body = "\n".join(paired_lines(runs, "learned policy"))
    assert "+67.5" in body        # the mean says a large win
    assert "1/4 islands" in body  # the win count says it was one island
