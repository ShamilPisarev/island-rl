"""Behavioural divergence metrics (Milestone 2)."""

from __future__ import annotations

import numpy as np
import pytest

from sim.agents import ACTION_NAMES, GATHER, IDLE, N_ACTIONS, N_MOVE_ACTIONS
from sim.divergence import (
    TERRITORY_BINS,
    _js_divergence,
    analyse,
    pairwise_divergence,
    summarise,
)
from sim.policy import greedy_forager_actions


@pytest.fixture
def short_cfg(cfg):
    return cfg.replace(**{"world.max_ticks": 80})


# --- divergence maths -------------------------------------------------------


def test_js_divergence_is_zero_for_identical_distributions():
    p = np.array([0.1, 0.2, 0.7])
    assert _js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_js_divergence_is_one_bit_for_disjoint_support():
    """Fully disjoint distributions sit at exactly 1 bit, which is what makes the
    reported numbers interpretable as a fraction of "completely different"."""
    assert _js_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_js_divergence_is_symmetric_and_finite_with_zeros():
    """KL would be infinite here; an agent that never visits a cell is normal."""
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.0, 0.5, 0.5])
    assert _js_divergence(p, q) == pytest.approx(_js_divergence(q, p))
    assert np.isfinite(_js_divergence(p, q))


def test_js_divergence_ignores_scale():
    counts = np.array([10.0, 30.0])
    assert _js_divergence(counts, counts * 7) == pytest.approx(0.0, abs=1e-12)


def test_js_divergence_handles_an_all_zero_distribution():
    """An agent that died instantly contributes no counts; that must not NaN."""
    assert np.isfinite(_js_divergence(np.zeros(4), np.array([1.0, 0, 0, 0])))


def test_pairwise_matrix_is_symmetric_with_zero_diagonal():
    dists = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0.5, 0.5, 0])]
    m = np.array(pairwise_divergence(dists))
    assert m.shape == (3, 3)
    assert np.allclose(np.diag(m), 0.0)
    assert np.allclose(m, m.T)


# --- analysis ---------------------------------------------------------------


def scripted(cfg):
    return lambda obs: greedy_forager_actions(obs, cfg)


def test_report_shape(short_cfg):
    report = analyse(short_cfg, scripted(short_cfg), episodes=2, seed=1)
    n = short_cfg.world.num_agents

    assert report["schema_version"] == 1
    assert report["num_agents"] == n
    assert len(report["agents"]) == n
    assert len(report["agent_colors"]) == n
    assert report["action_names"] == list(ACTION_NAMES)
    assert np.array(report["action_divergence"]).shape == (n, n)
    assert np.array(report["territory_divergence"]).shape == (n, n)

    for a in report["agents"]:
        assert len(a["action_counts"]) == N_ACTIONS
        assert np.array(a["territory"]).shape == (TERRITORY_BINS, TERRITORY_BINS)


def test_shares_sum_to_one_and_territory_is_normalised(short_cfg):
    report = analyse(short_cfg, scripted(short_cfg), episodes=2, seed=2)
    for a in report["agents"]:
        assert a["gather_share"] + a["travel_share"] + a["idle_share"] == pytest.approx(1.0)
        assert sum(a["action_mix"].values()) == pytest.approx(1.0)
        assert np.array(a["territory"]).sum() == pytest.approx(1.0)
        assert 0.0 <= a["gather_success_rate"] <= 1.0
        assert 0.0 <= a["time_in_gather_range"] <= 1.0


def test_action_counts_match_the_shares(short_cfg):
    report = analyse(short_cfg, scripted(short_cfg), episodes=2, seed=3)
    for a in report["agents"]:
        counts = np.array(a["action_counts"])
        total = counts.sum()
        assert a["gather_share"] == pytest.approx(counts[GATHER] / total)
        assert a["travel_share"] == pytest.approx(counts[:N_MOVE_ACTIONS].sum() / total)
        assert a["idle_share"] == pytest.approx(counts[IDLE] / total)
        assert a["gather_attempts"] == int(counts[GATHER])


def test_gather_successes_never_exceed_attempts(short_cfg):
    report = analyse(short_cfg, scripted(short_cfg), episodes=2, seed=4)
    for a in report["agents"]:
        assert a["gather_successes"] <= a["gather_attempts"]


def test_an_idle_only_policy_reports_pure_idle(short_cfg):
    """A degenerate controller pins every share to a known value, which catches
    off-by-one bookkeeping that realistic policies would hide."""
    n = short_cfg.world.num_agents
    report = analyse(short_cfg, lambda obs: np.full(n, IDLE), episodes=1, seed=5)
    for a in report["agents"]:
        assert a["idle_share"] == pytest.approx(1.0)
        assert a["gather_share"] == pytest.approx(0.0)
        assert a["travel_share"] == pytest.approx(0.0)
        assert a["gather_attempts"] == 0
        assert a["gather_success_rate"] == 0.0


def test_identical_behaviour_gives_near_zero_action_divergence(short_cfg):
    """Every agent running the same deterministic controller must look the same in
    action space. This is the control the learned numbers get read against."""
    report = analyse(short_cfg, scripted(short_cfg), episodes=3, seed=6)
    m = np.array(report["action_divergence"])
    off = ~np.eye(m.shape[0], dtype=bool)
    assert m[off].max() < 0.25


def test_deliberately_split_behaviour_shows_up_as_divergence(short_cfg):
    """Half the agents gather, half idle: divergence must be large and structured
    -- near zero within each faction, high between them."""
    n = short_cfg.world.num_agents
    half = n // 2

    def split(obs: np.ndarray) -> np.ndarray:
        actions = np.full(n, IDLE)
        actions[:half] = GATHER
        return actions

    report = analyse(short_cfg, split, episodes=1, seed=7)
    m = np.array(report["action_divergence"])
    within = m[:half, :half][~np.eye(half, dtype=bool)]
    between = m[:half, half:]
    assert within.max() == pytest.approx(0.0, abs=1e-9)
    assert between.min() == pytest.approx(1.0, abs=1e-6)


def test_territory_follows_position(short_cfg):
    """Agents parked in the corners must light up different cells."""
    n = short_cfg.world.num_agents

    def frozen(obs: np.ndarray) -> np.ndarray:
        return np.full(n, IDLE)

    report = analyse(short_cfg, frozen, episodes=1, seed=8)
    peaks = set()
    for a in report["agents"]:
        heat = np.array(a["territory"])
        assert heat.max() == pytest.approx(1.0)  # never moved, so one cell holds everything
        peaks.add(tuple(np.unravel_index(heat.argmax(), heat.shape)))
    assert len(peaks) > 1  # they spawned in different places


def test_fixed_map_uses_one_layout_and_reports_it(short_cfg):
    fixed = analyse(short_cfg, scripted(short_cfg), episodes=2, seed=9, fixed_map=True)
    assert fixed["fixed_map"] is True
    assert len(fixed["bushes"]) == short_cfg.bushes.count

    resampled = analyse(short_cfg, scripted(short_cfg), episodes=2, seed=9, fixed_map=False)
    assert resampled["fixed_map"] is False
    assert resampled["bushes"] == []  # withheld: it would be one episode's layout


def test_fixed_map_is_deterministic(short_cfg):
    a = analyse(short_cfg, scripted(short_cfg), episodes=2, seed=11)
    b = analyse(short_cfg, scripted(short_cfg), episodes=2, seed=11)
    assert a["agents"] == b["agents"]
    assert a["bushes"] == b["bushes"]


def test_summarise_is_one_line_per_agent_plus_furniture(short_cfg):
    report = analyse(short_cfg, scripted(short_cfg), episodes=1, seed=12)
    text = summarise(report)
    assert "agent" in text
    assert "JS divergence" in text
    for a in report["agents"]:
        assert f"{a['agent']:>5}" in text
