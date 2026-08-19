"""Opportunity versus uptake, and the two failures it exists to tell apart.

A rare behaviour is either never legal (a world problem -- `m4f`'s `chop`,
reachable on 0.2% of ticks) or legal and declined (a policy problem -- `m4h`'s
`build`, taken on 28.6% of the ticks it was legal). These pin that the two read
differently, and that the "never legal" case cannot be faked by a policy that
simply never chooses the action.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.agents import action_names, num_actions
from sim.config import load_config
from sim.opportunity import measure
from sim.policy import greedy_forager_actions


@pytest.fixture
def masked():
    """The M3 world with masking on, which is what makes `legal` the world's own
    statement rather than a heuristic. Short episodes: these tests are about the
    bookkeeping, not about any policy's numbers.
    """
    return load_config("config/m3_masked.yaml").replace(**{"world.max_ticks": 60})


def _by_name(report):
    return {a["name"]: a for a in report["actions"]}


def _forager_with(masked, override: str | None):
    """The scripted forager's movement, with its gathers optionally withheld.

    Opportunity has to be *created* by walking to bushes -- an idling policy never
    gets near one, so it would read as "no opportunity" and the two failure modes
    this module separates would collapse into each other. Both policies below
    travel identically; only what they do on arrival differs.
    """
    gather = action_names(masked).index("gather")
    idle = action_names(masked).index("idle")

    def act(obs, mask):
        actions = greedy_forager_actions(obs, masked).astype(np.int64)
        if override == "always":
            return np.where(mask[:, gather], gather, actions)
        if override == "never":
            return np.where(actions == gather, idle, actions)
        return actions

    return act


def test_uptake_is_a_share_of_legal_ticks_not_of_all_ticks(masked):
    """The whole point: `gather` is legal on a couple of percent of ticks in a
    scarce world, so a policy that takes every single opportunity still spends
    almost none of its actions gathering. Dividing by all ticks would report that
    as ~2% uptake and read as refusal.
    """
    report = measure(masked, _forager_with(masked, "always"), episodes=2)
    g = _by_name(report)["gather"]
    assert g["legal"] > 0, "fixture produced no gather opportunities at all"
    assert g["uptake"] == pytest.approx(1.0), "perfect uptake must read as 100%"
    assert g["legal_frac"] < 0.25, (
        "scarce-world gather opportunity should be rare; if this world hands out "
        "opportunity freely the test is not exercising the distinction")


def test_a_declined_action_reads_as_uptake_not_as_missing_opportunity(masked):
    """The `m4h` case. The same walking, the same opportunities, never taken: that
    must show as ~0% uptake with the legal share intact -- not as an absent
    opportunity, which is the `m4f` case and has a completely different fix.
    """
    took = measure(masked, _forager_with(masked, "always"), episodes=2)
    declined = measure(masked, _forager_with(masked, "never"), episodes=2)
    a, b = _by_name(took)["gather"], _by_name(declined)["gather"]
    assert b["legal"] > 0
    assert b["uptake"] == pytest.approx(0.0)
    # Opportunity survives the refusal: same walk, same berries in range. Not
    # asserted equal -- declining to gather leaves the berry on the bush, which
    # creates *more* opportunity on later ticks, not less.
    assert b["legal"] >= a["legal"]


def test_taken_while_masked_stays_zero_for_a_mask_respecting_policy(masked):
    """A tripwire on the mask plumbing rather than on any policy. If a masked
    action is ever *taken*, the count is reported separately so it cannot hide
    inside an uptake share that looks merely low.
    """
    rng = np.random.default_rng(0)

    def legal_random(obs, mask):
        out = np.zeros(mask.shape[0], dtype=np.int64)
        for i in range(mask.shape[0]):
            legal = np.flatnonzero(mask[i])
            out[i] = rng.choice(legal) if legal.size else 0
        return out

    report = measure(masked, legal_random, episodes=2)
    assert all(a["taken_while_masked"] == 0 for a in report["actions"])
    assert len(report["actions"]) == num_actions(masked)
    assert report["masked"] is True
