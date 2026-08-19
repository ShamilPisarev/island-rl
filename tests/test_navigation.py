"""The navigation measurement, and the metric lesson it exists to pin.

The headline finding it produced -- that the milestone chain loses navigation
between M1 and M3 -- rests entirely on the distance bucketing. An aggregate
toward-target share is dominated by whichever band the policy spends its ticks
in, which for a policy that eats is the near field. These tests pin both that the
measurement can tell a navigator from a random walk, and that an aggregate
cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.agents import MOVE_VECTORS, N_MOVE_ACTIONS
from sim.config import load_config
from sim.evaluate import make_act_fn
from sim.navigation import BANDS, measure
from sim.policy import greedy_forager_actions


@pytest.fixture
def far():
    """The probe world: one distant cluster, so agents spawn far from food and
    the far distance bands actually get samples. On default.yaml nothing is ever
    more than a few units from a bush and the measurement has nothing to measure.
    """
    return load_config("config/nav_probe.yaml").replace(**{"world.max_ticks": 150})


def _bands_with_samples(bands, lo_at_least=3, min_n=60):
    return [b for b in bands if b["lo"] >= lo_at_least and b["n"] >= min_n]


def test_a_navigator_separates_from_a_random_walk_in_the_same_world(far):
    """The claim the metric has to support: a known navigator reads far above a
    random walk *measured in the same world*.

    Deliberately not asserted against a nominal 50%. A random walk is not 50% in
    every world -- movement is projected back onto the disc, so a walker near the
    shoreline drifts inward, and if the food happens to be inward that reads as
    navigation. Measured here, random scores ~64% in the 10-20 band. The floor is
    a property of the world and has to be measured in it, not assumed.
    """
    nav = measure(far, lambda o, m: greedy_forager_actions(o, far), episodes=2)
    rnd = measure(far, make_act_fn("random", far, None, seed=3), episodes=2)
    sampled = _bands_with_samples(nav["food"])
    assert sampled, "no far-band samples: the fixture cannot exercise the metric"
    for b in sampled:
        floor = next((r["toward"] for r in rnd["food"] if r["lo"] == b["lo"] and r["n"] >= 30),
                     0.5)
        assert b["toward"] > 0.9, f"forager only {b['toward']:.1%} at {b['lo']}-{b['hi']}"
        assert b["toward"] - floor > 0.2, (
            f"at {b['lo']}-{b['hi']} the navigator ({b['toward']:.1%}) does not "
            f"separate from this world's random floor ({floor:.1%})")


def test_an_aggregate_would_hide_far_field_competence():
    """The lesson, as arithmetic: m1's real numbers.

    Near field 47.5% on 17,347 samples, far field 88.4% on 199. The sample-weighted
    aggregate is 48.0% -- indistinguishable from a random walk, and the reason the
    first version of this finding wrongly concluded the project had never learned
    to navigate. Nothing about the policy changed between those two readings; only
    which states were averaged together. See rule 6 in CLAUDE.md.
    """
    near_n, near_share = 17347, 0.475
    far_n, far_share = 199, 0.884
    aggregate = (near_n * near_share + far_n * far_share) / (near_n + far_n)
    assert aggregate == pytest.approx(0.480, abs=0.005)
    assert abs(aggregate - near_share) < 0.01, "the aggregate is just the near field"
    assert far_share - aggregate > 0.4, "and it buries a large real competence"


def test_only_moves_are_counted(far):
    """Gathering and idling are not moves and must never enter the denominator:
    counting them would let a bush-camper dilute its own score toward 50%."""
    def always_idle(obs, mask=None):
        return np.full(obs.shape[0], N_MOVE_ACTIONS)      # IDLE

    out = measure(far, always_idle, episodes=1)
    assert sum(b["n"] for b in out["food"]) == 0


def test_bands_are_contiguous_and_cover_everything():
    assert BANDS[0][0] == 0
    for (_, a_hi), (b_lo, _) in zip(BANDS, BANDS[1:]):
        assert a_hi == b_lo, "bands must not leave a gap a sample could fall into"
    assert BANDS[-1][1] > 1e8, "the last band must be unbounded"


def test_move_vectors_are_unit_length_and_distinct():
    """The measurement compares distance before and after applying MOVE_VECTORS,
    so a non-unit or duplicated direction would silently bias every result."""
    norms = np.hypot(MOVE_VECTORS[:, 0], MOVE_VECTORS[:, 1])
    assert np.allclose(norms, 1.0), norms
    assert len({tuple(np.round(v, 6)) for v in MOVE_VECTORS}) == N_MOVE_ACTIONS


def test_night_exposure_separates_nothing_built_from_nobody_went_home():
    """The two readings of an exposed night, which the shelter statistics conflate.

    Built out of a world where no shelter can ever be finished (a policy that never
    builds): every exposed tick must be attributed to "nothing built", and none to
    distance. If that attribution were reversed, m4h's night finding -- a finished
    shelter available for 100% of its exposed ticks -- would be meaningless.
    """
    from sim.config import load_config
    from sim.navigation import night_exposure

    cfg = load_config("config/m4h.yaml").replace(**{"world.max_ticks": 200})
    idle = np.full(cfg.world.num_agents, 8, dtype=np.int64)   # 8 == idle
    r = night_exposure(cfg, lambda obs, mask: idle, episodes=1)

    assert r["night_ticks"] > 0, "fixture saw no night at all"
    assert r["shelters"] == 0.0
    assert r["exposed"] == pytest.approx(1.0)
    assert r["none_finished"] == pytest.approx(1.0)
    assert np.isnan(r["mean_distance"]), (
        "no finished shelter can be a distance away; a number here means the "
        "distance branch is counting sites that are not done")
