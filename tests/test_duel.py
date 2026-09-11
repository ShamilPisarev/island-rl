"""The match routes by allegiance and is reproducible without training."""
import numpy as np
import pytest

from sim.config import load_config
from sim.duel import TribeArbiter, play
from sim.obsview import ObsView
from sim.world import World

CHECKPOINTS = ["checkpoints/arb7-village/latest.pt", "checkpoints/arb5-hh/latest.pt"]


def test_routing_tracks_changed_allegiance():
    from pathlib import Path
    if not all(Path(p).exists() for p in CHECKPOINTS):
        pytest.skip("local trained weights are not shipped in git")
    cfg = load_config("config/duel.yaml")
    world = World(cfg, seed=10000)
    arb = TribeArbiter(world, CHECKPOINTS, seed=10)
    obs = ObsView(world.observations(), cfg)
    mask = world.action_mask()
    rng = np.random.default_rng(1)
    world.tribe[:] = 0
    a = arb.choose(obs, mask, rng)
    world.tribe[:] = 1
    b = arb.choose(obs, mask, rng)
    world.tribe[::2] = 0
    mixed = arb.choose(obs, mask, rng)
    assert np.array_equal(mixed[::2], a[::2])
    assert np.array_equal(mixed[1::2], b[1::2])
    assert (a < arb.widths[0]).all()
    assert (b < arb.widths[1]).all()


def test_short_match_repeats_exactly(tmp_path):
    from pathlib import Path
    if not all(Path(p).exists() for p in CHECKPOINTS):
        pytest.skip("local trained weights are not shipped in git")
    cfg = load_config("config/duel.yaml").replace(**{"world.max_ticks": 8})
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    play(cfg, CHECKPOINTS, 10000, a)
    play(cfg, CHECKPOINTS, 10000, b)
    import json
    assert json.loads(a.read_text())["ticks"] == json.loads(b.read_text())["ticks"]
