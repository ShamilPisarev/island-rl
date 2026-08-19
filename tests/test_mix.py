"""cfg.mix: two world configs in one VecWorld, for interleaved training.

The point of these is that a mismatch must RAISE rather than train quietly. One
policy sees both worlds, so disagreeing layouts would feed trained weights the
wrong features with nothing printed -- the failure mode --init-from's column_map
already exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.agents import IDLE
from sim.config import load_config
from sim.world import VecWorld


@pytest.fixture
def mixed_cfg():
    return load_config("config/nav_spread_mix.yaml")


def test_mix_splits_the_envs_by_fraction(mixed_cfg):
    v = VecWorld(mixed_cfg, seed=1, num_envs=8)
    assert sum(v.is_mix) == 4                      # fraction 0.5
    assert v.is_mix == [True] * 4 + [False] * 4    # mixed envs take the low indices


def test_mixed_and_primary_worlds_have_different_geography(mixed_cfg):
    """The whole point: the mixed envs are a different food distribution."""
    v = VecWorld(mixed_cfg, seed=1, num_envs=8)
    mix_berries = v.worlds[0].bush_berries.sum()
    primary_berries = v.worlds[-1].bush_berries.sum()
    assert mix_berries > primary_berries
    assert v.worlds[0].cfg.bushes.num_clusters == 1
    assert v.worlds[-1].cfg.bushes.num_clusters == 6


def test_no_mix_config_means_every_env_is_primary(cfg):
    v = VecWorld(cfg, seed=1, num_envs=4)
    assert v.is_mix == [False] * 4
    assert v.mix_cfg is None


def test_zero_fraction_disables_mixing(mixed_cfg):
    v = VecWorld(mixed_cfg.replace(**{"mix.fraction": 0.0}), seed=1, num_envs=4)
    assert v.is_mix == [False] * 4
    assert v.mix_cfg is None


@pytest.mark.parametrize("override,what", [
    ({"observation.k_bushes": 6}, "observation dim"),
    ({"competition.enable_steal": False}, "action count"),
    ({"world.num_agents": 4}, "agent count"),
])
def test_layout_mismatch_raises(mixed_cfg, tmp_path, override, what):
    """A mix config that disagrees on anything the policy sees must be refused."""
    import yaml
    bad = yaml.safe_load(open("config/nav_probe_mix.yaml"))
    for key, value in override.items():
        section, _, leaf = key.partition(".")
        bad.setdefault(section, {})[leaf] = value
    path = tmp_path / "bad_mix.yaml"
    # extends is resolved relative to the child's directory, so point it at the real one
    bad["extends"] = str((__import__("pathlib").Path("config/m3_masked.yaml").resolve()))
    path.write_text(yaml.safe_dump(bad))

    cfg = mixed_cfg.replace(**{"mix.config": str(path)})
    with pytest.raises(ValueError, match=what):
        VecWorld(cfg, seed=1, num_envs=4)


def _short_mix_cfg(mixed_cfg, tmp_path, ticks: int = 4):
    """The mixed pair with BOTH worlds' max_ticks shortened.

    Shortening only the primary would not work, and that is the gotcha this
    helper exists to encode: the mix config is loaded from DISK inside VecWorld,
    so `cfg.replace(...)` and `--set world.*` reach the primary world only.
    """
    import yaml
    from pathlib import Path
    raw = yaml.safe_load(open("config/nav_probe_mix.yaml"))
    raw["extends"] = str(Path("config/m3_masked.yaml").resolve())
    raw.setdefault("world", {})["max_ticks"] = ticks
    path = tmp_path / "short_mix.yaml"
    path.write_text(yaml.safe_dump(raw))
    return mixed_cfg.replace(**{"world.max_ticks": ticks, "mix.config": str(path)})


def test_mix_config_keeps_its_own_world_settings(mixed_cfg):
    """`replace` on the primary does not reach the mix config -- it is re-read."""
    v = VecWorld(mixed_cfg.replace(**{"world.max_ticks": 7}), seed=1, num_envs=4)
    assert v.worlds[0].cfg.world.max_ticks == 600   # mixed env, from disk
    assert v.worlds[-1].cfg.world.max_ticks == 7    # primary env, overridden


def test_mix_episodes_are_trained_on_but_not_logged(mixed_cfg, tmp_path):
    """Mixed transitions reach the rollout; mixed EPISODES stay out of the metrics.

    A lifespan averaged over an easy probe world and a scarce one describes
    neither -- rule 6. The counter is what keeps the drop visible.
    """
    short = _short_mix_cfg(mixed_cfg, tmp_path)
    v = VecWorld(short, seed=1, num_envs=4)
    actions = np.full((4, short.world.num_agents), IDLE, dtype=np.int64)
    for _ in range(4):
        out = v.step(actions)
        # every env's observations come back, mixed ones included
        assert out["obs"].shape[0] == 4
    assert v.mix_episodes == 2          # the two mixed envs finished an episode
    assert len(v.drain_episode_stats()) == 2   # only the two primary ones logged
