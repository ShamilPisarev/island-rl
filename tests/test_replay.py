"""Replay round-trip, schema versioning, and byte-level determinism."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sim.agents import ACTION_NAMES, IDLE, N_ACTIONS
from sim.policy import greedy_forager_actions
from sim.replay import (
    MANIFEST_NAME,
    SCHEMA_VERSION,
    ReplaySchemaError,
    agent_color,
    load_replay,
    record_episode,
    replay_to_arrays,
    update_manifest,
)


@pytest.fixture
def short_cfg(cfg):
    return cfg.replace(**{"world.max_ticks": 60})


def random_actor(seed: int, n_agents: int):
    rng = np.random.default_rng(seed)
    return lambda obs, mask=None: rng.integers(0, N_ACTIONS, size=obs.shape[0])


def test_recorded_replay_has_the_documented_shape(short_cfg):
    rec = record_episode(short_cfg, seed=1, act_fn=random_actor(0, 6),
                         label="t", source="fake")
    data = rec.to_dict()

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["action_names"] == list(ACTION_NAMES)
    assert data["tick_fields"]["agent"] == ["x", "z", "hunger", "food", "alive", "action"]
    assert len(data["bushes"]) == short_cfg.bushes.count
    assert len(data["agents"]) == short_cfg.world.num_agents

    # one snapshot before the first action, then one per step
    assert data["ticks"][0]["t"] == 0
    assert len(data["ticks"]) == data["ticks"][-1]["t"] + 1
    for i, tick in enumerate(data["ticks"]):
        assert tick["t"] == i
        assert len(tick["a"]) == short_cfg.world.num_agents
        assert all(len(row) == 6 for row in tick["a"])
        assert len(tick["b"]) == short_cfg.bushes.count


def test_initial_tick_reports_idle_for_everyone(short_cfg):
    rec = record_episode(short_cfg, seed=1, act_fn=random_actor(0, 6), label="t", source="fake")
    first = rec.to_dict()["ticks"][0]
    assert all(row[5] == IDLE for row in first["a"])


def test_round_trip_through_disk_is_lossless(short_cfg, tmp_path):
    rec = record_episode(short_cfg, seed=2, act_fn=random_actor(1, 6), label="rt", source="fake")
    path = rec.save(tmp_path / "r.json")
    loaded = load_replay(path)
    assert loaded == json.loads(json.dumps(rec.to_dict()))
    assert loaded["label"] == "rt"


def test_replay_matches_the_simulation_it_recorded(short_cfg):
    """The final tick of the replay must equal the world's true final state."""
    rec = record_episode(short_cfg, seed=3, act_fn=random_actor(2, 6), label="t", source="fake")
    last = rec.to_dict()["ticks"][-1]["a"]
    pool = rec.world.pool
    for i, row in enumerate(last):
        assert row[0] == pytest.approx(round(float(pool.x[i]), 2))
        assert row[1] == pytest.approx(round(float(pool.z[i]), 2))
        assert row[2] == pytest.approx(round(float(pool.hunger[i]), 1))
        assert row[3] == int(pool.food[i])
        assert row[4] == int(pool.alive[i])
    assert rec.to_dict()["ticks"][-1]["b"] == [int(v) for v in rec.world.bush_berries]


def test_arrays_view(short_cfg):
    rec = record_episode(short_cfg, seed=4, act_fn=random_actor(3, 6), label="t", source="fake")
    arrays = replay_to_arrays(rec.to_dict())
    t = len(rec.ticks)
    assert arrays["agents"].shape == (t, short_cfg.world.num_agents, 6)
    assert arrays["bushes"].shape == (t, short_cfg.bushes.count)


def test_alive_is_monotonic_and_dead_bodies_stop_moving(short_cfg):
    """Death is permanent, and a corpse stays where it fell so the viewer can
    render a marker rather than a twitching body."""
    dying = short_cfg.replace(**{"world.max_ticks": 400, "hunger.drain_per_tick": 1.0})
    rec = record_episode(dying, seed=5, act_fn=random_actor(4, 6), label="t", source="fake")
    agents = replay_to_arrays(rec.to_dict())["agents"]
    alive = agents[:, :, 4]
    assert (np.diff(alive, axis=0) <= 0).all()
    assert alive[-1].sum() < alive[0].sum()  # somebody actually died

    for i in range(dying.world.num_agents):
        dead_from = np.flatnonzero(alive[:, i] == 0)
        if dead_from.size:
            frozen = agents[dead_from[0]:, i, :2]
            assert np.allclose(frozen, frozen[0])


def test_version_mismatch_is_loud(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 99, "ticks": []}))
    with pytest.raises(ReplaySchemaError, match="not supported"):
        load_replay(path)

    path.write_text(json.dumps({"ticks": []}))
    with pytest.raises(ReplaySchemaError):
        load_replay(path)


def test_manifest_lists_replays_and_skips_itself(short_cfg, tmp_path):
    for i in range(3):
        record_episode(short_cfg, seed=i, act_fn=random_actor(i, 6),
                       label=f"run {i}", source="fake").save(tmp_path / f"r{i}.json")
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert {e["file"] for e in manifest["replays"]} == {"r0.json", "r1.json", "r2.json"}
    assert all(e["ticks"] > 0 for e in manifest["replays"])
    assert all("summary" in e for e in manifest["replays"])


def test_manifest_survives_a_corrupt_file(short_cfg, tmp_path):
    record_episode(short_cfg, seed=0, act_fn=random_actor(0, 6),
                   label="ok", source="fake").save(tmp_path / "ok.json")
    (tmp_path / "broken.json").write_text("{not json")
    update_manifest(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert {e["file"] for e in manifest["replays"]} == {"ok.json"}


def test_same_seed_produces_byte_identical_replays(short_cfg, tmp_path):
    """The determinism guarantee, end to end: same seed + same config = same file."""
    paths = []
    for name in ("a.json", "b.json"):
        rec = record_episode(short_cfg, seed=99, act_fn=random_actor(7, 6),
                             label="det", source="fake")
        paths.append(rec.save(tmp_path / name, update_manifest_file=False))
    a, b = (load_replay(p) for p in paths)
    for key in ("ticks", "bushes", "summary", "config", "seed"):
        assert a[key] == b[key]


def test_different_seed_diverges(short_cfg, tmp_path):
    a = record_episode(short_cfg, seed=1, act_fn=random_actor(7, 6), label="a", source="fake")
    b = record_episode(short_cfg, seed=2, act_fn=random_actor(7, 6), label="b", source="fake")
    assert a.to_dict()["ticks"] != b.to_dict()["ticks"]


def test_agent_colors_are_distinct(cfg):
    colors = {agent_color(i, cfg.world.num_agents) for i in range(cfg.world.num_agents)}
    assert len(colors) == cfg.world.num_agents
    assert all(c.startswith("#") and len(c) == 7 for c in colors)


def test_agent_colors_stay_distinct_at_a_hundred_agents():
    """Island 2.0 populations need colours that survive a crowd.

    Evenly spaced hues put agent 41 and agent 42 3.6 degrees apart, i.e. the
    same colour. The golden-angle walk keeps CONSECUTIVE ids far apart, and the
    saturation/value cycles run on different periods from the hue so a hue
    collision still differs in weight.
    """
    n = 100
    colors = [agent_color(i, n) for i in range(n)]
    assert len(set(colors)) == n, "duplicate colours in a 100-agent population"

    def rgb(c):
        return tuple(int(c[k:k + 2], 16) for k in (1, 3, 5))

    # Neighbouring ids are what a viewer sees side by side in the swatch grid.
    for i in range(n - 1):
        a, b = rgb(colors[i]), rgb(colors[i + 1])
        gap = sum(abs(x - y) for x, y in zip(a, b))
        assert gap > 90, f"agents {i} and {i + 1} are near-identical ({gap})"


def test_scripted_forager_beats_random_survival(cfg):
    """Not a replay test as such, but this is where the greedy forager first runs:
    if it cannot outlive a random walk, the observation is not sufficient and
    every later result would be meaningless."""
    rng = np.random.default_rng(0)

    def run(act_fn, seeds):
        return np.mean([
            record_episode(cfg, seed=s, act_fn=act_fn, label="x", source="fake")
            .world.stats().mean_lifespan
            for s in seeds
        ])

    seeds = range(4)
    greedy = run(lambda obs, mask=None: greedy_forager_actions(obs, cfg), seeds)
    random_life = run(lambda obs, mask=None: rng.integers(0, N_ACTIONS, size=obs.shape[0]), seeds)
    assert greedy > random_life * 1.2


def test_construction_replay_is_v2_with_material_blocks(cfg):
    """v2 only for construction worlds; v1 files stay valid forever."""
    from sim.config import load_config

    m4 = load_config("config/m4.yaml").replace(**{"world.max_ticks": 40})
    rec = record_episode(m4, seed=2, act_fn=random_actor(0, 6), label="v2", source="fake")
    d = rec.to_dict()
    assert d["schema_version"] == 2
    assert d["tick_fields"]["agent"] == ["x", "z", "hunger", "food", "alive", "action",
                                         "wood", "stone"]
    assert len(d["trees"]) == m4.construction.num_trees
    assert len(d["sites"]) == m4.construction.num_sites
    for tick in d["ticks"]:
        assert len(tick["a"][0]) == 8
        assert len(tick["w"]) == m4.construction.num_trees
        assert len(tick["s"]) == m4.construction.num_sites
    assert "night_cycle" in d["world"]

    # and a non-construction world still writes v1 with 6-column rows
    rec1 = record_episode(cfg.replace(**{"world.max_ticks": 20}), seed=3,
                          act_fn=random_actor(1, 6), label="v1", source="fake")
    d1 = rec1.to_dict()
    assert d1["schema_version"] == 1
    assert len(d1["ticks"][0]["a"][0]) == 6
    assert "trees" not in d1


def test_v2_world_block_carries_the_shelter_rules(cfg):
    """The viewer draws protection from these, so they must travel with the
    replay rather than being assumed."""
    from sim.config import load_config

    for name, expected in (("config/m4b.yaml", False), ("config/m4c.yaml", True)):
        m4 = load_config(name).replace(**{"world.max_ticks": 20})
        world = record_episode(m4, seed=1, act_fn=random_actor(0, 6),
                               label="w", source="fake").to_dict()["world"]
        assert world["partial_shelter"] is expected
        assert world["shelter_radius"] == m4.construction.shelter_radius
        assert world["night_cycle"] == m4.construction.night_cycle


# --- schema v5: goals, shocks, and who is learned ---------------------------
#
# Each of these is named after the thing that was invisible before it, because
# that is what the test is protecting: a replay that renders and answers no
# question is the failure mode, not a crash.

def _society_replay(config="config/island2/society4_ramp.yaml", ticks=120,
                    policy="utility", **kw):
    from sim.config import load_config
    from sim.society import make_runner
    from sim.replay import ReplayRecorder
    from sim.utility import ArbiterConfig
    from sim.world import World

    cfg = load_config(config).replace(**{"world.max_ticks": ticks})
    world = World(cfg, seed=10000)
    runner = make_runner(cfg, policy, 10000, ArbiterConfig(), kw.get("checkpoint"))
    rec = ReplayRecorder(world, cfg, label="t", source="test", seed=10000,
                         goal_source=runner,
                         learn_mask=getattr(runner.arbiter, "learn_mask", None))
    rec.snapshot()
    obs = world.observations()
    while True:
        res = world.step(runner.act(obs, world.action_mask()))
        obs = res.obs
        rec.snapshot()
        if res.episode_done:
            break
    return cfg, rec.to_dict()


def test_society_replay_is_v5_and_carries_a_goal_per_agent():
    """The action column says "NE"; only the goal says what for."""
    cfg, d = _society_replay()
    assert d["schema_version"] == 5
    from sim.utility import GOAL_NAMES
    assert d["goal_names"] == list(GOAL_NAMES)
    for tick in d["ticks"]:
        assert len(tick["o"]) == cfg.world.num_agents
        assert all(0 <= g < len(GOAL_NAMES) for g in tick["o"])


def test_a_storm_is_recorded_on_the_tick_it_lands():
    """It was previously readable only as a jump in `s` between two frames."""
    cfg, d = _society_replay()
    storms = [(t["t"], t["n"][1]) for t in d["ticks"] if "n" in t and t["n"][1]]
    assert storms, "the ramp world fires a shock every shock_interval ticks"
    for tick_index, hit in storms:
        assert hit > 0
        # ...and the site counters really did move on that tick, which is what
        # makes the flag a record rather than a decoration.
        before = sum(sum(pair) for pair in d["ticks"][tick_index - 1]["s"])
        after = sum(sum(pair) for pair in d["ticks"][tick_index]["s"])
        assert after > before


def test_a_calm_tick_carries_no_shock_key_at_all():
    """A world with shocks off must write a file the size of a v4 one."""
    _, d = _society_replay(config="config/island2/society4.yaml", ticks=40)
    # society4 fires its first shock at tick 100, so 40 ticks is all calm.
    assert all("n" not in tick for tick in d["ticks"])


def test_learn_flag_is_absent_unless_the_population_is_mixed():
    """A `learn` key on every agent of an all-scripted run would read as a
    result -- "these are the learned ones" -- where there is none."""
    _, d = _society_replay(ticks=40)
    assert all("learn" not in a for a in d["agents"])


def test_blight_is_flagged_while_it_lasts():
    # 280 rather than 200: the ramp world's first shock (tick 100) is a storm and
    # the blight lands on the second, so a 200-tick episode ends on the frame the
    # blight begins and the span has nowhere to show.
    cfg, d = _society_replay(ticks=280)
    blighted = [t["t"] for t in d["ticks"] if "n" in t and t["n"][0]]
    assert blighted, "the ramp world blights on one of its first two shocks"
    # A blight is a span, not an instant: the flag must persist tick by tick or
    # a watcher sees a single frame and nothing else.
    assert len(blighted) > 1
