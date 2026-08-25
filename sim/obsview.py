"""Named, block-structured access to the egocentric observation vector.

Every scripted controller in `sim/policy.py` re-derives the same column offsets
by hand, and the one time that was done positionally instead of by name the M4
world's own-materials columns landed where the forager expected bushes and it
quietly starved. This does it once, by name, and hands back real arrays.

Island 2.0 code uses this exclusively. `sim/policy.py` is deliberately NOT
refactored onto it: those controllers are the yardstick every 1.0 result was
measured against, and a shared-helper change that silently altered one of them
would invalidate numbers this project cannot re-measure cheaply.

Distances come back in WORLD UNITS (offsets are multiplied back up by
`distance_scale`), because every threshold worth comparing against -- gather
radius, build radius, steal radius -- is quoted in world units. The clip at
+-1 in the raw observation means anything past `distance_scale` reads as exactly
`distance_scale` away, and `clipped` says where that happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agents import (bush_channels, neighbour_channels, observation_layout,
                     site_channels)
from .config import Config


@dataclass(frozen=True)
class EntityBlock:
    """k-nearest entities of one kind, as (agents, k) arrays.

    ``present`` is the padding test. A zero-padded slot reads as "an entity with
    nothing in it at zero offset", so presence cannot be inferred from the offset
    alone -- for bushes and material nodes the magnitude channel is the tell, and
    for neighbours it is hunger (a living agent always has hunger > 0).
    """

    dx: np.ndarray
    dz: np.ndarray
    extra: tuple[np.ndarray, ...]
    present: np.ndarray

    @property
    def distance(self) -> np.ndarray:
        """World-unit distance to each slot; ``inf`` where absent."""
        d = np.hypot(self.dx, self.dz)
        return np.where(self.present, d, np.inf)

    def nearest(self, want: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray,
                                                              np.ndarray, np.ndarray]:
        """(distance, dx, dz, slot) of the nearest slot satisfying ``want``.

        Distance is ``inf`` and the offsets are 0 where nothing qualifies, so a
        caller can test finiteness once and use the offsets without a second
        guard.
        """
        d = self.distance
        if want is not None:
            d = np.where(want, d, np.inf)
        slot = np.argmin(d, axis=1)
        rows = np.arange(d.shape[0])
        best = d[rows, slot]
        ok = np.isfinite(best)
        return (best,
                np.where(ok, self.dx[rows, slot], 0.0),
                np.where(ok, self.dz[rows, slot], 0.0),
                slot)


class ObsView:
    """Parses one batch of observations into named blocks.

    Cheap: every block is a reshaped view or a slice, and the column lookup is
    built from ``observation_layout`` so an inserted channel moves the offsets
    rather than corrupting them.
    """

    def __init__(self, obs: np.ndarray, cfg: Config) -> None:
        self.cfg = cfg
        self.obs = obs
        self.n = obs.shape[0]
        self.scale = cfg.observation.distance_scale
        col = {name: i for i, name in enumerate(observation_layout(cfg))}
        self._col = col

        self.hunger = obs[:, col["own.hunger"]]
        self.food = obs[:, col["own.food"]]
        if cfg.construction.enabled:
            self.wood = obs[:, col["own.wood"]]
            self.stone = obs[:, col["own.stone"]]
            self.phase = obs[:, col["night.phase"]]
            self.is_night = obs[:, col["night.is_night"]] > 0.5
        else:
            zero = np.zeros(self.n)
            self.wood = zero
            self.stone = zero
            self.phase = zero
            self.is_night = np.zeros(self.n, dtype=bool)

        self.bushes = self._block("bush", cfg.observation.k_bushes,
                                  bush_channels(cfg), present_channel=0)
        self.neighbours = self._block("neighbour", cfg.observation.k_agents,
                                      neighbour_channels(cfg), present_channel=0)
        if cfg.construction.enabled:
            cc = cfg.construction
            self.trees = self._block("tree", cc.k_trees, 3, present_channel=0)
            self.rocks = self._block("rock", cc.k_rocks, 3, present_channel=0)
            # A site's magnitude channels all go to zero when it is COMPLETE, so
            # unlike every other entity its presence cannot be read off them.
            # A real site always has a non-zero offset (it is never exactly
            # underfoot to float precision); a padded slot is all zeros.
            self.sites = self._block("site", cc.k_sites, site_channels(cfg),
                                     present_channel=None)
        else:
            self.trees = self.rocks = self.sites = self._empty()

        # --- Island 2.0 stage 4. Absent columns come back as zeros/inf rather
        # than raising, so a stage-2 arbiter reads a stage-4 view unchanged and a
        # stage-4 arbiter running in a stage-2 world simply finds nothing to do.
        sc = cfg.society
        zero = np.zeros(self.n)
        if sc.enabled:
            self.stock_food = obs[:, col["own.stock_food"]]
            self.stock_material = obs[:, col["own.stock_material"]]
            self.home_dx = obs[:, col["home.dx"]] * self.scale
            self.home_dz = obs[:, col["home.dz"]] * self.scale
            self.home_complete = obs[:, col["home.complete"]] > 0.5
            self.raid_dx = obs[:, col["raid.dx"]] * self.scale
            self.raid_dz = obs[:, col["raid.dz"]] * self.scale
            self.raid_food = obs[:, col["raid.food"]]
            self.raid_material = obs[:, col["raid.material"]]
            self.blight = obs[:, col["shock.blight"]] > 0.5 if sc.observe_shock \
                else np.zeros(self.n, dtype=bool)
        else:
            self.stock_food = self.stock_material = zero
            self.home_dx = self.home_dz = zero
            self.home_complete = np.zeros(self.n, dtype=bool)
            self.raid_dx = self.raid_dz = zero
            self.raid_food = self.raid_material = zero
            self.blight = np.zeros(self.n, dtype=bool)

        self.edge_room = obs[:, col["edge.room"]]
        self.outward_x = obs[:, col["edge.outward_x"]]
        self.outward_z = obs[:, col["edge.outward_z"]]

    def _empty(self) -> EntityBlock:
        z = np.zeros((self.n, 0))
        return EntityBlock(dx=z, dz=z, extra=(), present=np.zeros((self.n, 0), dtype=bool))

    def _block(self, prefix: str, k: int, stride: int,
               present_channel: int | None) -> EntityBlock:
        start = self._col[f"{prefix}0.dx"]
        raw = self.obs[:, start:start + k * stride].reshape(self.n, k, stride)
        dx = raw[:, :, 0] * self.scale
        dz = raw[:, :, 1] * self.scale
        extra = tuple(raw[:, :, 2 + c] for c in range(stride - 2))
        if present_channel is None:
            present = (np.abs(raw[:, :, 0]) + np.abs(raw[:, :, 1])) > 0.0
        else:
            present = extra[present_channel] > 0.0
        return EntityBlock(dx=dx, dz=dz, extra=extra, present=present)

    # --- convenience derived quantities -----------------------------------

    @property
    def loaded_bushes(self) -> np.ndarray:
        """Per bush slot: does it hold at least one berry?"""
        return self.bushes.present

    @property
    def material_carried(self) -> np.ndarray:
        return self.wood + self.stone

    @property
    def site_complete(self) -> np.ndarray:
        """Per site slot: is it finished? (channel 4 of the site block.)"""
        if not self.sites.extra:
            return np.zeros((self.n, 0), dtype=bool)
        return (self.sites.extra[2] > 0.5) & self.sites.present

    @property
    def site_remaining(self) -> np.ndarray:
        """Per site slot: need_wood + need_stone, as observed fractions."""
        if not self.sites.extra:
            return np.zeros((self.n, 0))
        return self.sites.extra[0] + self.sites.extra[1]

    @property
    def neighbour_hunger(self) -> np.ndarray:
        return self.neighbours.extra[0] if self.neighbours.extra else np.zeros((self.n, 0))

    @property
    def neighbour_food(self) -> np.ndarray:
        """Neighbours' carried food, or zeros when the channel is off."""
        if not self.cfg.competition.observe_neighbour_food or not self.neighbours.extra:
            return np.zeros((self.n, self.cfg.observation.k_agents))
        return self.neighbours.extra[1]

    @property
    def home_distance(self) -> np.ndarray:
        """World-unit distance to my household's stockpile.

        A household always has one, so unlike every other target this is never
        ``inf`` -- there is nothing to test for existence.
        """
        return np.hypot(self.home_dx, self.home_dz)

    @property
    def raid_distance(self) -> np.ndarray:
        """Distance to the nearest FOREIGN stockpile; ``inf`` in a one-household world.

        A single household has no foreign pile, and the observation writes zeros
        there. Zero offsets would read as "a stockpile exactly underfoot", which
        is the padding trap the observation's own docstring warns about -- so the
        one-household case is detected from the config and reported as absent.
        """
        if not self.cfg.society.enabled or self.cfg.society.num_households < 2:
            return np.full(self.n, np.inf)
        return np.hypot(self.raid_dx, self.raid_dz)

    def _neighbour_society_col(self, want_grudge: bool) -> int:
        """Index into ``neighbours.extra`` of a stage-4 neighbour channel."""
        base = 1 + int(self.cfg.competition.observe_neighbour_food) \
            + 2 * int(self.cfg.exchange.observe_neighbour_materials)
        if want_grudge:
            return base + int(self.cfg.society.observe_household)
        return base

    @property
    def same_household(self) -> np.ndarray:
        """Per neighbour slot: is that neighbour in my household?

        All-false when the channel is off, which makes every stage-4 goal that
        depends on kinship simply unavailable rather than silently treating
        strangers as family.
        """
        sc = self.cfg.society
        if not (sc.enabled and sc.observe_household):
            return np.zeros((self.n, self.cfg.observation.k_agents), dtype=bool)
        return self.neighbours.extra[self._neighbour_society_col(False)] > 0.5

    @property
    def grudge(self) -> np.ndarray:
        """Per neighbour slot: how much I remember that neighbour taking from me."""
        sc = self.cfg.society
        if not (sc.enabled and sc.observe_grudge):
            return np.zeros((self.n, self.cfg.observation.k_agents))
        return self.neighbours.extra[self._neighbour_society_col(True)]

    @property
    def neighbour_material(self) -> np.ndarray:
        if not self.cfg.exchange.observe_neighbour_materials or len(self.neighbours.extra) < 4:
            return np.zeros((self.n, self.cfg.observation.k_agents))
        return self.neighbours.extra[2] + self.neighbours.extra[3]
