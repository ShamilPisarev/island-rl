// Island replay viewer.
//
// Loads a replay written by sim/replay.py and plays it back in 3D. It renders
// only what the replay says; it never simulates anything itself. If the file's
// schema_version is not one this build understands, it refuses to render rather
// than quietly drawing something plausible and wrong.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createPopulation } from './biped.js';

// --- replay schema this build can read (see sim/replay.py) ------------------
const SUPPORTED_SCHEMA = [1, 2, 3, 4, 5, 6];   // v2 = M4 materials/shelters/night, v3 = M5 transfers,
                                              // v4 = island2 stage 4 households/stockpiles/raids,
                                              // v5 = goals (`o`), shocks (`n`), learned-agent flags,
                                              // v6 = predator positions (`d`)
// Column order inside each tick's `a` rows. Cross-checked against the file's
// own tick_fields on load, so a schema change cannot silently shift a column.
const A_X = 0, A_Z = 1, A_HUNGER = 2, A_FOOD = 3, A_ALIVE = 4, A_ACTION = 5;
const A_WOOD = 6, A_STONE = 7;                      // v2 only
const EXPECTED_AGENT_FIELDS = ['x', 'z', 'hunger', 'food', 'alive', 'action'];
const EXPECTED_AGENT_FIELDS_V2 = [...EXPECTED_AGENT_FIELDS, 'wood', 'stone'];

const GATHER_COLOR = 0x6ec46e;
const STEAL_COLOR = 0xe0563c;
const GIFT_COLORS = [0x8ce08c, 0xe0b83c, 0xbfcbd8];   // food, wood, stone (v3)
const GIFT_FADE_TICKS = 10;   // how long a transfer line lingers after the tick
// Sized for 6 agents this was 32, which silently truncated once a 100-agent
// island started moving hundreds of units a tick -- and a picture that drops
// half the transfers without saying so is worse than one that says "+N more".
const MAX_GIFT_LINES = 256;
// A raid is drawn like a transfer but red and from the STOCKPILE to the raider,
// because that is the direction the food went. Kept in the same pool: raids and
// gifts are both "an event with no trace in the state either side of it", and one
// pool means one cap and one place the fade logic can be wrong.
const RAID_COLOR = 0xff3b30;
const RAID_FADE_TICKS = 20;   // longer than a gift: a raid is worth noticing
// A storm lands in a single tick. Held on screen for a beat afterwards for the
// same reason a raid line is: an event one frame wide is an event nobody sees.
const STORM_HOLD_TICKS = 25;
// Above this population the per-agent list stops being glanceable and the panel
// switches to a swatch grid plus aggregates. 6 agents fit; 100 do not.
const ROSTER_LIMIT = 24;
const BASE_TICKS_PER_SECOND = 12;   // playback rate at 1x
const SPEEDS = [0.5, 1, 4, 16];
const DEATH_FADE_TICKS = 14;        // ticks over which a corpse settles

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

const stage = $('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
scene.fog = new THREE.Fog(0x0d1117, 220, 460);

const camera = new THREE.PerspectiveCamera(48, 1, 0.5, 2000);
camera.position.set(0, 62, 92);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.49;   // don't let the camera go under the sea
controls.minDistance = 8;
controls.maxDistance = 320;

const hemi = new THREE.HemisphereLight(0xbcd8ff, 0x3d5a32, 0.75);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff0d5, 1.5);
sun.position.set(70, 110, 45);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0006;
scene.add(sun);

// Water: a big low-res plane with a cheap sine ripple animated on the CPU. No
// textures, so nothing else has to be fetched.
const waterGeo = new THREE.PlaneGeometry(1400, 1400, 48, 48);
waterGeo.rotateX(-Math.PI / 2);
const water = new THREE.Mesh(waterGeo, new THREE.MeshStandardMaterial({
  color: 0x2f86bd, roughness: 0.3, metalness: 0.15, transparent: true, opacity: 0.95,
  flatShading: true,
}));
water.position.y = -1.6;
water.receiveShadow = true;
scene.add(water);
const waterBase = Float32Array.from(waterGeo.attributes.position.array);

// Everything that depends on the loaded replay lives under here, so loading a
// new file is one dispose-and-rebuild rather than a pile of nullable globals.
let worldGroup = new THREE.Group();
scene.add(worldGroup);

function disposeGroup(group) {
  group.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
  });
  scene.remove(group);
}

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

// ---------------------------------------------------------------------------
// Replay state
// ---------------------------------------------------------------------------

const state = {
  replay: null,
  ticks: [],
  lastTick: 0,
  t: 0,               // fractional tick position
  playing: true,
  speed: 1,
  follow: -1,
  population: null,   // instanced biped rig (viewer/biped.js)
  agentCount: 0,
  deathTicks: null,   // Float64Array, one entry per agent
  headings: null,     // last known facing, held while an agent stands still
  agentPos: null,     // interpolated x,z pairs, for the follow camera and picking
  bushes: [],         // { mesh, base }
  trees: [],
  rocks: [],
  sites: [],
  stockpiles: [],     // one per household (schema v4)
  households: [],     // { x, z, color } -- static for the episode
  society: false,
  actionNames: [],
  construction: false,
  exchange: false,
  giftLines: [],      // pooled line segments, reused every frame
  hover: -1,          // agent under the cursor, or -1
  rows: [],           // cached panel elements, per agent
  rosterMode: true,   // per-agent list (few agents) vs population view (many)
  // --- schema v5
  goalNames: null,    // index -> label, or null when the replay carries no goals
  learn: null,        // Uint8Array, 1 where a learned arbiter drives that agent
  learnCount: 0,
  learnRings: [],     // one ring mesh per learned agent (there are ~20, not 100)
  predators: [],      // one mesh per predator (schema v6)
};

function fatal(title, message) {
  $('fatalTitle').textContent = title;
  $('fatalMsg').textContent = message;
  $('fatal').style.display = 'grid';
  state.playing = false;
}
function clearFatal() { $('fatal').style.display = 'none'; }

function hasMaterials(replay) {
  return replay?.config?.construction?.enabled ?? (replay?.schema_version >= 2);
}

function validate(replay, origin) {
  const v = replay?.schema_version;
  if (!SUPPORTED_SCHEMA.includes(v)) {
    throw new Error(
      `${origin}\n\nReplay schema version: ${JSON.stringify(v)}\n` +
      `This viewer understands: ${SUPPORTED_SCHEMA.join(', ')}\n\n` +
      `Refusing to render. Regenerate the replay with the current sim/, or check ` +
      `out the viewer revision that matches it.`
    );
  }
  const fields = replay?.tick_fields?.agent;
  // The material columns follow whether the world had construction, not the
  // schema number: a v3 (exchange) replay from a world without construction
  // carries the v1 agent columns plus per-tick transfers.
  const expected = hasMaterials(replay) ? EXPECTED_AGENT_FIELDS_V2 : EXPECTED_AGENT_FIELDS;
  if (!Array.isArray(fields) || fields.length !== expected.length
      || expected.some((f, i) => fields[i] !== f)) {
    throw new Error(
      `${origin}\n\nAgent column layout is ${JSON.stringify(fields)}, expected ` +
      `${JSON.stringify(expected)}.\n\nRefusing to render: the columns ` +
      `would be misread.`
    );
  }
  if (!Array.isArray(replay.ticks) || replay.ticks.length === 0) {
    throw new Error(`${origin}\n\nReplay contains no ticks.`);
  }
}

// ---------------------------------------------------------------------------
// Building the scene from a replay
// ---------------------------------------------------------------------------

const ISLAND_TOP = 0;

function buildIsland(radius) {
  const g = new THREE.Group();

  const land = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius * 0.94, 6, 44, 1),
    new THREE.MeshStandardMaterial({ color: 0x5b8f4a, roughness: 0.95, flatShading: true }),
  );
  land.position.y = ISLAND_TOP - 3;
  land.receiveShadow = true;
  g.add(land);

  // A wider, shallower disc just below the land reads as a beach/shelf and stops
  // the island looking like a cylinder dropped into a bathtub.
  const shelf = new THREE.Mesh(
    new THREE.CylinderGeometry(radius * 1.09, radius * 1.16, 2.4, 44, 1),
    new THREE.MeshStandardMaterial({ color: 0xd9c58c, roughness: 1.0, flatShading: true }),
  );
  shelf.position.y = ISLAND_TOP - 4.6;
  shelf.receiveShadow = true;
  g.add(shelf);

  // Scatter a few dark rocks for parallax, so orbiting the camera reads as motion.
  const rockGeo = new THREE.DodecahedronGeometry(1, 0);
  const rockMat = new THREE.MeshStandardMaterial({ color: 0x6c7480, roughness: 1, flatShading: true });
  for (let i = 0; i < 26; i++) {
    const a = (i / 26) * Math.PI * 2 * 3.7;
    const r = radius * (0.15 + 0.8 * ((i * 0.37) % 1));
    const rock = new THREE.Mesh(rockGeo, rockMat);
    rock.position.set(Math.sin(a) * r, ISLAND_TOP - 0.2, Math.cos(a) * r);
    rock.scale.setScalar(0.5 + ((i * 0.61) % 1) * 0.9);
    rock.rotation.set(i, i * 2.1, i * 0.7);
    rock.castShadow = true;
    rock.receiveShadow = true;
    g.add(rock);
  }
  return g;
}

function buildBush(x, z) {
  const group = new THREE.Group();
  group.position.set(x, ISLAND_TOP, z);

  const trunk = new THREE.Mesh(
    new THREE.CylinderGeometry(0.16, 0.26, 1.0, 5),
    new THREE.MeshStandardMaterial({ color: 0x4a3524, roughness: 1, flatShading: true }),
  );
  trunk.position.y = 0.5;
  trunk.castShadow = true;
  group.add(trunk);

  const foliage = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1, 0),
    new THREE.MeshStandardMaterial({ color: 0x3f8f45, roughness: 0.85, flatShading: true }),
  );
  foliage.position.y = 1.5;
  foliage.castShadow = true;
  foliage.receiveShadow = true;
  group.add(foliage);

  return { group, foliage };
}

function buildTree(x, z) {
  const group = new THREE.Group();
  group.position.set(x, ISLAND_TOP, z);
  const trunk = new THREE.Mesh(
    new THREE.CylinderGeometry(0.28, 0.4, 2.2, 6),
    new THREE.MeshStandardMaterial({ color: 0x5d4327, roughness: 1, flatShading: true }),
  );
  trunk.position.y = 1.1;
  trunk.castShadow = true;
  group.add(trunk);
  const crown = new THREE.Mesh(
    new THREE.ConeGeometry(1.7, 3.6, 7),
    new THREE.MeshStandardMaterial({ color: 0x2f6b34, roughness: 0.9, flatShading: true }),
  );
  crown.position.y = 3.6;
  crown.castShadow = true;
  group.add(crown);
  return { group, crown, trunk };
}

function buildRock(x, z) {
  const mesh = new THREE.Mesh(
    new THREE.DodecahedronGeometry(1.5, 0),
    new THREE.MeshStandardMaterial({ color: 0x9aa7b8, roughness: 0.9, flatShading: true }),
  );
  mesh.position.set(x, ISLAND_TOP + 0.5, z);
  mesh.rotation.set(0.4, 0.9, 0.2);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return { mesh };
}

function buildSite(x, z, shelterRadius) {
  const group = new THREE.Group();
  group.position.set(x, ISLAND_TOP, z);

  // foundation ring: always visible, marks the buildable spot
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(2.0, 0.16, 6, 18),
    new THREE.MeshStandardMaterial({ color: 0x8b8378, roughness: 1, flatShading: true }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.1;
  ring.receiveShadow = true;
  group.add(ring);

  // walls rise with progress
  const walls = new THREE.Mesh(
    new THREE.CylinderGeometry(1.8, 1.9, 1.0, 8, 1, true),
    new THREE.MeshStandardMaterial({ color: 0xb08d57, roughness: 0.95, flatShading: true,
                                     side: THREE.DoubleSide }),
  );
  walls.castShadow = true;
  group.add(walls);

  // roof appears on completion
  const roof = new THREE.Mesh(
    new THREE.ConeGeometry(2.4, 1.6, 8),
    new THREE.MeshStandardMaterial({ color: 0x7a5230, roughness: 0.9, flatShading: true }),
  );
  roof.castShadow = true;
  roof.visible = false;
  group.add(roof);

  // faint protection halo, shown for completed shelters at night
  const halo = new THREE.Mesh(
    new THREE.RingGeometry(shelterRadius - 0.25, shelterRadius, 36),
    new THREE.MeshBasicMaterial({ color: 0xffd98a, transparent: true, opacity: 0.0,
                                  side: THREE.DoubleSide }),
  );
  halo.rotation.x = -Math.PI / 2;
  halo.position.y = 0.08;
  group.add(halo);

  // warm light for the hearth at night
  const lamp = new THREE.PointLight(0xffc36b, 0.0, shelterRadius * 2.2, 1.8);
  lamp.position.y = 2.2;
  group.add(lamp);

  return { group, walls, roof, halo, lamp };
}

// A household's stockpile: a crate whose lid rises with what is in it, ringed in
// the household's colour. Two stacks side by side rather than one blended bar,
// because food and material are two economies and a household can be rich in one
// and empty in the other -- which is exactly the state a relay or a raid is about.
function buildStockpile(x, z, color) {
  const group = new THREE.Group();
  group.position.set(x, ISLAND_TOP, z);

  const band = new THREE.Mesh(
    new THREE.TorusGeometry(3.1, 0.2, 6, 20),
    new THREE.MeshStandardMaterial({ color: new THREE.Color(color), roughness: 0.8,
                                     flatShading: true }),
  );
  band.rotation.x = -Math.PI / 2;
  band.position.y = 0.12;
  group.add(band);

  function stack(dx, hex) {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(1.1, 1.0, 1.1),
      new THREE.MeshStandardMaterial({ color: hex, roughness: 0.9, flatShading: true }),
    );
    mesh.position.set(dx, 0.5, 2.4);
    mesh.castShadow = true;
    group.add(mesh);
    return mesh;
  }
  // Same hues the gift arcs use for the same items, so a berry looks like a berry
  // whether it is in a pile, in a hand, or in mid-air.
  const food = stack(-0.7, GIFT_COLORS[0]);
  const material = stack(0.7, GIFT_COLORS[1]);
  return { group, band, food, material };
}

// Which colour the head pip takes for each action, so what a crowd is DOING is
// readable without selecting anybody. Built once, not per agent per frame.
const ACTION_PIP = {
  gather: GATHER_COLOR, steal: STEAL_COLOR,
  chop: 0x8a5a2b, mine: 0x9aa7b8, build: 0xe0b83c,
  give_food: 0x8ce08c, give_material: 0xe0b83c,
  deposit_food: 0x8ce08c, deposit_material: 0xe0b83c,
  withdraw_food: 0x8ce08c, withdraw_material: 0xe0b83c,
  raid: RAID_COLOR,
};

function loadReplay(replay, origin) {
  validate(replay, origin);

  disposeGroup(worldGroup);
  worldGroup = new THREE.Group();
  scene.add(worldGroup);

  state.replay = replay;
  state.ticks = replay.ticks;
  state.lastTick = replay.ticks.length - 1;
  state.t = 0;
  state.follow = -1;
  state.playing = true;
  state.actionNames = replay.action_names || [];

  const radius = replay.world.island_radius;
  worldGroup.add(buildIsland(radius));

  sun.shadow.camera.left = -radius * 1.4;
  sun.shadow.camera.right = radius * 1.4;
  sun.shadow.camera.top = radius * 1.4;
  sun.shadow.camera.bottom = -radius * 1.4;
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 400;
  sun.shadow.camera.updateProjectionMatrix();

  state.bushes = replay.bushes.map((b) => {
    const built = buildBush(b.x, b.z);
    worldGroup.add(built.group);
    return built;
  });

  state.construction = hasMaterials(replay);
  state.exchange = replay.schema_version >= 3;
  state.giftLines = [];   // the old pool went with the disposed worldGroup
  state.trees = [];
  state.rocks = [];
  state.sites = [];
  if (state.construction) {
    state.trees = (replay.trees || []).map((e) => {
      const built = buildTree(e.x, e.z);
      worldGroup.add(built.group);
      return built;
    });
    state.rocks = (replay.rocks || []).map((e) => {
      const built = buildRock(e.x, e.z);
      worldGroup.add(built.mesh);
      return built;
    });
    state.sites = (replay.sites || []).map((e) => {
      const built = buildSite(e.x, e.z, replay.world.shelter_radius || 6);
      worldGroup.add(built.group);
      return built;
    });
  }

  state.society = replay.schema_version >= 4;
  state.stockpiles = [];
  state.households = replay.households || [];
  if (state.society) {
    state.stockpiles = state.households.map((h) => {
      const built = buildStockpile(h.x, h.z, h.color || '#ffffff');
      worldGroup.add(built.group);
      return built;
    });
  }

  // --- schema v5: goals, and who is learned.
  // Both are read from the file rather than derived from the version, because a
  // society world always carries the shock channel but only carries goals when an
  // arbiter drove it, and only carries `learn` when the population was mixed.
  state.goalNames = Array.isArray(replay.goal_names) ? replay.goal_names : null;
  state.learn = null;
  state.learnCount = 0;
  state.learnRings = [];
  if (replay.agents.some((a) => 'learn' in a)) {
    state.learn = Uint8Array.from(replay.agents, (a) => (a.learn ? 1 : 0));
    state.learnCount = state.learn.reduce((n, v) => n + v, 0);
    // A ring on the ground under each learned agent. One mesh per learned agent
    // rather than an instanced attribute: the mixed runs put a learned agent in
    // one slot per household, so this is ~20 meshes, not 100.
    const ringGeo = new THREE.RingGeometry(1.5, 2.1, 20);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x6fd3ff, transparent: true, opacity: 0.75, side: THREE.DoubleSide,
    });
    for (let i = 0; i < state.learn.length; i++) {
      if (!state.learn[i]) continue;
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = ISLAND_TOP + 0.12;
      worldGroup.add(ring);
      state.learnRings.push({ ring, agent: i });
    }
  }

  // --- schema v6: the predators. A handful of meshes, not an instanced rig --
  // there are twelve of them, and a thing that hunts should look different from
  // the people it hunts rather than being a differently-coloured agent.
  state.predators = [];
  if (Array.isArray(replay.ticks[0]?.d)) {
    const reach = replay.world?.predator_attack_radius ?? 6;
    for (let k = 0; k < replay.ticks[0].d.length; k++) {
      const g = new THREE.Group();
      const body = new THREE.Mesh(
        new THREE.ConeGeometry(1.5, 4.5, 5),
        new THREE.MeshStandardMaterial({ color: 0x8a1420, roughness: 0.6, flatShading: true }),
      );
      body.position.y = ISLAND_TOP + 2.2;
      body.rotation.x = Math.PI;   // a snout, not a party hat
      g.add(body);
      // The reach it actually kills inside, drawn rather than guessed at: a
      // hazard whose radius you cannot see is a hazard you cannot learn to read.
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(reach - 0.35, reach, 32),
        new THREE.MeshBasicMaterial({
          color: 0xff3b30, transparent: true, opacity: 0.22, side: THREE.DoubleSide,
        }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = ISLAND_TOP + 0.1;
      g.add(ring);
      worldGroup.add(g);
      state.predators.push({ group: g, body, ring });
    }
  }

  // One instanced rig for the whole population. Replaces ~13 meshes per agent.
  state.population = createPopulation(
    replay.agents.length,
    replay.agents.map((a) => a.color),
  );
  worldGroup.add(state.population.group);
  state.deathTicks = deathTicks(replay);
  state.agentCount = replay.agents.length;
  state.headings = new Float64Array(state.agentCount);
  state.agentPos = new Float64Array(state.agentCount * 2);

  camera.position.set(0, radius * 1.55, radius * 2.3);
  controls.target.set(0, 0, 0);
  controls.update();

  buildAgentPanel();
  renderLegend();
  renderSummary();
  $('replayLabel').textContent = `${replay.label ?? 'replay'} · ${replay.source ?? '?'} · seed ${replay.seed}`;
  $('scrub').max = String(state.lastTick);
  $('scrub').value = '0';
  clearFatal();
  $('loading').style.display = 'none';
  updatePlayButton();
  applyTick(0);
}

// Every agent's death tick in ONE pass over the replay. The old version scanned
// all ticks once per agent, which is fine at 6 and a visible load stall at 100.
function deathTicks(replay) {
  const n = replay.agents.length;
  const out = new Float64Array(n).fill(Infinity);
  let remaining = n;
  for (let t = 0; t < replay.ticks.length && remaining > 0; t++) {
    const rows = replay.ticks[t].a;
    for (let i = 0; i < n; i++) {
      if (out[i] === Infinity && rows[i][A_ALIVE] === 0) {
        out[i] = t;
        remaining--;
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Applying a (fractional) tick to the scene
// ---------------------------------------------------------------------------

const _tmpVec = new THREE.Vector3();
const _tmpQuat = new THREE.Quaternion();

function applyTick(t) {
  const i0 = Math.min(Math.floor(t), state.lastTick);
  const i1 = Math.min(i0 + 1, state.lastTick);
  const f = i1 > i0 ? t - i0 : 0;
  const cur = state.ticks[i0], nxt = state.ticks[i1];
  const world = state.replay.world;

  // Night, needed below to decide whether a still agent is sitting out the dark
  // rather than merely standing about. Derived from the clock the replay already
  // carries, so no schema change was needed for animation state (design doc
  // section 5 asked that this be checked before bumping SCHEMA_VERSION -- the
  // action column plus the cycle is enough, so it was not bumped).
  const nightNow = state.construction && nightFactor(i0) > 0.5;
  const pop = state.population;
  for (let i = 0; i < state.agentCount; i++) {
    const a0 = cur.a[i], a1 = nxt.a[i];
    const alive = a0[A_ALIVE] === 1;

    const x = a0[A_X] + (a1[A_X] - a0[A_X]) * f;
    const z = a0[A_Z] + (a1[A_Z] - a0[A_Z]) * f;

    // Corpses settle over a few ticks and then stay put, computed from the tick
    // index rather than accumulated over frames, so scrubbing stays consistent.
    const dead = Math.min(Math.max((t - state.deathTicks[i]) / DEATH_FADE_TICKS, 0), 1);

    const dx = a1[A_X] - a0[A_X], dz = a1[A_Z] - a0[A_Z];
    const speed = Math.hypot(dx, dz);
    if (speed > 1e-3) state.headings[i] = Math.atan2(dx, dz);
    state.agentPos[i * 2] = x;
    state.agentPos[i * 2 + 1] = z;
    const action = alive ? state.actionNames[a0[A_ACTION]] : undefined;

    pop.setPose(i, {
      x, z,
      heading: state.headings[i],
      speed: alive ? speed : 0,
      // The fractional tick is the only clock, so a scrubbed pose is stable.
      // Offset per agent so a crowd does not march in lockstep.
      phase: t + i * 0.7,
      action,
      pipColor: ACTION_PIP[action],
      food: alive ? a0[A_FOOD] : 0,
      dead,
      resting: nightNow && (action === 'idle' || speed <= 1e-3),
    });
  }
  pop.commit();

  for (let b = 0; b < state.bushes.length; b++) {
    const frac = cur.b[b] / world.bush_capacity;
    const next = nxt.b[b] / world.bush_capacity;
    const v = frac + (next - frac) * f;
    const bush = state.bushes[b];
    bush.foliage.scale.setScalar(0.42 + v * 0.95);
    // Full = lush green, empty = bare twiggy brown, so depletion and regrowth
    // are legible from across the island without reading any numbers.
    bush.foliage.material.color.setRGB(0.25 + (1 - v) * 0.28, 0.24 + v * 0.36, 0.13 + (1 - v) * 0.08);
  }

  if (state.construction) {
    const wcfg = state.replay.world;
    for (let k = 0; k < state.trees.length; k++) {
      const frac = (cur.w?.[k] ?? 0) / (wcfg.tree_wood || 1);
      const tree = state.trees[k];
      tree.crown.scale.setScalar(0.25 + 0.75 * frac);
      tree.crown.position.y = 2.2 + 1.4 * frac;
      tree.crown.material.color.setRGB(0.18 + (1 - frac) * 0.25,
                                       0.42 - (1 - frac) * 0.15,
                                       0.2 - (1 - frac) * 0.05);
    }
    for (let k = 0; k < state.rocks.length; k++) {
      const frac = (cur.r?.[k] ?? 0) / (wcfg.rock_stone || 1);
      state.rocks[k].mesh.scale.setScalar(0.35 + 0.65 * frac);
    }
    const night = nightFactor(t);
    for (let k = 0; k < state.sites.length; k++) {
      const [wNeed, sNeed] = cur.s?.[k] ?? [1, 1];
      const total = (wcfg.site_wood_cost || 4) + (wcfg.site_stone_cost || 2);
      const progress = 1 - (wNeed + sNeed) / total;
      const done = wNeed === 0 && sNeed === 0;
      const site = state.sites[k];
      site.walls.visible = progress > 0;
      site.walls.scale.y = Math.max(progress, 0.05);
      site.walls.position.y = 1.25 * Math.max(progress, 0.05);   // grow upward
      site.roof.visible = done;
      site.roof.position.y = 2.5 * progress + 0.9;
      // Halo and hearth show how much protection this site actually gives. With
      // partial_shelter a half-built wall really does shelter you halfway, so
      // drawing nothing until completion would be the viewer telling a lie.
      const protection = wcfg.partial_shelter ? progress : (done ? 1 : 0);
      site.halo.material.opacity = night * 0.35 * protection;
      site.lamp.intensity = night * 1.4 * protection;
    }
    applyNight(night);
  } else {
    applyNight(0);
  }

  if (state.society) {
    const cap = state.replay.world;
    for (let k = 0; k < state.stockpiles.length; k++) {
      const [f, m] = cur.p?.[k] ?? [0, 0];
      const pile = state.stockpiles[k];
      // Floor the visible height rather than hiding an empty pile: a crate that
      // vanishes reads as "no household here", and an empty larder is the most
      // interesting state a household can be in.
      const fy = Math.max(f / (cap.stockpile_food_capacity || 12), 0.04);
      const my = Math.max(m / (cap.stockpile_material_capacity || 12), 0.04);
      pile.food.scale.y = fy * 3.0;
      pile.food.position.y = fy * 1.5;
      pile.material.scale.y = my * 3.0;
      pile.material.position.y = my * 1.5;
    }
  }

  if (state.exchange || state.society) applyTransfers(t);

  // Learned agents wear a ring (schema v5). Positions come from the same
  // interpolated buffer the follow camera uses, so the ring never lags the body.
  for (let k = 0; k < state.learnRings.length; k++) {
    const { ring, agent } = state.learnRings[k];
    const alive = cur.a[agent][A_ALIVE] === 1;
    ring.visible = alive;
    if (alive) {
      ring.position.x = state.agentPos[agent * 2];
      ring.position.z = state.agentPos[agent * 2 + 1];
    }
  }

  // Predators (schema v6). Interpolated between ticks like everything else, and
  // dimmed by day: they walk home at dawn and stop being a threat, and a
  // watcher should be able to see that without reading the clock.
  if (state.predators.length && Array.isArray(cur.d)) {
    const nxt = state.ticks[Math.min(i0 + 1, state.lastTick)];
    const f = t - i0;
    const hunting = state.construction ? nightFactor(t) > 0.5 : false;
    for (let k = 0; k < state.predators.length; k++) {
      const a = cur.d[k], b = nxt.d?.[k] ?? a;
      const p = state.predators[k];
      p.group.position.x = a[0] + (b[0] - a[0]) * f;
      p.group.position.z = a[1] + (b[1] - a[1]) * f;
      p.ring.visible = hunting;
      p.body.material.opacity = hunting ? 1 : 0.45;
      p.body.material.transparent = !hunting;
    }
  }

  applyShock(cur, i0);
  updateAgentPanel(cur);
  const night = state.construction && nightFactor(t) > 0.5;
  $('tickCount').textContent = `${night ? '\u263e ' : ''}${i0} / ${state.lastTick}`;
  $('scrub').value = String(t);
}

// --- shocks (schema v5) -----------------------------------------------------
//
// `n` is [blight, storm_sites], present only on ticks where something is
// happening. A blight is a STATE (berry regrowth is suspended, and the only
// visible consequence is bushes that quietly stop refilling); a storm is an
// EVENT that lands in one tick and is legible only as a jump in the site
// counters between two frames. Neither was visible before, so both get an
// explicit cue rather than being left to inference.
function applyShock(tick, tickIndex) {
  const box = $('shock'), label = $('shockLabel');
  if (!box || !label) return;
  const blight = Array.isArray(tick.n) && tick.n[0] > 0;
  // Scanned backwards over the hold window rather than remembered in state:
  // the scrubber can jump anywhere, and a remembered "last storm" shows a stale
  // banner when you drag forward and no banner at all when you drag back onto an
  // earlier one. Twenty-five array lookups a frame is nothing.
  let hit = 0;
  for (let k = 0; k < STORM_HOLD_TICKS; k++) {
    const idx = tickIndex - k;
    if (idx < 0) break;
    const n = state.ticks[idx]?.n;
    if (Array.isArray(n) && n[1] > 0) { hit = n[1]; break; }
  }
  const storming = hit > 0;
  box.classList.toggle('storm', storming);
  box.classList.toggle('blight', blight && !storming);
  label.classList.toggle('on', storming || blight);
  label.classList.toggle('storm', storming);
  label.classList.toggle('blight', blight && !storming);
  if (storming) label.textContent = `storm — ${hit} shelters hit`;
  else if (blight) label.textContent = 'blight — no berries are regrowing';
}

// --- transfers (schema v3) --------------------------------------------------
//
// A transfer leaves no trace in the state either side of it -- two inventories
// change and nothing says why -- so the replay lists them per tick and this
// draws each one as a short-lived line between where the two agents stood.
// Anchored to those positions rather than to the agents, so a gift stays where
// it happened instead of being dragged around afterwards.

// A transfer covers at most `give_radius` (2.5 units on a 40-unit island), so a
// straight line between the two agents is shorter than an agent is wide and
// disappears into their bodies. Drawn as a lobbed arc instead, with the item
// itself riding along it: legible at island zoom, and it shows the direction,
// which a symmetric line does not.
const GIFT_ARC_POINTS = 14;
const GIFT_ARC_HEIGHT = 3.0;

function giftArcPoint(ax, az, bx, bz, u) {
  return [
    ax + (bx - ax) * u,
    ISLAND_TOP + 2.0 + GIFT_ARC_HEIGHT * 4 * u * (1 - u),   // parabola, 0 at both ends
    az + (bz - az) * u,
  ];
}

function ensureGiftLines() {
  if (state.giftLines.length) return;
  const pip = new THREE.SphereGeometry(0.48, 10, 8);
  for (let i = 0; i < MAX_GIFT_LINES; i++) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position',
      new THREE.BufferAttribute(new Float32Array(GIFT_ARC_POINTS * 3), 3));
    const arc = new THREE.Line(geometry, new THREE.LineBasicMaterial({
      color: 0xffffff, transparent: true, opacity: 0, depthTest: false,
    }));
    arc.frustumCulled = false;
    const item = new THREE.Mesh(pip, new THREE.MeshBasicMaterial({
      color: 0xffffff, transparent: true, opacity: 0, depthTest: false,
    }));
    item.frustumCulled = false;
    arc.visible = item.visible = false;
    worldGroup.add(arc);
    worldGroup.add(item);
    state.giftLines.push({ arc, item });
  }
}

// One arc, drawn from (ax, az) to (bx, bz) with an item riding the first half.
// Shared by gifts and raids because the only difference between them is where the
// two ends are and what colour it is.
function drawArc(slot, ax, az, bx, bz, color, age) {
  const p = slot.arc.geometry.attributes.position.array;
  for (let j = 0; j < GIFT_ARC_POINTS; j++) {
    const [x, y, z] = giftArcPoint(ax, az, bx, bz, j / (GIFT_ARC_POINTS - 1));
    p[j * 3] = x; p[j * 3 + 1] = y; p[j * 3 + 2] = z;
  }
  slot.arc.geometry.attributes.position.needsUpdate = true;
  slot.arc.geometry.computeBoundingSphere();
  slot.arc.material.color.setHex(color);
  slot.arc.material.opacity = 0.55 * (1 - age);
  slot.arc.visible = true;
  const [x, y, z] = giftArcPoint(ax, az, bx, bz, Math.min(age * 2, 1));
  slot.item.position.set(x, y, z);
  slot.item.material.color.setHex(color);
  slot.item.material.opacity = 1 - age;
  slot.item.visible = true;
}

// Raids (schema v4). Drawn FROM the stockpile TO the raider, which is the
// direction the food went -- the opposite convention to a gift, where the arc
// starts at the giver. Both are "who ended up with it", and getting that backwards
// would make a raid look like a delivery.
function applyRaids(t, used) {
  const i0 = Math.min(Math.floor(t), state.lastTick);
  for (let k = Math.max(0, i0 - RAID_FADE_TICKS); k <= i0 && used < MAX_GIFT_LINES; k++) {
    const list = state.ticks[k].k;
    if (!list) continue;
    const age = Math.min(Math.max((t - k) / RAID_FADE_TICKS, 0), 1);
    const rows = state.ticks[k].a;
    for (const [raider, house] of list) {
      if (used >= MAX_GIFT_LINES) break;
      const home = state.households[house];
      if (!home) continue;
      drawArc(state.giftLines[used++], home.x, home.z,
              rows[raider][A_X], rows[raider][A_Z], RAID_COLOR, age);
    }
  }
  return used;
}

function applyTransfers(t) {
  ensureGiftLines();
  const i0 = Math.min(Math.floor(t), state.lastTick);
  let used = 0;
  if (state.society) used = applyRaids(t, used);
  for (let k = Math.max(0, i0 - GIFT_FADE_TICKS); k <= i0 && used < MAX_GIFT_LINES; k++) {
    const list = state.ticks[k].g;
    if (!list) continue;
    const age = Math.min(Math.max((t - k) / GIFT_FADE_TICKS, 0), 1);
    const rows = state.ticks[k].a;
    for (const [giver, receiver, item] of list) {
      if (used >= MAX_GIFT_LINES) break;
      const slot = state.giftLines[used++];
      const [ax, az] = [rows[giver][A_X], rows[giver][A_Z]];
      const [bx, bz] = [rows[receiver][A_X], rows[receiver][A_Z]];
      drawArc(slot, ax, az, bx, bz, GIFT_COLORS[item] ?? 0xffffff, age);
    }
  }
  for (let i = used; i < state.giftLines.length; i++) {
    state.giftLines[i].arc.visible = false;
    state.giftLines[i].item.visible = false;
  }
}

// Night as a 0..1 factor with a soft edge either side of the boundary, derived
// from the tick index and the cycle constants in the replay's world block --
// never from a per-tick flag that could disagree with them.
function nightFactor(t) {
  const w = state.replay.world;
  if (!w.night_cycle) return 0;
  const phase = (t % w.night_cycle) / w.night_cycle;
  const nightStart = 1 - (w.night_fraction || 0.25);
  const soft = 0.02;   // ~4 ticks of dusk on a 200-tick cycle
  const into = (phase - nightStart) / soft;
  const outof = (1 - phase) / soft;
  return Math.max(0, Math.min(1, into, outof));
}

const DAY = {
  sun: 1.5, sunColor: new THREE.Color(0xfff0d5),
  hemi: 0.75, water: new THREE.Color(0x2f86bd), bg: new THREE.Color(0x0d1117),
};
const NIGHT = {
  sun: 0.22, sunColor: new THREE.Color(0x7fa3d8),
  hemi: 0.18, water: new THREE.Color(0x123049), bg: new THREE.Color(0x05070c),
};

function applyNight(f) {
  sun.intensity = DAY.sun + (NIGHT.sun - DAY.sun) * f;
  sun.color.copy(DAY.sunColor).lerp(NIGHT.sunColor, f);
  hemi.intensity = DAY.hemi + (NIGHT.hemi - DAY.hemi) * f;
  water.material.color.copy(DAY.water).lerp(NIGHT.water, f);
  scene.background.copy(DAY.bg).lerp(NIGHT.bg, f);
  scene.fog.color.copy(scene.background);
}

// ---------------------------------------------------------------------------
// Side panel
// ---------------------------------------------------------------------------

// The side panel has two modes, because one design cannot serve both ends of the
// population range. Up to ROSTER_LIMIT agents it is the per-agent list the 1.0
// viewer had -- at six agents, reading every row IS watching the island. Beyond
// that it becomes a population view: aggregates, an action histogram, and a grid
// of clickable swatches, with a detail card for whoever is followed or hovered.
//
// The old list also re-ran four querySelector calls per row per frame. At 100
// agents that is ~400 DOM queries and ~700 style writes every frame, for rows
// nobody can read. Element references are cached at build time now.
function buildAgentPanel() {
  const host = $('agents');
  host.innerHTML = '';
  state.rows = [];
  state.rosterMode = state.replay.agents.length <= ROSTER_LIMIT;

  if (state.rosterMode) {
    state.replay.agents.forEach((a, i) => {
      const row = document.createElement('div');
      row.className = 'agent' + (state.learn?.[i] ? ' learned' : '');
      row.dataset.i = String(i);
      row.innerHTML =
        `<div class="swatch" style="background:${a.color}"></div>` +
        `<div><div class="name">agent ${a.id}</div>` +
        `<div class="meta"></div><div class="bar"><i></i></div></div>` +
        `<div class="act"></div>`;
      row.addEventListener('click', () => setFollow(state.follow === i ? -1 : i));
      host.appendChild(row);
      state.rows.push({
        row,
        meta: row.querySelector('.meta'),
        fill: row.querySelector('.bar > i'),
        act: row.querySelector('.act'),
      });
    });
    return;
  }

  const stats = document.createElement('div');
  stats.className = 'popstats';
  stats.innerHTML = '<div class="popline"></div><div class="acts"></div>';
  host.appendChild(stats);
  const detail = document.createElement('div');
  detail.className = 'popdetail';
  detail.textContent = 'click or hover an agent';
  host.appendChild(detail);

  const grid = document.createElement('div');
  grid.className = 'swatchgrid';
  state.replay.agents.forEach((a, i) => {
    const cell = document.createElement('button');
    cell.className = 'cell' + (state.learn?.[i] ? ' learned' : '');
    cell.style.background = a.color;
    cell.title = `agent ${a.id}${state.learn?.[i] ? ' (learned)' : ''}`;
    cell.addEventListener('click', () => setFollow(state.follow === i ? -1 : i));
    grid.appendChild(cell);
    state.rows.push({ cell });
  });
  host.appendChild(grid);
  state.popLine = stats.querySelector('.popline');
  state.popActs = stats.querySelector('.acts');
  state.popDetail = detail;
}

/** The arbiter goal an agent held at this tick, or '' when the replay has none. */
function goalName(i, tick) {
  if (!state.goalNames || !Array.isArray(tick.o)) return '';
  return state.goalNames[tick.o[i]] ?? '';
}

/** One-line description of an agent at the current tick, for hover and follow. */
function describeAgent(i) {
  const tick = state.ticks[Math.min(Math.floor(state.t), state.lastTick)];
  const a = tick.a[i];
  const world = state.replay.world;
  if (a[A_ALIVE] !== 1) return `agent ${i} · dead`;
  const act = state.actionNames[a[A_ACTION]] ?? '?';
  // Goal first, action second: the goal is what the agent is TRYING to do and
  // the action is one tick of it. "NE" says nothing; "deliver · NE" says the
  // agent is carrying material to a site, which is the whole question a mixed
  // run asks.
  const goal = goalName(i, tick);
  let s = `agent ${i}${state.learn?.[i] ? ' (learned)' : ''} · `
        + (goal ? `${goal} · ` : '') + `${act} · hunger ${a[A_HUNGER].toFixed(0)}`
        + ` · food ${a[A_FOOD]}/${world.food_capacity}`;
  if (state.construction) s += ` · w${a[A_WOOD]} s${a[A_STONE]}`;
  return s;
}

function updateAgentPanel(tick) {
  const world = state.replay.world;
  if (state.rosterMode) {
    for (let i = 0; i < state.rows.length; i++) {
      const r = state.rows[i], a = tick.a[i];
      const alive = a[A_ALIVE] === 1;
      const hunger = a[A_HUNGER] / world.max_hunger;
      r.row.classList.toggle('dead', !alive);
      r.row.classList.toggle('followed', state.follow === i);
      r.meta.textContent = alive
        ? `hunger ${a[A_HUNGER].toFixed(0)} · food ${a[A_FOOD]}/${world.food_capacity}`
          + (state.construction ? ` · w${a[A_WOOD]} s${a[A_STONE]}` : '')
        : 'dead';
      r.fill.style.width = `${Math.max(alive ? hunger : 0, 0) * 100}%`;
      r.fill.style.background = hunger > 0.55 ? 'var(--good)'
        : hunger > 0.28 ? 'var(--warn)' : 'var(--bad)';
      const g = alive ? goalName(i, tick) : '';
      r.act.textContent = alive
        ? (g ? `${g} · ${state.actionNames[a[A_ACTION]] ?? '?'}`
             : (state.actionNames[a[A_ACTION]] ?? '?'))
        : '—';
    }
    return;
  }

  // Population mode: one pass over the tick for aggregates, then style-only
  // writes on the swatch grid.
  let alive = 0, hungerSum = 0, food = 0, aliveLearned = 0;
  const counts = new Map();
  const hasGoals = !!state.goalNames && Array.isArray(tick.o);
  for (let i = 0; i < state.agentCount; i++) {
    const a = tick.a[i];
    const isAlive = a[A_ALIVE] === 1;
    const cell = state.rows[i].cell;
    cell.classList.toggle('dead', !isAlive);
    cell.classList.toggle('followed', state.follow === i);
    if (!isAlive) continue;
    alive++;
    if (state.learn?.[i]) aliveLearned++;
    hungerSum += a[A_HUNGER];
    food += a[A_FOOD];
    // The histogram counts GOALS when the replay has them (schema v5). A goal
    // histogram answers "how many of them are building?"; an action histogram
    // answers "how many of them are facing north-east", which nobody asked.
    const name = hasGoals ? (goalName(i, tick) || '?')
                          : (state.actionNames[a[A_ACTION]] ?? '?');
    counts.set(name, (counts.get(name) || 0) + 1);
  }
  const meanHunger = alive ? hungerSum / alive : 0;
  state.popLine.innerHTML =
    `<b>${alive}</b>/${state.agentCount} alive · mean hunger <b>${meanHunger.toFixed(0)}</b>`
    + ` · carrying <b>${food}</b> berries`
    + (state.learnCount
        ? ` · <span class="learnnote"><b>${aliveLearned}</b>/${state.learnCount} learned alive</span>`
        : '');

  const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8);
  state.popActs.innerHTML =
    `<div class="actrow" style="opacity:.55"><span>${hasGoals ? 'goal' : 'action'}</span>`
    + `<div class="actbar"></div><b></b></div>`
    + top.map(([name, n]) => {
      const pct = Math.round(100 * n / Math.max(alive, 1));
      return `<div class="actrow"><span>${name}</span>`
           + `<div class="actbar"><i style="width:${pct}%"></i></div><b>${n}</b></div>`;
    }).join('');

  const focus = state.follow >= 0 ? state.follow : state.hover;
  state.popDetail.textContent = focus >= 0 ? describeAgent(focus)
    : 'click or hover an agent';
}

// Name every marker that is actually on screen for THIS replay, from the same
// constants that draw it. Nothing here was discoverable before: a red ball at
// the hip (carried berries) and a red diamond over the head (stealing) look
// like the same "red dot" until someone tells you they are not.
function renderLegend() {
  const host = $('legend');
  if (!host) return;
  const hex = (n) => `#${n.toString(16).padStart(6, '0')}`;
  const names = new Set(state.actionNames);
  const rows = [];

  const item = (cls, color, label, note) =>
    `<div class="item"><span class="key ${cls}" style="background:${color}"></span>` +
    `<span>${label}${note ? ` <span style="opacity:.65">— ${note}</span>` : ''}</span></div>`;

  rows.push('<div class="grp">the agents</div>');
  rows.push(item('', 'linear-gradient(90deg,#e05a7a,#5ac8e0)', 'body colour',
                 'who they are, one per agent'));
  rows.push(item('round', hex(0xd0466a), 'ball at the hip',
                 'berries being carried — bigger means more'));

  rows.push('<div class="grp">diamond over the head — what they are doing</div>');
  // Only list actions this replay actually contains, so a foraging-only world
  // does not advertise stealing and building it never had.
  const labels = {
    gather: 'picking berries', steal: 'stealing from a neighbour',
    chop: 'chopping wood', mine: 'mining stone', build: 'building a shelter',
    give_food: 'giving food away', give_material: 'handing over material',
    deposit_food: 'putting food in the household store',
    deposit_material: 'putting material in the household store',
    withdraw_food: 'taking food from their own store',
    withdraw_material: 'taking material from their own store',
    raid: "RAIDING another household's store",
  };
  for (const [action, color] of Object.entries(ACTION_PIP)) {
    if (names.has(action)) rows.push(item('diamond', hex(color), labels[action] ?? action));
  }
  rows.push('<div class="item" style="opacity:.7">no diamond — walking, or doing nothing</div>');

  if (state.construction) {
    rows.push('<div class="grp">the island</div>');
    rows.push(item('ring', 'none', 'ring on the ground',
                   'how far a finished shelter protects'));
    rows.push(item('', '#7a5c3a', 'brown dome', 'a shelter (part-built ones are smaller)'));
    rows.push(item('', '#4f7a3a', 'green cone', 'a tree — wood'));
    rows.push(item('', '#8b93a0', 'grey lump', 'a rock — stone'));
  }
  rows.push(item('', '#3f6b32', 'leafy bush', 'food; it goes bare brown when picked clean'));

  if (state.society) {
    rows.push('<div class="grp">households (stage 4)</div>');
    rows.push(item('ring', 'none', 'coloured ring on the ground',
                   "a household's home and its stockpile"));
    rows.push(item('', hex(GIFT_COLORS[0]), 'green crate', 'food in the store'));
    rows.push(item('', hex(GIFT_COLORS[1]), 'yellow crate', 'material in the store'));
  }
  if (state.exchange || state.society) {
    rows.push('<div class="grp">arcs through the air</div>');
    if (state.exchange) {
      rows.push(item('', hex(GIFT_COLORS[0]), 'green or yellow arc',
                     'something was just handed over, from giver to receiver'));
    }
    if (state.society) {
      // The two arc colours mean opposite things about consent, and telling them
      // apart is the whole reason the legend exists (the two-red-dots lesson).
      rows.push(item('', hex(RAID_COLOR), 'red arc',
                     'a raid — it runs from the robbed store to the raider'));
    }
  }
  if (state.learnCount) {
    rows.push('<div class="grp">the mixed population (schema v5)</div>');
    rows.push(item('round', '#6fd3ff', 'cyan ring on the ground',
                   `one of the ${state.learnCount} agents whose goals come from a `
                   + 'LEARNED arbiter; everyone else is scripted'));
  }
  if (state.goalNames) {
    rows.push('<div class="grp">the panel (schema v5)</div>');
    rows.push(item('', 'transparent', 'the histogram counts GOALS, not actions',
                   'an intention like "deliver" rather than one tick of walking'));
  }
  if (state.predators.length) {
    rows.push('<div class="grp">the night predator (schema v6)</div>');
    rows.push(item('', '#8a1420', 'dark red spike',
                   'a predator — it hunts agents who are NOT under cover, and '
                   + 'walks home to its den at dawn'));
    rows.push(item('round', '#ff3b30', 'red circle around it',
                   'the reach it actually takes hunger inside; shown only at night'));
  }
  if (state.society) {
    rows.push('<div class="grp">the weather (schema v5)</div>');
    rows.push(item('', '#7fb3ff', 'blue flash and banner',
                   'a storm just knocked finished shelters back down'));
    rows.push(item('', '#e2c14a', 'yellow tint and banner',
                   'a blight: berries have stopped regrowing until it lifts'));
  }
  host.innerHTML = rows.join('');
}

function renderSummary() {
  const s = state.replay.summary || {};
  let html =
    `<b>${s.mean_lifespan ?? '?'}</b> mean lifespan · <b>${s.deaths ?? '?'}</b> deaths · ` +
    `<b>${s.berries_gathered ?? '?'}</b> berries · <b>${s.meals ?? '?'}</b> meals`;
  if (s.shelters_completed !== undefined) {
    const nights = s.night_ticks_sheltered + s.night_ticks_exposed;
    const pct = nights ? Math.round(100 * s.night_ticks_sheltered / nights) : 0;
    html += ` · <b>${s.shelters_completed}</b> shelters · <b>${pct}%</b> of night indoors`;
  }
  if (s.gifts !== undefined) {
    html += ` · <b>${s.gifts}</b> transfers (${s.food_given ?? 0} food / ${s.materials_given ?? 0} material)`;
  }
  if (s.raids !== undefined) {
    html += ` · <b>${s.deposits ?? 0}</b> deposits / <b>${s.withdrawals ?? 0}</b> withdrawals`
          + ` · <b>${s.raids}</b> raids`;
    if (s.storms) {
      html += ` · <b>${s.storms}</b> storms (${s.shelters_damaged ?? 0} shelters hit),`
            + ` <b>${s.blight_ticks ?? 0}</b> blighted ticks`;
    }
  }
  $('summary').innerHTML = html;
}

function setFollow(i) {
  state.follow = i;
  if (i < 0) controls.target.set(0, 0, 0);
}

// ---------------------------------------------------------------------------
// Transport controls
// ---------------------------------------------------------------------------

const speedsHost = $('speeds');
SPEEDS.forEach((s) => {
  const b = document.createElement('button');
  b.textContent = `${s}×`;
  b.dataset.speed = String(s);
  b.addEventListener('click', () => setSpeed(s));
  speedsHost.appendChild(b);
});

function setSpeed(s) {
  state.speed = s;
  [...speedsHost.children].forEach((b) => b.classList.toggle('on', Number(b.dataset.speed) === s));
}
setSpeed(1);

function updatePlayButton() {
  $('play').textContent = state.playing ? '❚❚ Pause' : '▶ Play';
}
$('play').addEventListener('click', () => {
  // Restarting from the end is friendlier than a dead button.
  if (!state.playing && state.t >= state.lastTick) state.t = 0;
  state.playing = !state.playing;
  updatePlayButton();
});

$('scrub').addEventListener('input', (e) => {
  state.t = Number(e.target.value);
  if (state.replay) applyTick(state.t);
});

addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { e.preventDefault(); $('play').click(); }
  if (e.code === 'ArrowRight' && state.replay) { state.t = Math.min(state.t + 1, state.lastTick); applyTick(state.t); }
  if (e.code === 'ArrowLeft' && state.replay) { state.t = Math.max(state.t - 1, 0); applyTick(state.t); }
});

// Click an agent in the scene to follow it, same as clicking its panel row.
const raycaster = new THREE.Raycaster();
renderer.domElement.addEventListener('pointerdown', (e) => {
  if (!state.replay) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((e.clientX - rect.left) / rect.width) * 2 - 1,
    -((e.clientY - rect.top) / rect.height) * 2 + 1,
  );
  raycaster.setFromCamera(ndc, camera);
  if (!state.population) return;
  // One instanced torso mesh for the whole population: a hit reports which
  // instance it was, so picking needs no per-agent object list.
  const hits = raycaster.intersectObject(state.population.pickTarget, false);
  if (hits.length && hits[0].instanceId !== undefined) {
    const idx = hits[0].instanceId;
    setFollow(state.follow === idx ? -1 : idx);
  }
});

// Hovering names an agent without selecting it -- the only way to tell who is
// who once the population outgrows the roster list.
renderer.domElement.addEventListener('pointermove', (e) => {
  if (!state.replay || !state.population) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((e.clientX - rect.left) / rect.width) * 2 - 1,
    -((e.clientY - rect.top) / rect.height) * 2 + 1,
  );
  raycaster.setFromCamera(ndc, camera);
  const hits = raycaster.intersectObject(state.population.pickTarget, false);
  const idx = hits.length && hits[0].instanceId !== undefined ? hits[0].instanceId : -1;
  if (idx === state.hover) return;
  state.hover = idx;
  const label = $('hoverLabel');
  if (!label) return;
  if (idx < 0) {
    label.style.display = 'none';
  } else {
    label.style.display = 'block';
    label.textContent = describeAgent(idx);
    label.style.borderColor = state.replay.agents[idx].color;
  }
  label.style.left = `${e.clientX - rect.left + 14}px`;
  label.style.top = `${e.clientY - rect.top + 12}px`;
});

// ---------------------------------------------------------------------------
// Replay sources: manifest dropdown, file picker, drag and drop
// ---------------------------------------------------------------------------

async function loadFromUrl(url) {
  $('loading').style.display = 'block';
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  loadReplay(await res.json(), url);
}

async function populateManifest() {
  const select = $('replaySelect');
  try {
    const res = await fetch('replays/index.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const manifest = await res.json();
    const entries = manifest.replays || [];
    // GROUPED, because a flat list does not survive its own success: there are
    // 300-odd training snapshots in here and five worlds anybody wants to look
    // at, and a dropdown that buries the second set in the first is a dropdown
    // nobody scrolls. The split is the recorder's own `source` field, so it
    // cannot drift from what actually wrote the file.
    const islands = entries.filter((e) => e.source === 'sim.society');
    const rest = entries.filter((e) => e.source !== 'sim.society');
    const addGroup = (label, list) => {
      if (!list.length) return;
      const g = document.createElement('optgroup');
      g.label = label;
      for (const e of list) {
        const opt = document.createElement('option');
        opt.value = e.file;
        const life = e.summary?.mean_lifespan;
        opt.textContent = `${e.label} — ${e.ticks} ticks`
                        + (life != null ? `, lifespan ${life}` : '');
        g.appendChild(opt);
      }
      select.appendChild(g);
    };
    addGroup(`the island — ${islands.length} worlds worth watching`, islands);
    addGroup(`older milestone runs (${rest.length}) — training snapshots`, rest);
    if (!entries.length) {
      $('pickerHint').textContent = 'No replays yet. Run sim/make_fake_replay.py or train.py.';
      $('loading').style.display = 'none';
      return;
    }
    // The manifest is newest-first, so entry 0 is whatever run finished last --
    // usually a diagnostic probe world with most mechanics switched off. Honour
    // an explicit ?replay= first so a link can point at a specific run.
    const asked = new URLSearchParams(location.search).get('replay');
    // Default to an ISLAND rather than to whatever finished last, which for
    // months has been a diagnostic probe world with most mechanics switched off.
    const fallback = (islands[0] || entries[0]).file;
    const wanted = entries.some((e) => e.file === asked) ? asked : fallback;
    select.value = wanted;
    await loadFromUrl(`replays/${wanted}`);
  } catch (err) {
    // file:// blocks fetch, so this is the normal path when the page is opened
    // by double-clicking it. Say so instead of showing a bare error.
    $('pickerHint').innerHTML =
      `Could not read <code>replays/index.json</code> (${err.message}).<br>` +
      `Serve the folder — <code>python -m http.server</code> — or just drop a replay file here.`;
    $('loading').style.display = 'none';
  }
}

$('replaySelect').addEventListener('change', (e) => {
  if (e.target.value) loadFromUrl(`replays/${e.target.value}`).catch((err) => fatal('Cannot load replay', err.message));
});

$('fileInput').addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (file) readFile(file);
});

function readFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      loadReplay(JSON.parse(reader.result), file.name);
    } catch (err) {
      fatal('Cannot render this replay', `${file.name}\n\n${err.message}`);
    }
  };
  reader.readAsText(file);
}

const dropOverlay = $('drop');
let dragDepth = 0;
addEventListener('dragenter', (e) => { e.preventDefault(); if (++dragDepth === 1) dropOverlay.style.display = 'grid'; });
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; dropOverlay.style.display = 'none'; } });
addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  dropOverlay.style.display = 'none';
  const file = e.dataTransfer?.files?.[0];
  if (file) readFile(file);
});

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);

  if (state.replay) {
    if (state.playing) {
      state.t += dt * BASE_TICKS_PER_SECOND * state.speed;
      if (state.t >= state.lastTick) {
        state.t = state.lastTick;
        state.playing = false;
        updatePlayButton();
      }
      applyTick(state.t);
    }

    // No per-agent billboarding any more: the always-on hunger bar is gone, so
    // there is nothing to keep facing the camera. It cost a world-quaternion
    // decomposition and an inversion per agent per frame, and at 100 agents a
    // hundred floating bars were unreadable anyway -- hunger now lives in the
    // panel, and on the followed agent's card.
    //
    // If following, glide the orbit pivot along with the agent while preserving
    // whatever orbit offset the user has dialled in.
    if (state.follow >= 0 && state.agentPos) {
      _tmpVec.set(state.agentPos[state.follow * 2], ISLAND_TOP,
                  state.agentPos[state.follow * 2 + 1])
        .sub(controls.target).multiplyScalar(0.12);
      controls.target.add(_tmpVec);
      camera.position.add(_tmpVec);
    }
  }

  // Cheap water ripple.
  const time = clock.elapsedTime;
  const pos = waterGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = waterBase[i * 3], z = waterBase[i * 3 + 2];
    pos.array[i * 3 + 1] = Math.sin(x * 0.03 + time * 0.7) * 0.7 + Math.cos(z * 0.045 + time * 0.5) * 0.5;
  }
  pos.needsUpdate = true;
  waterGeo.computeVertexNormals();

  controls.update();
  renderer.render(scene, camera);
}

resize();
animate();
populateManifest();
