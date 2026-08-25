// Island replay viewer.
//
// Loads a replay written by sim/replay.py and plays it back in 3D. It renders
// only what the replay says; it never simulates anything itself. If the file's
// schema_version is not one this build understands, it refuses to render rather
// than quietly drawing something plausible and wrong.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- replay schema this build can read (see sim/replay.py) ------------------
const SUPPORTED_SCHEMA = [1, 2, 3];   // v2 = M4 materials/shelters/night, v3 = M5 transfers
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
const MAX_GIFT_LINES = 32;
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
  agents: [],         // { group, body, material, barFill, berries[], deathTick }
  bushes: [],         // { mesh, base }
  trees: [],
  rocks: [],
  sites: [],
  actionNames: [],
  construction: false,
  exchange: false,
  giftLines: [],      // pooled line segments, reused every frame
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

function buildAgent(color) {
  const group = new THREE.Group();

  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(color), roughness: 0.55, metalness: 0.05,
    flatShading: true, transparent: true,
  });
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(1.0, 2.0, 3, 10), material);
  body.position.y = 2.0;
  body.castShadow = true;
  group.add(body);

  // A blunt nose so the direction of travel is readable from a high camera.
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.5, 1.3, 5), material);
  nose.position.set(0, 2.2, 1.15);
  nose.rotation.x = Math.PI / 2;
  nose.castShadow = true;
  group.add(nose);

  // Hunger bar: two unlit planes, billboarded at the head each frame.
  const bar = new THREE.Group();
  bar.position.y = 5.0;
  const track = new THREE.Mesh(
    new THREE.PlaneGeometry(2.8, 0.5),
    new THREE.MeshBasicMaterial({ color: 0x10161d, transparent: true, opacity: 0.85 }),
  );
  bar.add(track);
  const fillMat = new THREE.MeshBasicMaterial({ color: 0x6ec46e });
  const barFill = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 0.3), fillMat);
  barFill.position.z = 0.01;
  bar.add(barFill);
  group.add(bar);

  // Action marker: a disc at the agent's feet, lit while it is gathering or
  // stealing. Reading the side panel tells you what one agent is doing; this
  // tells you what all six are doing at once, which is the thing you actually
  // want when watching for competition.
  const marker = new THREE.Mesh(
    new THREE.RingGeometry(1.15, 1.65, 20),
    new THREE.MeshBasicMaterial({ color: 0x6ec46e, transparent: true, opacity: 0.85,
                                  side: THREE.DoubleSide }),
  );
  marker.rotation.x = -Math.PI / 2;
  marker.position.y = 0.06;
  marker.visible = false;
  group.add(marker);

  // Carried berries, shown as little pips under the bar.
  const berries = [];
  const berryGeo = new THREE.SphereGeometry(0.16, 6, 5);
  const berryMat = new THREE.MeshBasicMaterial({ color: 0xd0466a });
  for (let i = 0; i < 8; i++) {
    const pip = new THREE.Mesh(berryGeo, berryMat);
    pip.visible = false;
    bar.add(pip);
    berries.push(pip);
  }

  return { group, body, nose, material, bar, barFill, fillMat, berries, marker };
}

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

  const capacity = replay.world.food_capacity;
  state.agents = replay.agents.map((a, i) => {
    const built = buildAgent(a.color);
    // Lay out the berry pips for this replay's carrying capacity.
    built.berries.forEach((pip, k) => {
      const shown = Math.min(capacity, built.berries.length);
      pip.position.set((k - (shown - 1) / 2) * 0.44, -0.42, 0);
      pip.userData.slot = k;
    });
    worldGroup.add(built.group);
    built.deathTick = deathTickOf(replay, i);
    built.index = i;
    return built;
  });

  camera.position.set(0, radius * 1.55, radius * 2.3);
  controls.target.set(0, 0, 0);
  controls.update();

  buildAgentPanel();
  renderSummary();
  $('replayLabel').textContent = `${replay.label ?? 'replay'} · ${replay.source ?? '?'} · seed ${replay.seed}`;
  $('scrub').max = String(state.lastTick);
  $('scrub').value = '0';
  clearFatal();
  $('loading').style.display = 'none';
  updatePlayButton();
  applyTick(0);
}

function deathTickOf(replay, agentIndex) {
  for (let t = 0; t < replay.ticks.length; t++) {
    if (replay.ticks[t].a[agentIndex][A_ALIVE] === 0) return t;
  }
  return Infinity;
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

  for (let i = 0; i < state.agents.length; i++) {
    const A = state.agents[i];
    const a0 = cur.a[i], a1 = nxt.a[i];
    const alive = a0[A_ALIVE] === 1;

    const x = a0[A_X] + (a1[A_X] - a0[A_X]) * f;
    const z = a0[A_Z] + (a1[A_Z] - a0[A_Z]) * f;
    const hunger = (a0[A_HUNGER] + (a1[A_HUNGER] - a0[A_HUNGER]) * f) / world.max_hunger;

    // Corpses settle over a few ticks and then stay put, computed from the tick
    // index rather than accumulated over frames, so scrubbing stays consistent.
    const dead = Math.min(Math.max((t - A.deathTick) / DEATH_FADE_TICKS, 0), 1);

    A.group.position.set(x, ISLAND_TOP, z);
    A.group.visible = true;
    A.material.opacity = 1 - dead * 0.62;
    A.body.rotation.z = dead * Math.PI * 0.5;
    A.body.position.y = 2.0 - dead * 1.0;
    A.nose.visible = dead < 1;
    A.nose.rotation.z = dead * Math.PI * 0.5;
    A.bar.visible = dead === 0;

    if (alive) {
      const dx = a1[A_X] - a0[A_X], dz = a1[A_Z] - a0[A_Z];
      if (dx * dx + dz * dz > 1e-6) {
        A.group.rotation.y = Math.atan2(dx, dz);
      }
      A.fillMat.color.setHex(hunger > 0.55 ? 0x6ec46e : hunger > 0.28 ? 0xe0b83c : 0xe0563c);
      // Anchor the fill to the left edge rather than the centre.
      A.barFill.scale.x = Math.max(hunger, 0.001);
      A.barFill.position.x = -(1 - Math.max(hunger, 0)) * 1.3;

      const food = a0[A_FOOD];
      A.berries.forEach((pip, k) => { pip.visible = k < food; });

      const action = state.actionNames[a0[A_ACTION]];
      const MARKERS = { gather: GATHER_COLOR, steal: STEAL_COLOR,
                        chop: 0x8a5a2b, mine: 0x9aa7b8, build: 0xe0b83c,
                        give_food: 0x8ce08c, give_material: 0xe0b83c };
      const color = MARKERS[action];
      A.marker.visible = color !== undefined;
      if (color !== undefined) A.marker.material.color.setHex(color);
    } else {
      A.marker.visible = false;
    }
  }

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

  if (state.exchange) applyTransfers(t);

  updateAgentPanel(cur);
  const night = state.construction && nightFactor(t) > 0.5;
  $('tickCount').textContent = `${night ? '\u263e ' : ''}${i0} / ${state.lastTick}`;
  $('scrub').value = String(t);
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

function applyTransfers(t) {
  ensureGiftLines();
  const i0 = Math.min(Math.floor(t), state.lastTick);
  let used = 0;
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
      const color = GIFT_COLORS[item] ?? 0xffffff;

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

      // The item flies over the first half of the window, then the arc fades.
      const [x, y, z] = giftArcPoint(ax, az, bx, bz, Math.min(age * 2, 1));
      slot.item.position.set(x, y, z);
      slot.item.material.color.setHex(color);
      slot.item.material.opacity = 1 - age;
      slot.item.visible = true;
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

function buildAgentPanel() {
  const host = $('agents');
  host.innerHTML = '';
  state.replay.agents.forEach((a, i) => {
    const row = document.createElement('div');
    row.className = 'agent';
    row.dataset.i = String(i);
    row.innerHTML =
      `<div class="swatch" style="background:${a.color}"></div>` +
      `<div><div class="name">agent ${a.id}</div>` +
      `<div class="meta"></div><div class="bar"><i></i></div></div>` +
      `<div class="act"></div>`;
    row.addEventListener('click', () => setFollow(state.follow === i ? -1 : i));
    host.appendChild(row);
  });
}

function updateAgentPanel(tick) {
  const rows = $('agents').children;
  const world = state.replay.world;
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i], a = tick.a[i];
    const alive = a[A_ALIVE] === 1;
    const hunger = a[A_HUNGER] / world.max_hunger;
    row.classList.toggle('dead', !alive);
    row.classList.toggle('followed', state.follow === i);
    row.querySelector('.meta').textContent = alive
      ? `hunger ${a[A_HUNGER].toFixed(0)} · food ${a[A_FOOD]}/${world.food_capacity}`
        + (state.construction ? ` · w${a[A_WOOD]} s${a[A_STONE]}` : '')
      : 'dead';
    const fill = row.querySelector('.bar > i');
    fill.style.width = `${Math.max(alive ? hunger : 0, 0) * 100}%`;
    fill.style.background = hunger > 0.55 ? 'var(--good)' : hunger > 0.28 ? 'var(--warn)' : 'var(--bad)';
    row.querySelector('.act').textContent = alive ? (state.actionNames[a[A_ACTION]] ?? '?') : '—';
  }
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
  const hits = raycaster.intersectObjects(state.agents.map((a) => a.body), false);
  if (hits.length) {
    const idx = state.agents.findIndex((a) => a.body === hits[0].object);
    setFollow(state.follow === idx ? -1 : idx);
  }
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
    for (const e of entries) {
      const opt = document.createElement('option');
      opt.value = e.file;
      const life = e.summary?.mean_lifespan;
      opt.textContent = `${e.label} — ${e.ticks} ticks` + (life != null ? `, lifespan ${life}` : '');
      select.appendChild(opt);
    }
    if (!entries.length) {
      $('pickerHint').textContent = 'No replays yet. Run sim/make_fake_replay.py or train.py.';
      $('loading').style.display = 'none';
      return;
    }
    // The manifest is newest-first, so entry 0 is whatever run finished last --
    // usually a diagnostic probe world with most mechanics switched off. Honour
    // an explicit ?replay= first so a link can point at a specific run.
    const asked = new URLSearchParams(location.search).get('replay');
    const wanted = entries.some((e) => e.file === asked) ? asked : entries[0].file;
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

    // Billboard the hunger bars. The bar hangs off the agent group, which is
    // rotated to the direction of travel, so copying the camera quaternion
    // straight onto it would leave the parent's heading baked in and the bar
    // would swing edge-on. Cancel the parent first: local = parentWorld^-1 * camera.
    for (const A of state.agents) {
      A.group.getWorldQuaternion(_tmpQuat).invert();
      A.bar.quaternion.copy(_tmpQuat).multiply(camera.quaternion);
    }
    // If following, glide the orbit pivot along with the agent while preserving
    // whatever orbit offset the user has dialled in.
    if (state.follow >= 0) {
      const A = state.agents[state.follow];
      _tmpVec.copy(A.group.position).sub(controls.target).multiplyScalar(0.12);
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
