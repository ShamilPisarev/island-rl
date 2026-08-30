// Procedural low-poly bipeds for the replay viewer, instanced for 100 agents.
//
// Design doc route 1: build a humanoid in code from boxes, animate it
// procedurally from replay data. No asset files, no rigging, no extra CDN
// dependency, so the viewer keeps working offline exactly as the rest of it does.
//
// WHY INSTANCED, AND WHAT IT REPLACES. The 6-agent viewer gave every agent its
// own THREE.Group of ~13 meshes -- capsule, nose, two bar planes, a ground ring
// and eight berry pips -- with its own geometries and material. That is about
// 1300 meshes and 1300 draw calls at 100 agents, plus 100 shadow casters, and it
// is the hard ceiling the design doc's graphics section was worried about. Here
// each BODY PART is one InstancedMesh across the whole population, so a 100-agent
// island is 8 draw calls. Limbs still move independently because each part's
// per-instance matrix is composed separately; instancing costs the hierarchy, not
// the animation.
//
// Triangle budget: 7 boxes of 12 triangles = 84 per agent, inside the doc's ~150.
//
// EVERYTHING IS A FUNCTION OF THE FRACTIONAL TICK, never of accumulated frame
// time. Scrubbing the timeline backwards has to give the same pose as arriving
// there forwards, which is the same rule the existing corpse fade follows.

import * as THREE from 'three';

// Limb geometry, in world units. An agent is ~4 units tall, matching the capsule
// it replaces, so camera framing and the island scale are unchanged.
const TORSO = { w: 1.05, h: 1.45, d: 0.62 };
const HEAD = { w: 0.78, h: 0.72, d: 0.74 };
const ARM = { w: 0.28, h: 1.05, d: 0.28 };
const LEG = { w: 0.34, h: 1.15, d: 0.34 };

const HIP_Y = LEG.h + 0.15;                 // where the legs meet the torso
const TORSO_Y = HIP_Y + TORSO.h / 2;
const SHOULDER_Y = HIP_Y + TORSO.h * 0.86;
const HEAD_Y = HIP_Y + TORSO.h + HEAD.h / 2 - 0.08;

// Which actions get which animation. Names come from the replay's own
// action_names, so this never has to know action indices.
const ANIM = {
  gather: 'reach', steal: 'reach',
  chop: 'swing', mine: 'swing', build: 'swing',
  give_food: 'offer', give_material: 'offer',
};

const PART_NAMES = ['torso', 'head', 'armL', 'armR', 'legL', 'legR'];

function boxes() {
  return {
    torso: new THREE.BoxGeometry(TORSO.w, TORSO.h, TORSO.d),
    head: new THREE.BoxGeometry(HEAD.w, HEAD.h, HEAD.d),
    armL: new THREE.BoxGeometry(ARM.w, ARM.h, ARM.d),
    armR: new THREE.BoxGeometry(ARM.w, ARM.h, ARM.d),
    legL: new THREE.BoxGeometry(LEG.w, LEG.h, LEG.d),
    legR: new THREE.BoxGeometry(LEG.w, LEG.h, LEG.d),
  };
}

// Pivot the limb geometries at the TOP face rather than the centre, so rotating
// one swings it from the shoulder or hip instead of spinning about its middle.
function pivotAtTop(geo, height) {
  geo.translate(0, -height / 2, 0);
  return geo;
}

/**
 * Build an instanced population.
 *
 * @param {number} count  number of agents
 * @param {string[]} colors  per-agent hex colour from the replay
 * @param {object} opts  { headColor } tint for heads/limbs, kept darker than the
 *   body so a 100-agent crowd still reads as figures rather than as confetti.
 */
export function createPopulation(count, colors, opts = {}) {
  const group = new THREE.Group();
  const geo = boxes();
  pivotAtTop(geo.armL, ARM.h);
  pivotAtTop(geo.armR, ARM.h);
  pivotAtTop(geo.legL, LEG.h);
  pivotAtTop(geo.legR, LEG.h);

  // One material for every part: instance colour carries identity, so a single
  // material is all the population needs and the parts can share it.
  const material = new THREE.MeshStandardMaterial({
    roughness: 0.62, metalness: 0.04, flatShading: true,
  });

  const parts = {};
  for (const name of PART_NAMES) {
    const mesh = new THREE.InstancedMesh(geo[name], material, count);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    // Shadows from the torso alone. 100 instanced shadow casters is affordable;
    // six per agent is not, and the limbs' own shadows are invisible at any
    // camera distance you would actually watch a 100-agent island from.
    mesh.castShadow = name === 'torso';
    mesh.frustumCulled = false;   // instances move; one shared box would mis-cull
    parts[name] = mesh;
    group.add(mesh);
  }

  // A floating pip above the head, coloured by what the agent is doing. This
  // replaces the ground ring: 100 overlapping rings of radius ~1.5 read as noise
  // on the island floor, while a head-height pip stays legible in a crowd.
  const pipGeo = new THREE.OctahedronGeometry(0.32, 0);
  const pipMat = new THREE.MeshBasicMaterial({ transparent: true });
  const actionPip = new THREE.InstancedMesh(pipGeo, pipMat, count);
  actionPip.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  actionPip.frustumCulled = false;
  group.add(actionPip);

  // Carried food, as one pip whose size grows with the load. The old viewer used
  // eight sphere meshes per agent purely for this readout -- 800 meshes at 100
  // agents to show a number between 0 and 3.
  const carryGeo = new THREE.SphereGeometry(0.2, 6, 5);
  const carryMat = new THREE.MeshBasicMaterial({ color: 0xd0466a });
  const carryPip = new THREE.InstancedMesh(carryGeo, carryMat, count);
  carryPip.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  carryPip.frustumCulled = false;
  group.add(carryPip);

  // Per-instance colours. Bodies take the replay's agent colour; heads and limbs
  // take a darkened version of it, which is what makes a figure read as a figure.
  const tint = new THREE.Color(opts.headColor ?? 0x1a2230);
  const base = new THREE.Color();
  for (let i = 0; i < count; i++) {
    base.set(colors[i] ?? '#8899aa');
    parts.torso.setColorAt(i, base);
    const limb = base.clone().lerp(tint, 0.45);
    parts.head.setColorAt(i, limb);
    parts.armL.setColorAt(i, limb);
    parts.armR.setColorAt(i, limb);
    parts.legL.setColorAt(i, limb);
    parts.legR.setColorAt(i, limb);
    actionPip.setColorAt(i, base);
    carryPip.setColorAt(i, new THREE.Color(0xd0466a));
  }
  for (const name of PART_NAMES) parts[name].instanceColor.needsUpdate = true;
  actionPip.instanceColor.needsUpdate = true;

  // Reusable temporaries: this runs count x parts times per frame, and allocating
  // a Matrix4 in that loop is exactly how a viewer starts stuttering at 100.
  const M = new THREE.Matrix4();
  const root = new THREE.Matrix4();
  const L = new THREE.Matrix4();
  const R = new THREE.Matrix4();
  const T = new THREE.Matrix4();
  const S = new THREE.Matrix4();
  const HIDDEN = new THREE.Matrix4().makeScale(0, 0, 0);
  const pipColor = new THREE.Color();

  /** Compose base-space translate + rotate for one part and write it out. */
  function place(mesh, i, root, x, y, z, rx, rz, scale) {
    T.makeTranslation(x, y, z);
    R.makeRotationX(rx || 0);
    L.copy(T).multiply(R);
    if (rz) {
      R.makeRotationZ(rz);
      L.multiply(R);
    }
    if (scale !== undefined && scale !== 1) {
      S.makeScale(scale, scale, scale);
      L.multiply(S);
    }
    M.copy(root).multiply(L);
    mesh.setMatrixAt(i, M);
  }

  /**
   * Pose one agent.
   *
   * @param {number} i agent index
   * @param {object} p pose:
   *   x, z        world position
   *   heading     facing, radians (atan2(dx, dz))
   *   speed       world units per tick, drives the stride
   *   phase       fractional tick; the ONLY clock, so scrubbing is stable
   *   action      action name from the replay, or undefined
   *   food        carried berries
   *   dead        0..1 collapse fraction
   *   resting     true to sit down (night, indoors, doing nothing)
   *   hidden      true to skip drawing entirely
   */
  function setPose(i, p) {
    if (p.hidden) {
      for (const name of PART_NAMES) parts[name].setMatrixAt(i, HIDDEN);
      actionPip.setMatrixAt(i, HIDDEN);
      carryPip.setMatrixAt(i, HIDDEN);
      return;
    }

    const dead = p.dead || 0;
    // A corpse folds forward and sinks. Composed into the root so every part
    // follows, rather than tipping the torso and leaving the legs standing.
    T.makeTranslation(p.x, 0, p.z);
    R.makeRotationY(p.heading || 0);
    root.copy(T).multiply(R);
    if (dead > 0) {
      T.makeTranslation(0, -dead * (HIP_Y * 0.72), 0);
      R.makeRotationX(dead * Math.PI * 0.46);
      root.multiply(T).multiply(R);
    }
    // A child is a smaller adult (schema v7). Composed into the root, so every
    // part, pip and carried berry scales with it and no per-part branch is
    // needed. A village whose children are drawn the same size as its parents
    // is a village where the whole reproduction result is invisible.
    if (p.scale && p.scale !== 1) {
      S.makeScale(p.scale, p.scale, p.scale);
      root.multiply(S);
    }

    const anim = ANIM[p.action];
    const resting = p.resting && !anim;
    // Stride: sinusoidal swing keyed to actual speed, so a stationary agent
    // stands still instead of marching on the spot.
    const gait = Math.min((p.speed || 0) / 0.8, 1.4);
    const swing = Math.sin((p.phase || 0) * 1.35) * gait * 0.85;
    const bob = Math.abs(Math.sin((p.phase || 0) * 1.35)) * gait * 0.09;
    // A fast local cycle for work animations -- a hammer swing is not a stride.
    const work = Math.sin((p.phase || 0) * 3.1);

    let lean = gait * 0.12;
    let armL = swing, armR = -swing;
    let legL = -swing * 0.8, legR = swing * 0.8;
    let hipDrop = 0;

    if (anim === 'reach') {
      // Both arms forward and down, torso tipped: picking something up.
      armL = armR = -1.15 - work * 0.12;
      lean = 0.34;
    } else if (anim === 'swing') {
      // Overhead hammer loop: arms together, cycling through the vertical.
      armL = armR = -2.5 + work * 1.5;
      lean = 0.2 + work * 0.08;
    } else if (anim === 'offer') {
      // Arms straight out front, holding something toward someone.
      armL = armR = -Math.PI / 2;
      lean = 0.1;
    } else if (resting) {
      // Sitting: hips drop, thighs forward, arms relaxed. This is what a night
      // spent inside a finished shelter looks like from the camera.
      hipDrop = -LEG.h * 0.52;
      legL = legR = -1.45;
      armL = armR = -0.25;
      lean = 0.16;
    }

    // Torso and head hang off the (possibly dropped) hip height.
    place(parts.torso, i, root, 0, TORSO_Y + hipDrop + bob, 0, lean, 0, 1);
    place(parts.head, i, root, 0, HEAD_Y + hipDrop + bob, lean * 0.5, lean * 0.6, 0, 1);
    const shoulder = SHOULDER_Y + hipDrop + bob;
    place(parts.armL, i, root, -(TORSO.w / 2 + ARM.w / 2 - 0.03), shoulder, 0, armL, 0, 1);
    place(parts.armR, i, root, TORSO.w / 2 + ARM.w / 2 - 0.03, shoulder, 0, armR, 0, 1);
    const hip = HIP_Y + hipDrop + bob;
    place(parts.legL, i, root, -LEG.w * 0.62, hip, 0, legL, 0, 1);
    place(parts.legR, i, root, LEG.w * 0.62, hip, 0, legR, 0, 1);

    // Action pip: shown only while the agent is doing something other than
    // walking, and hidden outright for a corpse.
    if (p.action && p.pipColor !== undefined && dead < 1) {
      place(actionPip, i, root, 0, HEAD_Y + hipDrop + 1.15 + bob, 0, 0, 0, 1);
      pipColor.setHex(p.pipColor);
      actionPip.setColorAt(i, pipColor);
      actionPip.instanceColor.needsUpdate = true;
    } else {
      actionPip.setMatrixAt(i, HIDDEN);
    }

    // Carried food: one pip that grows with the load, tucked at the near hip.
    if (p.food > 0 && dead < 1) {
      const scale = 0.7 + 0.42 * Math.min(p.food, 6);
      place(carryPip, i, root, TORSO.w * 0.52, hip + 0.35, 0.34, 0, 0, scale);
    } else {
      carryPip.setMatrixAt(i, HIDDEN);
    }
  }

  function commit() {
    for (const name of PART_NAMES) parts[name].instanceMatrix.needsUpdate = true;
    actionPip.instanceMatrix.needsUpdate = true;
    carryPip.instanceMatrix.needsUpdate = true;
  }

  function dispose() {
    for (const name of PART_NAMES) parts[name].geometry.dispose();
    pipGeo.dispose();
    carryGeo.dispose();
    material.dispose();
    pipMat.dispose();
    carryMat.dispose();
  }

  return {
    group,
    setPose,
    commit,
    dispose,
    // Raycast against the torsos: an InstancedMesh hit reports `instanceId`,
    // which is the agent index, so picking survives the move to instancing.
    pickTarget: parts.torso,
    parts,
  };
}
