// Behavioural divergence view (Milestone 2).
//
// Renders the JSON report written by sim/divergence.py. No three.js here: the
// interesting questions are all two-dimensional (who goes where, who does what),
// and a top-down heatmap answers them better than a 3D overlay would.

const SUPPORTED_SCHEMA = [1];
const $ = (id) => document.getElementById(id);

const STAT_COLUMNS = [
  { key: 'lifespan', label: 'lifespan', digits: 1 },
  { key: 'gather_share', label: 'gather %', digits: 1, pct: true },
  { key: 'travel_share', label: 'travel %', digits: 1, pct: true },
  { key: 'idle_share', label: 'idle %', digits: 1, pct: true },
  { key: 'gather_success_rate', label: 'hit rate %', digits: 1, pct: true },
  { key: 'mean_dist_to_nearest_bush', label: 'bush dist', digits: 2 },
  { key: 'time_in_gather_range', label: 'in range %', digits: 1, pct: true },
  { key: 'mean_radius', label: 'radius', digits: 1 },
];

function fail(message) {
  $('empty').hidden = false;
  $('content').hidden = true;
  $('empty').innerHTML = `<div style="border-left:3px solid var(--bad);padding:10px 14px">
    <b style="color:var(--bad)">Cannot render this report</b><br>
    <pre style="white-space:pre-wrap;color:var(--dim);margin:8px 0 0">${message}</pre></div>`;
}

function validate(report, origin) {
  if (!SUPPORTED_SCHEMA.includes(report?.schema_version)) {
    throw new Error(`${origin}\n\nReport schema version: ${JSON.stringify(report?.schema_version)}` +
      `\nThis view understands: ${SUPPORTED_SCHEMA.join(', ')}\n\nRefusing to render.`);
  }
  if (!Array.isArray(report.agents) || !report.agents.length) {
    throw new Error(`${origin}\n\nReport contains no agents.`);
  }
}

// ---------------------------------------------------------------------------

function render(report) {
  const agents = report.agents;
  const colors = report.agent_colors || agents.map(() => '#6fc3df');

  $('empty').hidden = true;
  $('content').hidden = false;
  $('subtitle').textContent =
    `${report.label ?? 'report'} · ${report.policy_mode ?? report.policy} · ` +
    `${report.episodes} episodes · seed ${report.seed}`;

  renderKpis(report);
  renderStats(agents, colors, report.policy_mode);
  renderMix(agents, colors);
  renderTerritory(report, colors);
  renderMatrices(report, colors);

  const warn = $('mapWarning');
  if (!report.fixed_map) {
    warn.hidden = false;
    warn.innerHTML = '<b>Resampled map.</b> Bushes were re-scattered every episode, so the ' +
      'territory heatmaps below average over different worlds and mean very little. ' +
      'Re-run without <code>--no-fixed-map</code> for comparable territory.';
  } else {
    warn.hidden = true;
  }
}

function renderKpis(report) {
  const offDiagonalMean = (m) => {
    let sum = 0, n = 0;
    for (let i = 0; i < m.length; i++)
      for (let j = 0; j < m.length; j++)
        if (i !== j) { sum += m[i][j]; n++; }
    return n ? sum / n : 0;
  };
  const actionDiv = offDiagonalMean(report.action_divergence);
  const terrDiv = offDiagonalMean(report.territory_divergence);
  const lifespans = report.agents.map((a) => a.lifespan);
  $('kpi').innerHTML = `
    <div><b>${actionDiv.toFixed(4)}</b>mean action divergence (bits)</div>
    <div><b>${terrDiv.toFixed(4)}</b>mean territory divergence (bits)</div>
    <div><b>${(Math.max(...lifespans) - Math.min(...lifespans)).toFixed(1)}</b>lifespan spread (ticks)</div>
    <div><b>${report.num_agents}</b>agents</div>`;
}

const SPREAD_NOTE =
  'The <span class="spread">spread</span> row is max&nbsp;−&nbsp;min: it is the quick ' +
  'answer to whether these agents are doing different things. Hit rate is the share of ' +
  '<code>gather</code> actions that actually took a berry.';

function renderStats(agents, colors, mode) {
  // A shared-brain report is the control, not the experiment; saying otherwise
  // would invite reading its (near-zero) divergence as a finding about six brains.
  $('behaviourNote').innerHTML = (mode === 'individual'
    ? 'Each agent has its own brain, forked from the shared Milestone&nbsp;1 policy and then ' +
      'trained independently. '
    : 'These agents all share <em>one</em> brain (Milestone&nbsp;1 parameter sharing), so this ' +
      'is the control: whatever spread appears here is spawn position and sampling noise, ' +
      'not specialisation. ') + SPREAD_NOTE;

  const value = (a, col) => a[col.key] * (col.pct ? 100 : 1);
  const head = `<thead><tr><th class="agent">agent</th>${
    STAT_COLUMNS.map((c) => `<th>${c.label}</th>`).join('')}</tr></thead>`;

  const rows = agents.map((a, i) => `<tr>
    <td class="agent"><span class="swatch" style="background:${colors[i]}"></span>${a.agent}</td>
    ${STAT_COLUMNS.map((c) => `<td>${value(a, c).toFixed(c.digits)}</td>`).join('')}
  </tr>`).join('');

  const spread = STAT_COLUMNS.map((c) => {
    const vals = agents.map((a) => value(a, c));
    return `<td class="spread">${(Math.max(...vals) - Math.min(...vals)).toFixed(c.digits)}</td>`;
  }).join('');

  $('statsTable').innerHTML = head + `<tbody>${rows}
    <tr><td class="agent spread">spread</td>${spread}</tr></tbody>`;
}

function renderMix(agents, colors) {
  $('mix').innerHTML = agents.map((a, i) => {
    const g = a.gather_share * 100, t = a.travel_share * 100, idle = a.idle_share * 100;
    return `<div class="mixrow">
      <div><span class="swatch" style="background:${colors[i]}"></span>agent ${a.agent}</div>
      <div class="mixbar" title="gather ${g.toFixed(1)}% · travel ${t.toFixed(1)}% · idle ${idle.toFixed(1)}%">
        <div style="width:${g}%;background:#6ec46e"></div>
        <div style="width:${t}%;background:#6fc3df"></div>
        <div style="width:${idle}%;background:#7c8b9c"></div>
      </div></div>`;
  }).join('');
}

// --- territory heatmaps -----------------------------------------------------

function hexToRgb(hex) {
  const n = parseInt(hex.replace('#', ''), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function renderTerritory(report, colors) {
  const bins = report.territory_bins;
  const size = 176;
  const cell = size / bins;

  $('territoryNote').textContent =
    'Where each agent spent its living ticks, top-down. Each map is normalised to its ' +
    'own peak, so these show the shape of an agent\'s range rather than how much time ' +
    'it logged overall. Grey dots are bushes.';

  $('territory').innerHTML = report.agents.map((a, i) => `
    <div class="card">
      <h3><span class="swatch" style="background:${colors[i]}"></span>agent ${a.agent}</h3>
      <canvas id="terr${i}" width="${size}" height="${size}"></canvas>
      <div class="meta">peak ${(Math.max(...a.territory.flat()) * 100).toFixed(1)}% of its time in one cell</div>
    </div>`).join('');

  report.agents.forEach((a, i) => {
    const ctx = $(`terr${i}`).getContext('2d');
    const [r, g, b] = hexToRgb(colors[i]);
    const radius = size / 2;

    ctx.fillStyle = '#0f151c';
    ctx.fillRect(0, 0, size, size);

    // Clip to the island disc so the heat cannot bleed into the sea.
    ctx.save();
    ctx.beginPath();
    ctx.arc(radius, radius, radius - 1, 0, Math.PI * 2);
    ctx.clip();
    ctx.fillStyle = '#182430';
    ctx.fillRect(0, 0, size, size);

    const peak = Math.max(...a.territory.flat()) || 1;
    for (let bz = 0; bz < bins; bz++) {
      for (let bx = 0; bx < bins; bx++) {
        const v = a.territory[bz][bx] / peak;
        if (v <= 0) continue;
        ctx.fillStyle = `rgba(${r},${g},${b},${Math.min(v * 0.92 + 0.08, 1)})`;
        // +z is north in world space; canvas y grows downward, so flip the row.
        ctx.fillRect(bx * cell, (bins - 1 - bz) * cell, cell + 0.5, cell + 0.5);
      }
    }

    if (report.bushes?.length) {
      const scale = radius / report.island_radius;
      ctx.fillStyle = 'rgba(190,205,220,0.55)';
      for (const bush of report.bushes) {
        ctx.beginPath();
        ctx.arc(radius + bush.x * scale, radius - bush.z * scale, 1.9, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();

    ctx.strokeStyle = '#2b3946';
    ctx.beginPath();
    ctx.arc(radius, radius, radius - 1, 0, Math.PI * 2);
    ctx.stroke();
  });
}

// --- divergence matrices ----------------------------------------------------

function renderMatrices(report, colors) {
  const build = (title, matrix) => {
    const n = matrix.length;
    const flat = matrix.flatMap((row, i) => row.filter((_, j) => i !== j));
    const peak = Math.max(...flat, 1e-9);
    let html = `<div><h3 style="font-size:12px;color:var(--dim);margin:0">${title}</h3>`;
    html += `<div class="matrix" style="grid-template-columns:repeat(${n + 1},auto)">`;
    html += '<div class="head"></div>';
    for (let j = 0; j < n; j++) html += `<div class="head">${j}</div>`;
    for (let i = 0; i < n; i++) {
      html += `<div class="head" style="color:${colors[i]}">${i}</div>`;
      for (let j = 0; j < n; j++) {
        if (i === j) {
          html += '<div style="background:#161e27;color:#3d4b5a">·</div>';
        } else {
          const alpha = 0.10 + 0.80 * (matrix[i][j] / peak);
          html += `<div style="background:rgba(111,195,223,${alpha.toFixed(3)})">` +
            `${matrix[i][j].toFixed(3)}</div>`;
        }
      }
    }
    return html + '</div></div>';
  };

  $('matrices').innerHTML =
    build('actions', report.action_divergence) +
    build('territory', report.territory_divergence);
}

// --- loading ----------------------------------------------------------------

function load(report, origin) {
  try {
    validate(report, origin);
    render(report);
  } catch (err) {
    fail(err.message);
  }
}

function readFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      load(JSON.parse(reader.result), file.name);
    } catch (err) {
      fail(`${file.name}\n\n${err.message}`);
    }
  };
  reader.readAsText(file);
}

$('fileInput').addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (file) readFile(file);
});

addEventListener('dragover', (e) => e.preventDefault());
addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer?.files?.[0];
  if (file) readFile(file);
});

fetch('reports/divergence.json', { cache: 'no-store' })
  .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
  .then((report) => load(report, 'reports/divergence.json'))
  .catch(() => { /* no report yet, or opened over file:// — the empty state explains it */ });
