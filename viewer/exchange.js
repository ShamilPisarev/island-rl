// Exchange view (Milestone 5).
//
// Renders the JSON report written by sim/exchange.py. The replay viewer shows
// transfers as they happen, one at a time; this is the other half of the same
// question -- what the whole block of episodes added up to, and whether the
// pattern looks like trade or like churn.

const SUPPORTED_SCHEMA = [1, 2];   // 2 adds the island2 stage-4 household block
const $ = (id) => document.getElementById(id);

const LEDGER_COLUMNS = [
  { key: 'lifespan', label: 'lifespan', digits: 1 },
  { key: 'given', label: 'gave', digits: 0 },
  { key: 'received', label: 'received', digits: 0 },
  { key: 'net', label: 'net', digits: 0, signed: true },
  { key: 'given_food', label: 'food out', digits: 0 },
  { key: 'given_material', label: 'material out', digits: 0 },
  { key: 'give_share', label: 'give %', digits: 2, pct: true },
  { key: 'give_success_rate', label: 'hit rate %', digits: 1, pct: true },
  { key: 'berries_gathered', label: 'berries', digits: 0 },
  { key: 'materials_harvested', label: 'harvested', digits: 0 },
  { key: 'deliveries', label: 'delivered', digits: 0 },
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
  const colors = report.agent_colors || report.agents.map(() => '#6fc3df');

  $('empty').hidden = true;
  $('content').hidden = false;
  $('subtitle').textContent =
    `${report.label ?? 'report'} · ${report.policy_mode ?? report.policy} · ` +
    `${report.episodes} episodes · seed ${report.seed}`;

  renderKpis(report);
  renderHouseholds(report);
  renderLedger(report, colors);
  renderMatrices(report, colors);
  renderRoles(report, colors);

  // A paid gift is a different experiment from an unpaid one, and the flow
  // matrix looks superficially the same either way. Say which this is.
  const warn = $('shapedWarning');
  if (report.reward_give > 0) {
    warn.hidden = false;
    warn.innerHTML = `<b>Shaped run.</b> A successful transfer paid ` +
      `<code>reward.give = ${report.reward_give}</code>, so the counts below include ` +
      `whatever giving is worth <em>as a reward</em>. Compare against the unpaid run ` +
      `on lifespan, not on the number of transfers.`;
  } else {
    warn.hidden = true;
  }
}

function renderKpis(report) {
  const t = report.totals;
  $('kpi').innerHTML = `
    <div><b>${t.transfers_per_episode.toFixed(1)}</b>transfers per episode</div>
    <div><b>${t.food} / ${t.material}</b>food / material</div>
    <div><b>${(t.food_utilisation * 100).toFixed(0)}% · ${(t.material_utilisation * 100).toFixed(0)}%</b>used within ${report.use_window} ticks</div>
    <div><b>${report.reciprocity.toFixed(3)}</b>reciprocity</div>`;
}

function renderLedger(report, colors) {
  $('ledgerNote').innerHTML =
    'Counts over the whole block, not per episode. <b>net</b> is gave&nbsp;−&nbsp;received: ' +
    'a persistent donor and a persistent recipient are what one-directional trade would ' +
    'look like, while everyone near zero means gifts are simply circulating. ' +
    '<b>hit rate</b> is the share of give actions that actually moved something.';

  const value = (a, c) => (a[c.key] ?? 0) * (c.pct ? 100 : 1);
  const head = `<thead><tr><th class="agent">agent</th>${
    LEDGER_COLUMNS.map((c) => `<th>${c.label}</th>`).join('')}</tr></thead>`;
  const rows = report.agents.map((a, i) => `<tr>
    <td class="agent"><span class="swatch" style="background:${colors[i]}"></span>${a.agent}</td>
    ${LEDGER_COLUMNS.map((c) => {
      const v = value(a, c);
      const cls = c.signed ? (v > 0 ? ' class="pos"' : v < 0 ? ' class="neg"' : '') : '';
      return `<td${cls}>${c.signed && v > 0 ? '+' : ''}${v.toFixed(c.digits)}</td>`;
    }).join('')}
  </tr>`).join('');
  $('ledgerTable').innerHTML = head + `<tbody>${rows}</tbody>`;
}

// Island 2.0 stage 4. The household ledger, and it leads with WITHIN vs ACROSS
// because that is the question a group-level view exists to answer: a gift to a
// housemate is housekeeping, and only one that crosses a boundary is trade. M5's
// standing lesson is that a busy flow matrix can mean nothing at all, so the
// headline here is not volume.
function renderHouseholds(report) {
  const host = $('households');
  const h = report.households;
  if (!host) return;
  if (!h) { host.hidden = true; return; }
  host.hidden = false;

  const raids = h.raids;
  const flow = h.flow;
  const n = h.count;
  const total = h.gifts_within + h.gifts_across;
  const across = Math.round(100 * h.gifts_across / Math.max(total, 1));
  const raidTotal = raids.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0);

  let html = `<div class="kpi">
    <div><b>${h.gifts_across}</b>gifts across a household boundary</div>
    <div><b>${across}%</b>of all gifts crossed one</div>
    <div><b>${raidTotal}</b>raids</div>
    <div><b>${h.raid_reciprocity.toFixed(2)}</b>raid reciprocity</div>
  </div>`;

  const peak = Math.max(...raids.flat(), 1);
  html += `<h3 style="font-size:12px;color:var(--dim);margin:14px 0 4px">raids (row raided column)</h3>`;
  html += `<div class="scrollx"><div class="matrix" style="grid-template-columns:repeat(${n + 1},auto)">`;
  html += '<div class="head"></div>';
  for (let j = 0; j < n; j++) html += `<div class="head" style="color:${h.colors[j]}">${j}</div>`;
  for (let i = 0; i < n; i++) {
    html += `<div class="head" style="color:${h.colors[i]}">${i}</div>`;
    for (let j = 0; j < n; j++) {
      if (i === j) { html += '<div style="background:#161e27;color:#3d4b5a">·</div>'; continue; }
      const alpha = raids[i][j] ? 0.12 + 0.78 * (raids[i][j] / peak) : 0.04;
      html += `<div style="background:rgba(255,89,74,${alpha.toFixed(3)})">${raids[i][j]}</div>`;
    }
  }
  html += '</div></div>';

  const rows = [];
  for (let j = 0; j < n; j++) {
    const took = raids[j].reduce((a, b) => a + b, 0);
    const lost = raids.reduce((s, row) => s + row[j], 0);
    const gave = flow[j].reduce((a, b) => a + b, 0) - flow[j][j];
    const got = flow.reduce((s, row) => s + row[j], 0) - flow[j][j];
    rows.push(`<tr><td><span class="swatch" style="background:${h.colors[j]}"></span>${j}</td>` +
      `<td>${h.deposits[j]}</td><td>${h.withdrawals[j]}</td>` +
      `<td>${took}</td><td>${lost}</td><td>${gave}</td><td>${got}</td></tr>`);
  }
  html += `<div class="scrollx"><table style="margin-top:14px"><thead><tr><th>household</th><th>deposited</th>` +
    `<th>withdrew</th><th>raided</th><th>was raided</th><th>gave</th><th>received</th>` +
    `</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
  host.innerHTML = html;
}

function renderMatrices(report, colors) {
  const build = (title, matrix) => {
    const n = matrix.length;
    const peak = Math.max(...matrix.flat(), 1);
    let html = `<div><h3 style="font-size:12px;color:var(--dim);margin:0">${title}</h3>`;
    html += `<div class="matrix" style="grid-template-columns:repeat(${n + 1},auto)">`;
    html += '<div class="head"></div>';
    for (let j = 0; j < n; j++) html += `<div class="head" style="color:${colors[j]}">${j}</div>`;
    for (let i = 0; i < n; i++) {
      html += `<div class="head" style="color:${colors[i]}">${i}</div>`;
      for (let j = 0; j < n; j++) {
        if (i === j) {
          html += '<div style="background:#161e27;color:#3d4b5a">·</div>';
        } else {
          const alpha = matrix[i][j] ? 0.12 + 0.78 * (matrix[i][j] / peak) : 0.04;
          html += `<div style="background:rgba(111,195,223,${alpha.toFixed(3)})">` +
            `${matrix[i][j]}</div>`;
        }
      }
    }
    return html + '</div></div>';
  };

  const material = report.flow.map((row, i) => row.map((v, j) => v - report.flow_food[i][j]));
  $('matrices').innerHTML =
    build('all transfers', report.flow) +
    build('food only', report.flow_food) +
    build('material only', material);
}

function renderRoles(report, colors) {
  const agents = report.agents;
  const totalGiven = Math.max(agents.reduce((s, a) => s + a.given, 0), 1);
  $('roles').innerHTML = agents.map((a, i) => {
    const parts = [
      { key: 'harvested', color: '#8a5a2b', v: a.harvest_share },
      { key: 'delivered', color: '#e0b83c', v: a.delivery_share },
      { key: 'given away', color: '#6fc3df', v: a.given / totalGiven },
    ];
    const title = parts.map((p) => `${p.key} ${(p.v * 100).toFixed(1)}%`).join(' · ');
    // Each segment is this agent's share of a population total, so the three
    // bars are comparable down a column rather than across a row.
    return `<div class="rolerow">
      <div><span class="swatch" style="background:${colors[i]}"></span>agent ${a.agent}</div>
      <div class="rolebar" title="${title}">
        ${parts.map((p) => `<div style="width:${(p.v * 100 / 3).toFixed(2)}%;background:${p.color}"></div>`).join('')}
      </div></div>`;
  }).join('');
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

fetch('reports/exchange.json', { cache: 'no-store' })
  .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
  .then((report) => load(report, 'reports/exchange.json'))
  .catch(() => { /* no report yet, or opened over file:// — the empty state explains it */ });
