"use strict";
const GRAPH = JSON.parse(document.getElementById('data').textContent);

const KIND = {
  repo:   {c:'#fbbf24', label:'repository'},
  file:   {c:'#4d7cfe', label:'file'},
  symbol: {c:'#34d399', label:'symbol'},
  memory: {c:'#8b5cf6', label:'memory'},
  event:  {c:'#7b849f', label:'event'},
};
const SEV = { critical:'#f4677c', high:'#fb923c', medium:'#fbbf24', low:'#7b849f' };

// Structure is quiet, knowledge is loud. Anchor edges are what make this a
// knowledge graph rather than a call graph, so they get colour and weight
// while CALLS/DEFINES recede into context.
const EDGE = {
  ANCHORED_TO:{c:'#8b5cf6', w:1.7, dash:false, glow:true},
  APPLIES_TO: {c:'#8b5cf6', w:1.3, dash:false},
  GUARDED_BY: {c:'#34d399', w:1.3, dash:true},
  IMPACTS:    {c:'#8a6a3f', w:.9, dash:true},
  CAUSED:     {c:'#f4677c', w:1.8, dash:false, glow:true},
  LED_TO:     {c:'#f4677c', w:1.8, dash:false, glow:true},
  ESTABLISHED:{c:'#f4677c', w:1.4, dash:true},
  CONTRADICTS:{c:'#f4677c', w:1.6, dash:true},
  SUPERSEDES: {c:'#7b849f', w:1.2, dash:true},
  CALLS:      {c:'#39436e', w:.8,  dash:false},
  DEFINES:    {c:'#2e3760', w:.7,  dash:false},
  CONTAINS:   {c:'#272f52', w:.6,  dash:false},
  IMPORTS:    {c:'#2e3760', w:.7,  dash:true},
};
const edgeStyle = k => EDGE[k] || {c:'#39436e', w:.8, dash:false};

const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const app = document.getElementById('app'), tip = document.getElementById('tip');

const nodes = GRAPH.nodes.map(n => ({...n, x:0, y:0, vx:0, vy:0}));
const byId = new Map(nodes.map(n => [n.id, n]));
const edges = GRAPH.edges.filter(e => byId.has(e.from) && byId.has(e.to));

const adj = new Map();
for (const e of edges) {
  (adj.get(e.from) || adj.set(e.from, []).get(e.from)).push(e);
  (adj.get(e.to)   || adj.set(e.to,   []).get(e.to)).push(e);
}
nodes.forEach(n => {
  n.deg = (adj.get(n.id) || []).length;
  // Memories get a floor: a warning with one anchor still matters more than a
  // helper with one caller, and equal-sized dots made the graph unreadable.
  n.r = n.kind === 'memory' ? 7 + Math.min(7, Math.sqrt(n.deg) * 1.9)
       : n.kind === 'file'   ? 5 + Math.min(8, Math.sqrt(n.deg) * 1.5)
       : n.kind === 'repo'   ? 13
                             : 4.2 + Math.min(10, Math.sqrt(n.deg) * 2.1);
});

const S = { kinds:new Set(), sevs:new Set(), anchors:new Set(), rels:new Set(),
            q:'', labels:true, sel:null, hover:null, zoom:1, px:0, py:0, focus:null };

/* ------------------------------------------------------------------ filters */
function tally(list, key) {
  const m = new Map();
  for (const x of list) { const k = key(x); if (k) m.set(k, (m.get(k)||0)+1); }
  return [...m].sort((a,b) => b[1]-a[1]);
}
function chips(host, entries, set, mark, defaultOff = () => false) {
  entries.forEach(([v]) => { if (!defaultOff(v)) set.add(v); });
  host.innerHTML = entries.map(([v,n]) => `
    <div class="row ${defaultOff(v) ? 'off' : ''}" data-v="${v}">${mark(v)}
      <span class="name">${v.replace(/_/g,' ').toLowerCase()}</span>
      <span class="n">${n.toLocaleString()}</span></div>`).join('');
  host.querySelectorAll('.row').forEach(row => row.onclick = () => {
    const v = row.dataset.v;
    if (set.has(v)) { set.delete(v); row.classList.add('off'); }
    else { set.add(v); row.classList.remove('off'); }
    kick(.22);
  });
}
const dot = c => `<span class="dot" style="background:${c}"></span>`;
const bar = s => `<span class="bar" style="background:${s.c};${s.dash?'opacity:.6':''}"></span>`;

chips(document.getElementById('kinds'), tally(nodes, n => n.kind), S.kinds,
      v => dot(KIND[v]?.c || '#8b96a8'), v => v === 'symbol');
chips(document.getElementById('sevs'),
      tally(nodes.filter(n => n.kind==='memory'), n => n.detail?.severity), S.sevs,
      v => dot(SEV[v] || '#7d8797'));
chips(document.getElementById('anchors'),
      tally(nodes.filter(n => n.kind==='memory'), n => n.detail?.anchor_status), S.anchors,
      v => dot(v === 'ACTIVE' ? '#34d399' : '#fb923c'));
chips(document.getElementById('edges'), tally(edges, e => e.kind), S.rels,
      v => bar(edgeStyle(v)));

document.getElementById('stats').textContent =
  `${GRAPH.stats.nodes.toLocaleString()} nodes · ${GRAPH.stats.edges.toLocaleString()} edges`;

document.getElementById('legend').innerHTML =
  Object.entries(KIND).filter(([k]) => nodes.some(n => n.kind === k))
    .map(([k,v]) => `<span><i style="background:${v.c}"></i><b>${v.label}</b></span>`).join('') +
  `<span><span class="ln" style="border-color:#8b5cf6"></span>anchored</span>` +
  `<span><span class="ln dash" style="border-color:#34d399"></span>guarded by</span>` +
  `<span><span class="ln" style="border-color:#39436e"></span>calls</span>` +
  `<span style="color:var(--faint)">size = connections</span>`;

/* ------------------------------------------------------------------ visible */
let vis = [], vedges = [];
function recompute() {
  const q = S.q;
  vis = nodes.filter(n => {
    if (!S.kinds.has(n.kind)) return false;
    if (n.kind === 'memory') {
      if (n.detail?.severity && !S.sevs.has(n.detail.severity)) return false;
      if (n.detail?.anchor_status && !S.anchors.has(n.detail.anchor_status)) return false;
    }
    if (q) { n.hit = (n.label + ' ' + JSON.stringify(n.detail||{})).toLowerCase().includes(q);
             if (!n.hit) return false; }
    return true;
  });
  const ok = new Set(vis.map(n => n.id));
  vedges = edges.filter(e => S.rels.has(e.kind) && ok.has(e.from) && ok.has(e.to));
}

/* ------------------------------------------------------------------- layout */
const REP = 2100 + nodes.length * 13;
const LINK = 74 + Math.min(95, nodes.length / 7);
let alpha = 1;
const kick = a => { alpha = Math.max(alpha, a); };

function step() {
  if (alpha < .004) return;
  const cell = 150, grid = new Map();
  for (const n of vis) {
    const k = `${Math.round(n.x/cell)},${Math.round(n.y/cell)}`;
    (grid.get(k) || grid.set(k, []).get(k)).push(n);
  }
  for (const n of vis) {
    const gx = Math.round(n.x/cell), gy = Math.round(n.y/cell);
    for (let dx=-1; dx<=1; dx++) for (let dy=-1; dy<=1; dy++)
      for (const m of grid.get(`${gx+dx},${gy+dy}`) || []) {
        if (m === n) continue;
        const ddx = n.x-m.x, ddy = n.y-m.y, d2 = ddx*ddx + ddy*ddy || .01;
        if (d2 > 120000) continue;
        const f = REP / d2;
        n.vx += ddx*f*alpha; n.vy += ddy*f*alpha;
      }
    n.vx -= n.x * .0011 * alpha; n.vy -= n.y * .0011 * alpha;
  }
  for (const e of vedges) {
    const a = byId.get(e.from), b = byId.get(e.to);
    const dx = b.x-a.x, dy = b.y-a.y, d = Math.hypot(dx,dy) || .01;
    // Anchors pull harder, so a memory sits with the code it describes and
    // clusters form around knowledge rather than around call fan-out.
    const pull = (e.kind === 'ANCHORED_TO' || e.kind === 'APPLIES_TO') ? .013 : .0055;
    const f = (d - LINK) * pull * alpha, ux = dx/d*f, uy = dy/d*f;
    a.vx += ux; a.vy += uy; b.vx -= ux; b.vy -= uy;
  }
  for (const n of vis) { n.x += n.vx *= .8; n.y += n.vy *= .8; }
  alpha *= .986;
}

/* --------------------------------------------------------------------- draw */
function resize() {
  const dpr = window.devicePixelRatio || 1, host = cv.parentElement;
  cv.width = host.clientWidth*dpr; cv.height = host.clientHeight*dpr;
  cv.style.width = host.clientWidth+'px'; cv.style.height = host.clientHeight+'px';
  ctx.setTransform(dpr,0,0,dpr,0,0);
}
function nodeColor(n) {
  if (n.kind === 'memory') {
    if (n.detail?.anchor_status && n.detail.anchor_status !== 'ACTIVE') return '#fb923c';
    return SEV[n.detail?.severity] || KIND.memory.c;
  }
  if (n.status && n.status !== 'ACTIVE') return '#3b4470';
  return KIND[n.kind]?.c || '#8b96a8';
}
function neighbourhood(id) {
  const set = new Set([id]);
  for (const e of adj.get(id) || []) { set.add(e.from); set.add(e.to); }
  return set;
}

function draw() {
  recompute(); step(); resize();
  const w = cv.clientWidth, h = cv.clientHeight;
  ctx.clearRect(0,0,w,h);
  ctx.save();
  ctx.translate(w/2 + S.px, h/2 + S.py);
  ctx.scale(S.zoom, S.zoom);

  const focus = S.focus ? neighbourhood(S.focus) : null;

  for (const e of vedges) {
    const a = byId.get(e.from), b = byId.get(e.to), st = edgeStyle(e.kind);
    const lit = focus && (focus.has(e.from) && focus.has(e.to));
    if (focus && !lit) { ctx.globalAlpha = .05; } else { ctx.globalAlpha = lit ? .95 : .34; }
    ctx.strokeStyle = st.c;
    ctx.lineWidth = (lit ? st.w*1.7 : st.w) / S.zoom;
    ctx.setLineDash(st.dash ? [4/S.zoom, 3/S.zoom] : []);
    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();

    // Direction only where it carries meaning and only when readable.
    if (lit && S.zoom > .5 && st.glow) {
      const dx = b.x-a.x, dy = b.y-a.y, d = Math.hypot(dx,dy)||1;
      const tx = b.x - dx/d*(b.r+3), ty = b.y - dy/d*(b.r+3), ang = Math.atan2(dy,dx);
      const s = 6/S.zoom;
      ctx.setLineDash([]); ctx.fillStyle = st.c;
      ctx.beginPath();
      ctx.moveTo(tx,ty);
      ctx.lineTo(tx - s*Math.cos(ang-.42), ty - s*Math.sin(ang-.42));
      ctx.lineTo(tx - s*Math.cos(ang+.42), ty - s*Math.sin(ang+.42));
      ctx.closePath(); ctx.fill();
    }
  }
  ctx.setLineDash([]); ctx.globalAlpha = 1;

  for (const n of vis) {
    const lit = !focus || focus.has(n.id);
    ctx.globalAlpha = lit ? 1 : .1;
    const col = nodeColor(n);

    if (n.kind === 'memory' && lit) {
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r+4.5, 0, 6.2832);
      ctx.fillStyle = col + '26'; ctx.fill();
    }
    ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, 6.2832);
    ctx.fillStyle = col; ctx.fill();

    // A memory the anchor cannot vouch for wears a ring, at any zoom.
    if (n.kind === 'memory' && n.detail?.anchor_status &&
        n.detail.anchor_status !== 'ACTIVE' && lit) {
      ctx.strokeStyle = '#fb923c'; ctx.lineWidth = 1.8/S.zoom;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r+3.2, 0, 6.2832); ctx.stroke();
    }
    if (S.sel?.id === n.id || S.hover?.id === n.id) {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2/S.zoom;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r+2.6, 0, 6.2832); ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;

  // Labels are the thing that made the old view unreadable, so they are
  // rationed: only what is important enough to earn the space, and never
  // overlapping - a claimed-box test drops a label rather than letting two
  // collide.
  if (S.labels) {
    const boxes = [];
    const budget = S.zoom > 1.5 ? 90 : S.zoom > .9 ? 55 : 26;
    const rank = n => (S.sel?.id === n.id ? 1e6 : 0) + (focus?.has(n.id) ? 1e5 : 0)
                    + (n.kind === 'memory' ? 900 : 0) + (n.hit ? 800 : 0) + n.deg;
    const pool = [...vis].sort((a,b) => rank(b)-rank(a)).slice(0, budget);

    ctx.font = `500 ${Math.max(9.5, 11.5/S.zoom)}px ui-sans-serif,system-ui,sans-serif`;
    ctx.textBaseline = 'middle';
    for (const n of pool) {
      if (focus && !focus.has(n.id)) continue;
      let text = (n.label || '').split('\n')[0];
      if (text.length > 42) text = text.slice(0,40) + '…';
      const wpx = ctx.measureText(text).width, hpx = 13/S.zoom;
      const x = n.x + n.r + 5/S.zoom, y = n.y;
      const box = [x, y-hpx/2, x+wpx, y+hpx/2];
      if (boxes.some(b => !(box[2]<b[0] || box[0]>b[2] || box[3]<b[1] || box[1]>b[3]))) continue;
      boxes.push(box);
      ctx.fillStyle = 'rgba(20,25,48,.8)';
      ctx.fillRect(x-2/S.zoom, y-hpx/2, wpx+4/S.zoom, hpx);
      ctx.fillStyle = n.kind === 'memory' ? '#ddd2fb' : '#aab3d4';
      ctx.fillText(text, x, y+.5);
    }
  }
  ctx.restore();
  document.getElementById('count').textContent =
    `${vis.length.toLocaleString()} shown · ${vedges.length.toLocaleString()} links`;
}

/* ---------------------------------------------------------------- inspector */
const esc = s => String(s ?? '').replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const repositories = GRAPH.repositories || [];
if (repositories.length > 1) {
  const picker = document.getElementById('repo-picker'), select = document.getElementById('repo');
  picker.hidden = false;
  for (const repository of repositories) {
    const option = document.createElement('option');
    option.value = repository.href;
    option.textContent = `${repository.name || repository.repo_id} (${repository.memories || 0})`;
    option.selected = repository.repo_id === GRAPH.repo_id;
    select.appendChild(option);
  }
  select.onchange = () => { if (select.value) location.href = select.value; };
}
const REL_LABEL = { ANCHORED_TO:'anchored to', APPLIES_TO:'applies to',
  GUARDED_BY:'guarded by', IMPACTS:'impacts', CALLS:'calls', DEFINES:'defines',
  CONTAINS:'contains', CAUSED:'caused', LED_TO:'led to', ESTABLISHED:'established',
  CONTRADICTS:'contradicts', SUPERSEDES:'supersedes', IMPORTS:'imports' };

function bodyBelowTitle(n, d) {
  // record() stores the summary as both title and first body line, so showing
  // the body verbatim repeats the heading directly under itself.
  const body = (d.body || '').trim();
  if (!body) return '';
  const title = (n.label || '').trim();
  if (body === title) return '';
  if (body.startsWith(title)) {
    const rest = body.slice(title.length).replace(/^\s+/, '');
    return rest || '';
  }
  return body;
}

function inspect(n) {
  S.sel = n; S.focus = n ? n.id : null;
  const tags = document.getElementById('shead-tags'), body = document.getElementById('sbody');
  if (!n) {
    app.classList.remove('open');
    return draw();
  }
  app.classList.add('open');

  const d = n.detail || {}, col = nodeColor(n);
  const stale = d.anchor_status && d.anchor_status !== 'ACTIVE';
  tags.innerHTML =
    `<span class="tag" style="color:${col};border-color:${col}55;background:${col}14">
       ${esc(d.memory_kind || n.kind)}</span>` +
    (d.severity ? `<span class="tag" style="color:${SEV[d.severity]};
       border-color:${SEV[d.severity]}55;background:${SEV[d.severity]}14">${esc(d.severity)}</span>` : '') +
    (stale ? `<span class="tag" style="color:#ff9f43;border-color:#ff9f4355;
       background:#ff9f4314">${esc(d.anchor_status)}</span>` : '');

  const rels = (adj.get(n.id) || []).map(e => {
    const other = byId.get(e.from === n.id ? e.to : e.from);
    return {e, other, out: e.from === n.id};
  }).filter(r => r.other);

  // Knowledge relationships first: what governs this, what it guards, what it
  // caused. Call structure is real but it is not why anyone opened the panel.
  const PRIORITY = ['ANCHORED_TO','APPLIES_TO','GUARDED_BY','CAUSED','LED_TO',
                    'ESTABLISHED','CONTRADICTS','SUPERSEDES','IMPACTS'];
  const groups = new Map();
  for (const r of rels) (groups.get(r.e.kind) || groups.set(r.e.kind, []).get(r.e.kind)).push(r);
  const ordered = [...groups.keys()].sort((a,b) => {
    const ia = PRIORITY.indexOf(a), ib = PRIORITY.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });

  const isPath = n.kind === 'file' || (n.kind === 'symbol' && d.path);
  body.innerHTML = `
    <div class="title">${esc((n.label||'').split('\n')[0])}</div>
    ${isPath ? `<div class="path">${esc(d.path || n.label)}${d.lines ? ':'+esc(d.lines) : ''}</div>` : ''}
    ${bodyBelowTitle(n, d) ? `<div class="quote">${esc(bodyBelowTitle(n, d))}</div>` : ''}
    ${stale ? `<div class="quote warn">
        Not verified against the current code. Treat this as a lead, not a fact.</div>` : ''}
    ${n.kind === 'memory' && d.authority === 'agent' ? `<div class="quote warn">
        Agent-supplied claim. Its anchor verifies location, not semantic correctness.</div>` : ''}
    <dl class="meta">
      ${Object.entries(d).filter(([k,v]) => k!=='body' && k!=='path' && v!=null && v!=='')
        .map(([k,v]) => `<dt>${esc(k.replace(/_/g,' '))}</dt><dd>${esc(v)}</dd>`).join('')}
      <dt>connections</dt><dd>${rels.length}</dd>
    </dl>
    ${ordered.map(kind => `
      <div class="group">
        <h3>${esc(REL_LABEL[kind] || kind.replace(/_/g,' ').toLowerCase())}
            &middot; ${groups.get(kind).length}</h3>
        ${groups.get(kind).slice(0,24).map(r => `
          <div class="link" data-id="${r.other.id}">
            ${dot(nodeColor(r.other))}
            <span class="lbl">${esc((r.other.label||'').split('\n')[0])}</span>
            <span class="rel">${r.out ? '→' : '←'}</span>
          </div>`).join('')}
      </div>`).join('') || '<p class="hint">No connections.</p>'}`;

  body.querySelectorAll('.link').forEach(row => row.onclick = () => {
    const next = byId.get(row.dataset.id);
    if (!next) return;
    inspect(next);
    S.px = -next.x*S.zoom; S.py = -next.y*S.zoom;
  });
  draw();
}
document.getElementById('close').onclick = () => inspect(null);

/* ------------------------------------------------------------ interactions */
function toWorld(ev) {
  const r = cv.getBoundingClientRect();
  return { x:(ev.clientX-r.left-cv.clientWidth/2-S.px)/S.zoom,
           y:(ev.clientY-r.top-cv.clientHeight/2-S.py)/S.zoom };
}
function pick(ev) {
  const {x,y} = toWorld(ev);
  let best = null, bd = 1e9;
  for (const n of vis) {
    const d = Math.hypot(n.x-x, n.y-y);
    if (d < Math.max(n.r+7, 12) && d < bd) { best = n; bd = d; }
  }
  return best;
}

let drag = null;
cv.addEventListener('mousedown', ev => {
  drag = {x:ev.clientX, y:ev.clientY, moved:false}; cv.classList.add('drag');
});
addEventListener('mousemove', ev => {
  if (drag) {
    if (Math.abs(ev.clientX-drag.x) + Math.abs(ev.clientY-drag.y) > 2) drag.moved = true;
    S.px += ev.clientX-drag.x; S.py += ev.clientY-drag.y;
    drag.x = ev.clientX; drag.y = ev.clientY;
    tip.style.display = 'none';
    return;
  }
  if (ev.target !== cv) { tip.style.display='none'; S.hover=null; return; }
  const n = pick(ev);
  S.hover = n;
  cv.style.cursor = n ? 'pointer' : 'grab';
  if (!n) { tip.style.display = 'none'; return; }
  const d = n.detail || {};
  tip.innerHTML = `<span class="k">${esc(d.memory_kind || n.kind)}</span>
                   ${esc((n.label||'').split('\n')[0].slice(0,130))}`;
  tip.style.display = 'block';
  tip.style.left = Math.min(ev.clientX+14, innerWidth-356) + 'px';
  tip.style.top = (ev.clientY+16) + 'px';
});
addEventListener('mouseup', ev => {
  cv.classList.remove('drag');
  if (drag && !drag.moved && ev.target === cv) inspect(pick(ev));
  drag = null;
});
cv.addEventListener('wheel', ev => {
  ev.preventDefault();
  const before = toWorld(ev);
  S.zoom = Math.min(6, Math.max(.04, S.zoom * (ev.deltaY < 0 ? 1.13 : .885)));
  const after = toWorld(ev);
  S.px += (after.x-before.x)*S.zoom; S.py += (after.y-before.y)*S.zoom;
  tip.style.display = 'none';
}, {passive:false});

const q = document.getElementById('q');
q.oninput = () => { S.q = q.value.trim().toLowerCase(); kick(.3); };
addEventListener('keydown', ev => {
  if (ev.key === '/' && document.activeElement !== q) { ev.preventDefault(); q.focus(); }
  if (ev.key === 'Escape') { if (document.activeElement === q) { q.value=''; q.blur(); S.q=''; }
                             else inspect(null); }
});
document.getElementById('fit').onclick = () => fit();
document.getElementById('relayout').onclick = () => { seed(); settle(300); fit(); };
document.getElementById('lbl').onclick = ev => {
  S.labels = !S.labels; ev.currentTarget.classList.toggle('on', !S.labels);
};
addEventListener('resize', () => draw());

/* ------------------------------------------------------------------ export */
/* Everything here runs in the browser against the embedded graph. The markdown
   deliberately mirrors export.to_markdown() byte for byte, including the
   trailing JSON block, so a file saved from this page re-imports through
   `icn-explore import` exactly like one written by the CLI. Diverging would
   produce a file that looks right and silently fails to import. */
const MEM_ORDER = ['security','invariant','warning','contract','failed_attempt',
  'decision','bug_history','fix_history','migration','performance','convention',
  'rationale','test_evidence'];

function toast(msg) {
  let el = document.querySelector('.toast');
  if (!el) { el = document.createElement('div'); el.className = 'toast';
             document.body.appendChild(el); }
  el.textContent = msg;
  requestAnimationFrame(() => el.classList.add('show'));
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), 2600);
}

function download(name, blob) {
  const url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast('Saved ' + name);
}

const slug = s => (s || 'knowledge').replace(/[^a-z0-9._-]+/gi, '-').toLowerCase();

function toMarkdown(graph) {
  const mems = graph.nodes.filter(n => n.kind === 'memory');
  const out = [
    `# Knowledge: ${graph.repo_name}`, '',
    `Exported ${graph.exported_at} from \`${graph.repo_id}\`.`,
    `${mems.length} memories over ${graph.stats.nodes} nodes` +
      ` and ${graph.stats.edges} edges.`, '',
    'Import with `icn-explore import <this file>`.', '',
  ];

  const labels = new Map(graph.nodes.map(n => [n.id, n.label]));
  const anchored = new Map();
  for (const e of graph.edges) if (e.kind === 'ANCHORED_TO')
    (anchored.get(e.from) || anchored.set(e.from, []).get(e.from))
      .push(labels.get(e.to) || e.to);

  const byKind = new Map();
  for (const m of mems) {
    const k = m.detail?.memory_kind || 'rationale';
    (byKind.get(k) || byKind.set(k, []).get(k)).push(m);
  }
  const kinds = [...MEM_ORDER.filter(k => byKind.has(k)),
                 ...[...byKind.keys()].filter(k => !MEM_ORDER.includes(k)).sort()];

  for (const kind of kinds) {
    out.push('## ' + kind.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase()), '');
    for (const m of byKind.get(kind).sort((a,b) => (a.label||'').localeCompare(b.label||''))) {
      const d = m.detail || {};
      const flag = d.anchor_status === 'ACTIVE' ? '' : ` **[${d.anchor_status}]**`;
      out.push(`### ${m.label}${flag}`, '');
      out.push(`- severity: ${d.severity} | authority: ${d.authority}`);
      const at = anchored.get(m.id) || [];
      if (at.length) out.push('- applies to: ' + at.slice(0,8).map(a => '`'+a+'`').join(', '));
      out.push('');
      const body = (d.body || '').trim();
      if (body && body !== m.label) out.push(body, '');
    }
  }

  out.push('---', '', '<!-- infinite-code-next:graph -->', '```json',
           JSON.stringify(graph, null, 1), '```', '');
  return out.join('\n');
}

function visibleGraph() {
  // What is on screen, as a real graph: filters and search already decided
  // this, so exporting it is how a filtered view becomes a shareable subset.
  const ids = new Set(vis.map(n => n.id));
  const nodes = vis.map(({x,y,vx,vy,deg,r,hit,...keep}) => keep);
  const edges = vedges.filter(e => ids.has(e.from) && ids.has(e.to));
  const byNodeKind = {}, byEdgeKind = {};
  for (const n of nodes) byNodeKind[n.kind] = (byNodeKind[n.kind]||0)+1;
  for (const e of edges) byEdgeKind[e.kind] = (byEdgeKind[e.kind]||0)+1;
  return {...GRAPH, nodes, edges,
          exported_at: new Date().toISOString().replace(/\.(\d{3})Z$/, '.$1+00:00'),
          stats:{nodes:nodes.length, edges:edges.length,
                 by_node_kind:byNodeKind, by_edge_kind:byEdgeKind}};
}

function exportPng() {
  // Compose onto an opaque ground: the canvas is transparent, and a PNG that
  // renders as dark-on-dark everywhere except a white viewer is a trap.
  const out = document.createElement('canvas');
  out.width = cv.width; out.height = cv.height;
  const c = out.getContext('2d');
  c.fillStyle = getComputedStyle(document.documentElement)
                  .getPropertyValue('--stage').trim() || '#141930';
  c.fillRect(0, 0, out.width, out.height);
  c.drawImage(cv, 0, 0);
  out.toBlob(b => download(slug(GRAPH.repo_name) + '-graph.png', b), 'image/png');
}

const EXPORTS = {
  md:   () => download(slug(GRAPH.repo_name) + '-knowledge.md',
                new Blob([toMarkdown(GRAPH)], {type:'text/markdown;charset=utf-8'})),
  json: () => download(slug(GRAPH.repo_name) + '-graph.json',
                new Blob([JSON.stringify(GRAPH, null, 1)], {type:'application/json'})),
  html: () => download(slug(GRAPH.repo_name) + '-explorer.html',
                new Blob(['<!doctype html>\n' + document.documentElement.outerHTML],
                         {type:'text/html;charset=utf-8'})),
  png:  exportPng,
  clip: async () => {
    const md = toMarkdown(visibleGraph());
    try {
      await navigator.clipboard.writeText(md);
      toast(`Copied ${vis.length} nodes as markdown`);
    } catch {
      // Clipboard needs a secure context; falling back to a file is better
      // than telling someone their export vanished.
      download(slug(GRAPH.repo_name) + '-visible.md',
               new Blob([md], {type:'text/markdown;charset=utf-8'}));
    }
  },
};

const exportBtn = document.getElementById('exportBtn');
const exportMenu = document.getElementById('exportMenu');
function closeMenu() {
  exportMenu.hidden = true;
  exportBtn.classList.remove('on');
  exportBtn.setAttribute('aria-expanded', 'false');
}
exportBtn.onclick = ev => {
  ev.stopPropagation();
  const open = exportMenu.hidden;
  exportMenu.hidden = !open;
  exportBtn.classList.toggle('on', open);
  exportBtn.setAttribute('aria-expanded', String(open));
};
exportMenu.querySelectorAll('button[data-fmt]').forEach(b => b.onclick = () => {
  closeMenu();
  try { EXPORTS[b.dataset.fmt](); }
  catch (err) { toast('Export failed: ' + err.message); }
});
addEventListener('click', ev => {
  if (!exportMenu.hidden && !exportMenu.contains(ev.target)) closeMenu();
});

/* --------------------------------------------------------------- lifecycle */
function seed() {
  nodes.forEach((n,i) => {
    const a = i*2.399963, r = 46*Math.sqrt(i+1);
    n.x = Math.cos(a)*r; n.y = Math.sin(a)*r; n.vx = n.vy = 0;
  });
  alpha = 1;
}
function settle(n) { for (let i=0;i<n;i++) { recompute(); step(); } }
function pct(vals, p) {
  const s = [...vals].sort((a,b)=>a-b);
  return s[Math.min(s.length-1, Math.max(0, Math.floor(s.length*p)))];
}
function fit() {
  recompute();
  if (!vis.length) return;
  // Percentile bounds: one far-flung node must not decide the viewport for
  // everything else.
  const x0 = pct(vis.map(n=>n.x), .02), x1 = pct(vis.map(n=>n.x), .98);
  const y0 = pct(vis.map(n=>n.y), .02), y1 = pct(vis.map(n=>n.y), .98);
  const w = Math.max(1, x1-x0), h = Math.max(1, y1-y0);
  S.zoom = Math.min(2.2, Math.max(.05,
    .82*Math.min(cv.clientWidth/w, cv.clientHeight/h)));
  S.px = -((x0+x1)/2)*S.zoom; S.py = -((y0+y1)/2)*S.zoom;
}

if (!nodes.length) {
  document.querySelector('.stage').innerHTML =
    `<div class="empty"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/>
     <path d="M9 12h6"/></svg><div>Nothing indexed yet.</div></div>`;
} else {
  resize(); seed(); settle(480); alpha = .05; settle(140); fit();
  (function loop(){ draw(); requestAnimationFrame(loop); })();
}
