"""The knowledge explorer: a local, self-contained graph UI.

Renders the whole graph - repository, files, symbols, memories and every edge
between them - as one HTML file with the data inlined, then serves it on
localhost and opens a browser.

Two constraints shape it:

  * No network, no build step, no npm. The visualisation is hand-written
    canvas plus a force layout in about two hundred lines. A knowledge tool
    that needs a toolchain to look at its own knowledge will not get looked at.

  * The file is self-contained. Saving it, mailing it, or committing it all
    work, because the data is embedded rather than fetched.

Canvas rather than SVG: a repository with a few thousand symbols is tens of
thousands of DOM nodes in SVG, and the browser stops being interactive long
before the graph stops being useful.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
import webbrowser
from pathlib import Path
from typing import Any

# Node palettes are picked for meaning, not decoration: memories are warm so
# knowledge stands out against the cool structural nodes, and severity shifts
# hue rather than size so a critical warning reads at any zoom.
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ - Knowledge Explorer</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --line: #21262d; --text: #e6edf3;
    --muted: #8b949e; --accent: #58a6ff; --warn: #f0883e; --bad: #f85149;
    --good: #3fb950; --mem: #d2a8ff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 13px/1.5 ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif;
         overflow: hidden; }
  #app { display: grid; grid-template-columns: 268px 1fr 340px; height: 100vh; }
  aside { background: var(--panel); border-right: 1px solid var(--line);
          overflow-y: auto; padding: 14px; }
  #side { border-right: 0; border-left: 1px solid var(--line); }
  h1 { font-size: 14px; margin: 0 0 2px; letter-spacing: .2px; }
  h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .09em;
       color: var(--muted); margin: 18px 0 8px; font-weight: 600; }
  .sub { color: var(--muted); font-size: 11px; margin-bottom: 14px; }
  input[type=search] { width: 100%; padding: 7px 9px; border-radius: 6px;
    border: 1px solid var(--line); background: #0d1117; color: var(--text); font: inherit; }
  label { display: flex; align-items: center; gap: 7px; padding: 3px 0;
          cursor: pointer; user-select: none; }
  label:hover { color: #fff; }
  .swatch { width: 10px; height: 10px; border-radius: 3px; flex: none; }
  .count { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
  canvas { display: block; cursor: grab; }
  canvas:active { cursor: grabbing; }
  #detail { white-space: pre-wrap; word-break: break-word; }
  .pill { display: inline-block; padding: 1px 7px; border-radius: 999px;
          font-size: 10px; border: 1px solid var(--line); margin: 0 4px 4px 0; }
  .kv { display: grid; grid-template-columns: 84px 1fr; gap: 3px 8px;
        margin: 8px 0; font-size: 12px; }
  .kv span:first-child { color: var(--muted); }
  .body { background: #0d1117; border: 1px solid var(--line); border-radius: 6px;
          padding: 9px; margin-top: 9px; font-size: 12px; max-height: 260px;
          overflow: auto; }
  .rel { padding: 4px 0; border-bottom: 1px solid var(--line); cursor: pointer; font-size: 12px; }
  .rel:hover { color: var(--accent); }
  .rel small { color: var(--muted); }
  button { background: #21262d; color: var(--text); border: 1px solid var(--line);
           border-radius: 6px; padding: 6px 10px; font: inherit; cursor: pointer; }
  button:hover { background: #30363d; }
  #hud { position: fixed; bottom: 12px; left: 282px; color: var(--muted);
         font-size: 11px; background: rgba(13,17,23,.86); padding: 5px 10px;
         border-radius: 6px; border: 1px solid var(--line); }
  .empty { color: var(--muted); font-style: italic; }
</style>
</head>
<body>
<div id="app">
  <aside>
    <h1>__TITLE__</h1>
    <div class="sub" id="stats"></div>
    <input type="search" id="q" placeholder="Search nodes...">
    <h2>Node types</h2><div id="kinds"></div>
    <h2>Memory severity</h2><div id="sevs"></div>
    <h2>Anchor status</h2><div id="anchors"></div>
    <h2>Edges</h2><div id="edges"></div>
    <h2>Layout</h2>
    <label><input type="checkbox" id="isolate"> Hide unconnected</label>
    <label><input type="checkbox" id="labels" checked> Show labels</label>
    <div style="margin-top:10px; display:flex; gap:6px;">
      <button id="refit">Fit</button><button id="reheat">Re-layout</button>
    </div>
  </aside>
  <main style="position:relative"><canvas id="c"></canvas></main>
  <aside id="side">
    <h2>Selection</h2>
    <div id="detail" class="empty">Click a node to inspect it.</div>
  </aside>
</div>
<div id="hud">drag to pan &middot; scroll to zoom &middot; click a node to inspect</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const GRAPH = JSON.parse(document.getElementById('data').textContent);
const KIND_COLOR = { repo:'#58a6ff', file:'#39c5cf', symbol:'#7ee787',
                     memory:'#d2a8ff', event:'#8b949e' };
const SEV_COLOR = { critical:'#f85149', high:'#f0883e', medium:'#d29922', low:'#8b949e' };
const canvas = document.getElementById('c'), ctx = canvas.getContext('2d');

let nodes = GRAPH.nodes.map(n => ({...n, x: 0, y: 0, vx: 0, vy: 0}));
const byId = new Map(nodes.map(n => [n.id, n]));
let edges = GRAPH.edges.filter(e => byId.has(e.from) && byId.has(e.to));

// Degree drives node size: a symbol that everything calls should look like one.
const degree = new Map();
for (const e of edges) {
  degree.set(e.from, (degree.get(e.from) || 0) + 1);
  degree.set(e.to, (degree.get(e.to) || 0) + 1);
}
nodes.forEach(n => { n.deg = degree.get(n.id) || 0;
                     n.r = 4 + Math.min(11, Math.sqrt(n.deg) * 2.1); });

const state = { kinds:new Set(), sevs:new Set(), anchors:new Set(), edgeKinds:new Set(),
                q:'', isolate:false, labels:true, sel:null,
                zoom:1, panX:0, panY:0 };

function tally(list, key) {
  const m = new Map();
  for (const item of list) { const k = key(item); if (k) m.set(k, (m.get(k)||0)+1); }
  return [...m.entries()].sort((a,b) => b[1]-a[1]);
}

function checkboxes(host, entries, set, colorOf) {
  entries.forEach(([value, count]) => set.add(value));
  host.innerHTML = entries.map(([value, count]) => `
    <label><input type="checkbox" data-v="${value}" checked>
    ${colorOf ? `<span class="swatch" style="background:${colorOf(value)}"></span>` : ''}
    <span>${value}</span><span class="count">${count}</span></label>`).join('');
  host.querySelectorAll('input').forEach(box => box.onchange = () => {
    box.checked ? set.add(box.dataset.v) : set.delete(box.dataset.v);
    draw();
  });
}

checkboxes(document.getElementById('kinds'),
           tally(nodes, n => n.kind), state.kinds, v => KIND_COLOR[v] || '#8b949e');
checkboxes(document.getElementById('sevs'),
           tally(nodes.filter(n => n.kind === 'memory'), n => n.detail?.severity),
           state.sevs, v => SEV_COLOR[v] || '#8b949e');
checkboxes(document.getElementById('anchors'),
           tally(nodes.filter(n => n.kind === 'memory'), n => n.detail?.anchor_status),
           state.anchors, v => v === 'ACTIVE' ? '#3fb950' : '#f0883e');
checkboxes(document.getElementById('edges'), tally(edges, e => e.kind), state.edgeKinds, null);

document.getElementById('stats').textContent =
  `${GRAPH.stats.nodes} nodes | ${GRAPH.stats.edges} edges | exported ${(GRAPH.exported_at||'').slice(0,10)}`;

function visible(n) {
  if (!state.kinds.has(n.kind)) return false;
  if (n.kind === 'memory') {
    if (n.detail?.severity && !state.sevs.has(n.detail.severity)) return false;
    if (n.detail?.anchor_status && !state.anchors.has(n.detail.anchor_status)) return false;
  }
  if (state.q) {
    const hay = (n.label + ' ' + JSON.stringify(n.detail || {})).toLowerCase();
    if (!hay.includes(state.q)) return false;
  }
  return true;
}

let shown = [], shownEdges = [];
function recompute() {
  shown = nodes.filter(visible);
  const ok = new Set(shown.map(n => n.id));
  shownEdges = edges.filter(e => state.edgeKinds.has(e.kind) && ok.has(e.from) && ok.has(e.to));
  if (state.isolate) {
    const linked = new Set();
    shownEdges.forEach(e => { linked.add(e.from); linked.add(e.to); });
    shown = shown.filter(n => linked.has(n.id));
  }
}

// Force layout. Barnes-Hut would be faster, but a grid approximation keeps
// this readable and holds up past the point the labels stop being legible.
// Tuned against a real 509-node graph: with a fixed repulsion the layout
// collapsed into an unreadable ball. Both scale with node count so a small
// graph stays compact and a large one still separates.
const REPULSION = 900 + nodes.length * 6;
const LINK = 58 + Math.min(70, nodes.length / 9);
let alpha = 1;
function step() {
  if (alpha < 0.005) return;
  const cell = 130, grid = new Map();
  for (const n of shown) {
    const k = `${Math.round(n.x/cell)},${Math.round(n.y/cell)}`;
    (grid.get(k) || grid.set(k, []).get(k)).push(n);
  }
  for (const n of shown) {
    const gx = Math.round(n.x/cell), gy = Math.round(n.y/cell);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      for (const m of grid.get(`${gx+dx},${gy+dy}`) || []) {
        if (m === n) continue;
        let ddx = n.x-m.x, ddy = n.y-m.y;
        let d2 = ddx*ddx + ddy*ddy || 0.01;
        if (d2 > 90000) continue;
        const f = REPULSION / d2;
        n.vx += ddx * f * alpha; n.vy += ddy * f * alpha;
      }
    }
    n.vx -= n.x * 0.0009 * alpha; n.vy -= n.y * 0.0009 * alpha;
  }
  for (const e of shownEdges) {
    const a = byId.get(e.from), b = byId.get(e.to);
    if (!a || !b) continue;
    const dx = b.x-a.x, dy = b.y-a.y, d = Math.hypot(dx, dy) || 0.01;
    const f = (d - LINK) * 0.006 * alpha;
    const ux = dx/d*f, uy = dy/d*f;
    a.vx += ux; a.vy += uy; b.vx -= ux; b.vy -= uy;
  }
  for (const n of shown) { n.x += n.vx *= 0.82; n.y += n.vy *= 0.82; }
  alpha *= 0.985;
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const main = canvas.parentElement;
  canvas.width = main.clientWidth * dpr;
  canvas.height = main.clientHeight * dpr;
  canvas.style.width = main.clientWidth + 'px';
  canvas.style.height = main.clientHeight + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function colorOf(n) {
  if (n.kind === 'memory') {
    if (n.detail?.anchor_status && n.detail.anchor_status !== 'ACTIVE') return '#f0883e';
    return SEV_COLOR[n.detail?.severity] || KIND_COLOR.memory;
  }
  if (n.status && n.status !== 'ACTIVE') return '#484f58';
  return KIND_COLOR[n.kind] || '#8b949e';
}

function draw() {
  recompute(); step(); resize();
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.translate(w/2 + state.panX, h/2 + state.panY);
  ctx.scale(state.zoom, state.zoom);

  ctx.lineWidth = 1 / state.zoom;
  for (const e of shownEdges) {
    const a = byId.get(e.from), b = byId.get(e.to);
    const hot = state.sel && (e.from === state.sel.id || e.to === state.sel.id);
    ctx.strokeStyle = hot ? 'rgba(88,166,255,.85)'
                          : (e.status && e.status !== 'ACTIVE' ? 'rgba(248,81,73,.22)'
                                                               : 'rgba(139,148,158,.17)');
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }

  for (const n of shown) {
    ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, 6.284);
    ctx.fillStyle = colorOf(n); ctx.fill();
    if (state.sel && state.sel.id === n.id) {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2 / state.zoom; ctx.stroke();
    }
  }

  if (state.labels && state.zoom > 0.55) {
    ctx.fillStyle = '#c9d1d9';
    ctx.font = `${Math.max(9, 11/state.zoom)}px ui-monospace, monospace`;
    for (const n of shown) {
      if (n.deg < 2 && state.zoom < 1.1) continue;
      const text = (n.label || '').slice(0, 34);
      ctx.fillText(text, n.x + n.r + 3, n.y + 3);
    }
  }
  ctx.restore();
  document.getElementById('hud').textContent =
    `${shown.length} nodes | ${shownEdges.length} edges | drag to pan, scroll to zoom`;
}

function seed() {
  const n = nodes.length;
  nodes.forEach((node, i) => {
    const a = i * 2.399963, r = 40 * Math.sqrt(i + 1);
    node.x = Math.cos(a) * r; node.y = Math.sin(a) * r;
  });
  alpha = 1;
}

function span(values) {
  // 2nd-98th percentile. One stray node must not decide the viewport for the
  // other five hundred - with raw min/max the cluster ended up off-screen.
  const sorted = [...values].sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length * 0.02)];
  const hi = sorted[Math.ceil(sorted.length * 0.98) - 1];
  return [lo, hi];
}

function fit() {
  recompute();
  if (!shown.length) return;
  const [x0, x1] = span(shown.map(n => n.x));
  const [y0, y1] = span(shown.map(n => n.y));
  const w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
  state.zoom = Math.min(2.4, Math.max(0.06,
    0.82 * Math.min(canvas.clientWidth / w, canvas.clientHeight / h)));
  state.panX = -((x0 + x1) / 2) * state.zoom;
  state.panY = -((y0 + y1) / 2) * state.zoom;
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"]/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function select(node) {
  state.sel = node;
  const host = document.getElementById('detail');
  if (!node) { host.className = 'empty'; host.textContent = 'Click a node to inspect it.'; return; }
  host.className = '';
  const d = node.detail || {};
  const related = edges.filter(e => e.from === node.id || e.to === node.id).slice(0, 40);

  host.innerHTML = `
    <div><span class="pill" style="border-color:${colorOf(node)};color:${colorOf(node)}">${escapeHtml(node.kind)}</span>
    ${d.severity ? `<span class="pill">${escapeHtml(d.severity)}</span>` : ''}
    ${d.anchor_status && d.anchor_status !== 'ACTIVE'
        ? `<span class="pill" style="border-color:var(--warn);color:var(--warn)">${escapeHtml(d.anchor_status)}</span>` : ''}
    ${node.status && node.status !== 'ACTIVE' ? `<span class="pill">${escapeHtml(node.status)}</span>` : ''}</div>
    <div style="font-weight:600;margin:9px 0 2px">${escapeHtml(node.label)}</div>
    <div class="kv">
      ${Object.entries(d).filter(([k]) => k !== 'body')
        .map(([k,v]) => `<span>${escapeHtml(k)}</span><span>${escapeHtml(v)}</span>`).join('')}
      <span>id</span><span style="font-family:ui-monospace,monospace">${escapeHtml(node.id)}</span>
    </div>
    ${d.body ? `<div class="body">${escapeHtml(d.body)}</div>` : ''}
    <h2>Connections (${related.length})</h2>
    ${related.map(e => {
      const other = byId.get(e.from === node.id ? e.to : e.from);
      const dir = e.from === node.id ? '&rarr;' : '&larr;';
      return `<div class="rel" data-id="${other?.id}">${dir}
        <small>${escapeHtml(e.kind)}</small> ${escapeHtml(other?.label || '?')}</div>`;
    }).join('') || '<div class="empty">none</div>'}`;

  host.querySelectorAll('.rel').forEach(row => row.onclick = () => {
    const next = byId.get(row.dataset.id);
    if (next) { select(next); state.panX = -next.x*state.zoom; state.panY = -next.y*state.zoom; }
  });
  draw();
}

let drag = null;
canvas.addEventListener('mousedown', ev => drag = {x: ev.clientX, y: ev.clientY, moved: false});
addEventListener('mousemove', ev => {
  if (!drag) return;
  state.panX += ev.clientX - drag.x; state.panY += ev.clientY - drag.y;
  drag = {x: ev.clientX, y: ev.clientY, moved: true};
  draw();
});
addEventListener('mouseup', ev => {
  if (drag && !drag.moved) {
    const rect = canvas.getBoundingClientRect();
    const x = (ev.clientX - rect.left - canvas.clientWidth/2 - state.panX) / state.zoom;
    const y = (ev.clientY - rect.top - canvas.clientHeight/2 - state.panY) / state.zoom;
    let best = null, bestD = 1e9;
    for (const n of shown) {
      const d = Math.hypot(n.x-x, n.y-y);
      if (d < Math.max(n.r + 6, 11) && d < bestD) { best = n; bestD = d; }
    }
    select(best);
  }
  drag = null;
});
canvas.addEventListener('wheel', ev => {
  ev.preventDefault();
  state.zoom = Math.min(5, Math.max(0.05, state.zoom * (ev.deltaY < 0 ? 1.12 : 0.89)));
  draw();
}, {passive: false});

document.getElementById('q').oninput = ev => {
  state.q = ev.target.value.trim().toLowerCase(); alpha = Math.max(alpha, .3); draw();
};
document.getElementById('isolate').onchange = ev => { state.isolate = ev.target.checked; draw(); };
document.getElementById('labels').onchange = ev => { state.labels = ev.target.checked; draw(); };
document.getElementById('refit').onclick = () => { fit(); draw(); };
document.getElementById('reheat').onclick = () => { seed(); draw(); };
addEventListener('resize', draw);

seed();
for (let i = 0; i < 220; i++) { recompute(); step(); }
fit();
(function loop() { draw(); requestAnimationFrame(loop); })();
</script>
</body>
</html>
"""


def render(graph: dict[str, Any]) -> str:
    """Inline the graph into the page. Self-contained, no fetch at runtime."""
    payload = json.dumps(graph, ensure_ascii=False).replace("</", "<\\/")
    title = graph.get("repo_name") or graph.get("repo_id") or "Knowledge"
    return PAGE.replace("__DATA__", payload).replace("__TITLE__", _escape(title))


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def write_html(graph: dict[str, Any], target: Path) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(graph), encoding="utf-8")
    return target


def serve(html: Path, port: int = 0, open_browser: bool = True) -> None:
    """Serve one file on localhost until interrupted.

    A file:// URL would avoid the server entirely, but browsers treat local
    files inconsistently and a real origin keeps behaviour predictable. Port 0
    lets the OS pick, so two explorers never collide.
    """
    html = Path(html).resolve()
    directory = str(html.parent)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, *args):    # keep the terminal for the user
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as server:
        actual = server.server_address[1]
        url = f"http://127.0.0.1:{actual}/{html.name}"
        print(f"  Knowledge explorer: {url}")
        print("  Press Ctrl+C to stop.")
        if open_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n  Stopped.")
