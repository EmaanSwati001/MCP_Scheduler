/* MCG-Sched Warm Editorial 60fps Animated Simulator */

const SESSION_ID = "sim_" + Math.random().toString(36).slice(2, 10);

let META = { traces: [], topologies: [], schedulers: [] };
let SNAPSHOT = null;
let FRAME_BUFFER = [];
let IS_FETCHING_BATCH = false;
let IS_PLAYING = false;
let LAST_STEP_AT = 0;
let PREV_NOW = 0;
let DISPLAY_NOW = 0;
let MAP_TOPOLOGY_SIG = "";
let LAST_COMPARE_DATA = null;
let SAVED_RESEARCH = null;
let CHARTS = {};

const PARTICLES = [];
const RIPPLES = [];
const CHART_PALETTE = ["#b8532f", "#2c6e49", "#b77213", "#475569", "#7c3aed", "#0e7490"];

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  lucide.createIcons();
  await loadMetadata();
  await onScenarioChange(true);
  requestAnimationFrame(animationLoop);
  runN2Inspector().catch(() => {});
  loadSavedResearch().catch(() => {});
});

async function apiGet(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiPost(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function fmtShortTime(sec) {
  if (sec === null || sec === undefined || isNaN(sec)) return "0s";
  if (sec >= 10000) return `${(sec / 1000).toFixed(1)}k s`;
  return `${Math.round(sec).toLocaleString()}s`;
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------
async function loadMetadata() {
  META = await apiGet("/api/meta");

  const traceOpts = META.traces
    .filter(t => t.available)
    .map(t => `<option value="${t.id}">${t.name}</option>`)
    .join("");
  document.getElementById("live-trace").innerHTML = traceOpts;
  document.getElementById("cmp-trace").innerHTML = traceOpts;

  const topoOpts = META.topologies
    .map(t => `<option value="${t.id}">${t.name}</option>`)
    .join("");
  document.getElementById("live-topo").innerHTML = topoOpts;
  document.getElementById("cmp-topo").innerHTML = topoOpts;

  document.getElementById("live-sched").innerHTML = META.schedulers
    .map(s => `<option value="${s.id}" ${s.id === "mcg" ? "selected" : ""}>${s.name}</option>`)
    .join("");

  const defCmp = new Set(["fcfs", "consolidate", "fgd", "mcg", "mcg_fair", "mcg_fc"]);
  document.getElementById("cmp-sched-chips").innerHTML = META.schedulers
    .map(s => `<button type="button" class="policy-chip ${defCmp.has(s.id) ? "selected" : ""}" data-sched="${s.id}" onclick="this.classList.toggle('selected')">${s.id}</button>`)
    .join("");

  renderPolicyParams();
}

function renderPolicyParams() {
  const id = document.getElementById("live-sched").value;
  const s = META.schedulers.find(x => x.id === id);
  const box = document.getElementById("live-params-box");
  if (!s || !s.params || s.params.length === 0) {
    box.innerHTML = "";
    return;
  }
  const defs = {
    w: { label: "Weight w", val: 0.5, step: 0.05, min: 0, max: 1 },
    lam: { label: "Blend λ", val: 1.0, step: 0.1, min: 0, max: 10 },
    k: { label: "Guard k", val: 1.0, step: 0.5, min: 0.2, max: 10 },
    beta: { label: "Aging β", val: 1.0, step: 0.25, min: 0, max: 5 },
  };
  box.innerHTML = s.params
    .filter(p => defs[p])
    .map(p => {
      const d = defs[p];
      return `<div><label class="field-lbl">${d.label}</label><input id="param-${p}" type="number" value="${d.val}" step="${d.step}" min="${d.min}" max="${d.max}" class="inp mono w-20" /></div>`;
    })
    .join("");
}

function readParams() {
  const id = document.getElementById("live-sched").value;
  const s = META.schedulers.find(x => x.id === id);
  const out = {};
  (s?.params || []).forEach(p => {
    const el = document.getElementById(`param-${p}`);
    if (el && el.value !== "") out[p] = parseFloat(el.value);
  });
  return out;
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
function switchTab(tabId) {
  document.querySelectorAll(".top-tab").forEach(b => b.classList.toggle("active", b.dataset.tab === tabId));
  document.querySelectorAll(".tab-view").forEach(v => v.classList.toggle("hidden", v.id !== `tab-${tabId}`));
  if (tabId === "research" && SAVED_RESEARCH) showSavedStudy("e5");
}

function toggleAdvancedDrawer() {
  document.getElementById("advanced-drawer").classList.toggle("hidden");
}

// ---------------------------------------------------------------------------
// Scenarios & Real Simulator Control
// ---------------------------------------------------------------------------
async function onScenarioChange(autoPlayAfter = false) {
  stopPlay();
  const sc = document.getElementById("live-scenario").value;
  let seekMode = "peak";

  if (sc === "venus_peak") {
    document.getElementById("live-trace").value = "venus";
    document.getElementById("live-njobs").value = 0;
    document.getElementById("live-eta").value = 15;
    document.getElementById("live-burst").checked = false;
    document.getElementById("live-topo").value = "venus15";
    document.getElementById("live-sched").value = "mcg";
    seekMode = "peak";
  } else if (sc === "burst_800") {
    document.getElementById("live-trace").value = "synthetic";
    document.getElementById("live-njobs").value = 800;
    document.getElementById("live-eta").value = 10;
    document.getElementById("live-burst").checked = true;
    document.getElementById("live-topo").value = "venus15";
    document.getElementById("live-sched").value = "mcg";
    seekMode = "none";
  } else if (sc === "fairness_n1") {
    document.getElementById("live-trace").value = "venus";
    document.getElementById("live-njobs").value = 0;
    document.getElementById("live-eta").value = 15;
    document.getElementById("live-burst").checked = false;
    document.getElementById("live-topo").value = "venus15";
    document.getElementById("live-sched").value = "mcg_fair";
    seekMode = "reserve";
  } else if (sc === "mixed_n2") {
    document.getElementById("live-trace").value = "venus";
    document.getElementById("live-njobs").value = 0;
    document.getElementById("live-eta").value = 15;
    document.getElementById("live-burst").checked = false;
    document.getElementById("live-topo").value = "venus15_mixed";
    document.getElementById("live-sched").value = "mcg_fc";
    seekMode = "peak";
  } else if (sc === "philly_std") {
    document.getElementById("live-trace").value = "philly";
    document.getElementById("live-njobs").value = 0;
    document.getElementById("live-eta").value = 4;
    document.getElementById("live-burst").checked = false;
    document.getElementById("live-topo").value = "paper_8x16";
    document.getElementById("live-sched").value = "mcg";
    seekMode = "peak";
  }

  renderPolicyParams();
  await initSimulation(false, seekMode);
  if (autoPlayAfter) {
    startPlay();
  }
}

async function onPolicyChange() {
  renderPolicyParams();
  const sc = document.getElementById("live-scenario").value;
  const sched = document.getElementById("live-sched").value;
  const seekMode = sc === "burst_800" ? "none" : (sched === "mcg_fair" ? "reserve" : "peak");
  await initSimulation(false, seekMode);
}

function setStatus(state, text) {
  const dot = document.getElementById("status-dot");
  dot.className = "w-2 h-2 rounded-full " + (
    state === "running" ? "bg-[#b8532f] animate-ping" :
    state === "done" ? "bg-[#2c6e49]" : "bg-[#b77213]"
  );
  document.getElementById("status-text").textContent = text;
}

async function initSimulation(keepTune = false, seekOverride = null) {
  stopPlay();
  FRAME_BUFFER = [];
  PARTICLES.length = 0;
  RIPPLES.length = 0;

  const sc = document.getElementById("live-scenario").value;
  const seekMode = seekOverride !== null ? seekOverride : (
    keepTune ? "none" : (sc === "burst_800" ? "none" : (sc === "fairness_n1" ? "reserve" : "peak"))
  );

  SNAPSHOT = await apiPost("/api/sim/init", {
    session_id: SESSION_ID,
    trace_id: document.getElementById("live-trace").value,
    topo_id: document.getElementById("live-topo").value,
    sched_id: document.getElementById("live-sched").value,
    n_jobs: parseInt(document.getElementById("live-njobs").value || "0", 10),
    eta: parseFloat(document.getElementById("live-eta").value || "15"),
    burst: document.getElementById("live-burst").checked,
    seed: 0,
    params: readParams(),
    seek_mode: seekMode,
  });

  PREV_NOW = SNAPSHOT.now;
  DISPLAY_NOW = SNAPSHOT.now;
  setStatus("idle", "Ready");
  applySnapshot(SNAPSHOT, false);
}

async function stepSimulation(count = 1) {
  if (!SNAPSHOT || SNAPSHOT.done) {
    stopPlay();
    return;
  }
  FRAME_BUFFER = [];
  const nextSnap = await apiPost("/api/sim/step", { session_id: SESSION_ID, count });
  applySnapshot(nextSnap, true);
}

async function runToEnd() {
  stopPlay();
  FRAME_BUFFER = [];
  if (!SNAPSHOT) return;
  setStatus("running", "Completing...");
  const btn = document.getElementById("btn-run-end");
  if (btn) btn.disabled = true;
  try {
    const endSnap = await apiPost("/api/sim/run", { session_id: SESSION_ID });
    applySnapshot(endSnap, false);
    setStatus("done", "Complete");
  } finally {
    if (btn) btn.disabled = SNAPSHOT?.done ?? false;
  }
}

function togglePlay() {
  if (IS_PLAYING) stopPlay();
  else startPlay();
}

function startPlay() {
  if (!SNAPSHOT || SNAPSHOT.done) return;
  IS_PLAYING = true;
  LAST_STEP_AT = performance.now();
  const btn = document.getElementById("btn-play");
  btn.innerHTML = `<i data-lucide="pause" class="w-3.5 h-3.5"></i> Pause`;
  btn.classList.remove("btn-primary");
  btn.classList.add("btn-ochre");
  lucide.createIcons();
  setStatus("running", "Running");
  prefetchBatch();
}

function stopPlay() {
  IS_PLAYING = false;
  const btn = document.getElementById("btn-play");
  if (btn) {
    btn.innerHTML = `<i data-lucide="play" class="w-3.5 h-3.5"></i> Play`;
    btn.classList.add("btn-primary");
    btn.classList.remove("btn-ochre");
    lucide.createIcons();
  }
  if (SNAPSHOT && !SNAPSHOT.done) setStatus("idle", "Paused");
}

async function prefetchBatch() {
  if (!IS_PLAYING || IS_FETCHING_BATCH || (SNAPSHOT && SNAPSHOT.done)) return;
  if (FRAME_BUFFER.length >= 10) return;
  IS_FETCHING_BATCH = true;
  try {
    const res = await apiPost("/api/sim/step_batch", { session_id: SESSION_ID, count: 16 });
    if (res.frames && res.frames.length) {
      FRAME_BUFFER.push(...res.frames);
    }
  } catch (e) {
    console.error(e);
    stopPlay();
  } finally {
    IS_FETCHING_BATCH = false;
  }
}

// ---------------------------------------------------------------------------
// 60 FPS Animation & Buffered Playback Loop
// ---------------------------------------------------------------------------
function animationLoop(nowMs) {
  requestAnimationFrame(animationLoop);

  if (IS_PLAYING && SNAPSHOT && !SNAPSHOT.done) {
    const speedEl = document.getElementById("play-speed");
    const interval = parseInt(speedEl ? speedEl.value : "180", 10);
    const elapsed = nowMs - LAST_STEP_AT;

    if (FRAME_BUFFER.length > 0) {
      const nextTarget = FRAME_BUFFER[0].now;
      const frac = Math.min(1, elapsed / interval);
      DISPLAY_NOW = PREV_NOW + (nextTarget - PREV_NOW) * frac;
      document.getElementById("header-clock").textContent = `t = ${DISPLAY_NOW.toFixed(1)} s`;
    }

    if (elapsed >= interval) {
      LAST_STEP_AT = nowMs;
      if (FRAME_BUFFER.length > 0) {
        const frame = FRAME_BUFFER.shift();
        applySnapshot(frame, true);
        if (FRAME_BUFFER.length < 6) prefetchBatch();
      } else if (!IS_FETCHING_BATCH) {
        prefetchBatch();
      }
    }
  }

  drawFlowCanvas();
}

function applySnapshot(snap, animateParticles) {
  PREV_NOW = SNAPSHOT ? SNAPSHOT.now : snap.now;
  DISPLAY_NOW = snap.now;
  SNAPSHOT = snap;

  document.getElementById("header-clock").textContent = `t = ${snap.now.toFixed(1)} s`;
  if (snap.done) {
    stopPlay();
    setStatus("done", "Complete");
  }

  const m = snap.metrics;
  document.getElementById("kpi-util").textContent = `${m.utilization_pct.toFixed(1)}%`;
  document.getElementById("kpi-gpus-sub").textContent = `${m.used_gpus.toLocaleString()} / ${m.total_gpus.toLocaleString()} GPUs`;
  document.getElementById("util-progress-bar").style.width = `${m.utilization_pct}%`;

  document.getElementById("kpi-wait-cnt").textContent = `${m.waiting} queued`;
  document.getElementById("kpi-avg-wait").textContent = `Avg wait: ${fmtShortTime(snap.summary ? snap.summary.avg_wait : m.avg_wait)}`;
  document.getElementById("kpi-done").textContent = `${m.completed.toLocaleString()} / ${snap.total_jobs.toLocaleString()} done`;

  document.getElementById("kpi-lb").textContent = (snap.summary ? snap.summary.avg_lb_index : m.lb_index).toFixed(4);
  const resBadge = document.getElementById("kpi-res-badge");
  if (m.reserved_count > 0) {
    resBadge.textContent = `${m.reserved_count} reserved`;
    resBadge.classList.remove("hidden");
  } else {
    resBadge.classList.add("hidden");
  }

  document.getElementById("running-mini").textContent = `${m.running} running`;

  document.getElementById("btn-step1").disabled = snap.done;
  document.getElementById("btn-play").disabled = snap.done;
  const btnEnd = document.getElementById("btn-run-end");
  if (btnEnd) btnEnd.disabled = snap.done;

  renderQueueBlocks();
  renderDecisionSpotlight();
  updateClusterMapInPlace(animateParticles);
  renderLiveChart();

  if (animateParticles) {
    spawnDispatchParticles();
  }
}

// ---------------------------------------------------------------------------
// Clean 5x3 Cluster Capacity Bars (Zero Pixel Clutter)
// ---------------------------------------------------------------------------
function ensureClusterMapBuilt() {
  const s = SNAPSHOT;
  if (!s) return;
  const sig = s.clusters.map(c => c.cluster_id + ":" + c.servers.map(sv => sv.capacity).join(",")).join("|");
  if (sig === MAP_TOPOLOGY_SIG) return;
  MAP_TOPOLOGY_SIG = sig;

  let cap4 = 0, cap8 = 0;
  s.clusters.forEach(c => c.servers.forEach(sv => { if (sv.capacity <= 4) cap4++; else cap8++; }));
  document.getElementById("topo-mix-tag").textContent =
    cap4 > 0 ? `${cap4}×4G + ${cap8}×8G Mixed` : `${cap8}×8G Homogeneous`;

  document.getElementById("cluster-map").innerHTML = s.clusters.map(c => {
    const bars = c.servers.map(sv =>
      `<div id="srv-${c.cluster_id}_${sv.server_id}" class="srv-bar ${sv.capacity <= 4 ? "cap4" : ""}">
        <div id="sfill-${c.cluster_id}_${sv.server_id}" class="srv-fill" style="width:0%"></div>
      </div>`
    ).join("");

    return `<div id="ctile-${c.cluster_id}" class="c-tile">
      <div class="flex items-center justify-between text-[11px]">
        <span class="font-bold text-[#1d1b18]">C${c.cluster_id}</span>
        <span id="cpct-${c.cluster_id}" class="mono text-[#5c574e]">0%</span>
      </div>
      <div class="w-full h-1.5 bg-[#e5dfd3] rounded-full overflow-hidden">
        <div id="cbar-${c.cluster_id}" class="h-full bg-[#2c6e49] transition-all duration-200" style="width:0%"></div>
      </div>
      <div class="srv-bars-wrap">${bars}</div>
    </div>`;
  }).join("");
}

function updateClusterMapInPlace(flashChanges) {
  ensureClusterMapBuilt();
  const s = SNAPSHOT;
  if (!s) return;

  const startedKeys = new Set(
    (s.step_started || []).filter(x => x.placement).map(x => `${x.placement[0]}_${x.placement[1]}`)
  );
  const finishedKeys = new Set(
    (s.step_finished || []).filter(x => x.placement).map(x => `${x.placement[0]}_${x.placement[1]}`)
  );

  s.clusters.forEach(c => {
    const uPct = Math.round(c.utilization * 100);
    const pctEl = document.getElementById(`cpct-${c.cluster_id}`);
    const barEl = document.getElementById(`cbar-${c.cluster_id}`);
    const tileEl = document.getElementById(`ctile-${c.cluster_id}`);
    if (pctEl) pctEl.textContent = `${uPct}%`;
    if (barEl) barEl.style.width = `${uPct}%`;

    let hasRes = false;
    c.servers.forEach(sv => {
      const key = `${c.cluster_id}_${sv.server_id}`;
      const srvEl = document.getElementById(`srv-${key}`);
      const fillEl = document.getElementById(`sfill-${key}`);
      if (!srvEl || !fillEl) return;

      const isRes = sv.reserved_for !== null;
      if (isRes) hasRes = true;

      const fillPct = sv.capacity > 0 ? Math.round((sv.used / sv.capacity) * 100) : 0;
      fillEl.style.width = `${fillPct}%`;

      srvEl.classList.toggle("reserved", isRes);
      if (flashChanges) {
        srvEl.classList.remove("flash-place", "flash-free");
        if (startedKeys.has(key)) {
          void srvEl.offsetWidth;
          srvEl.classList.add("flash-place");
        } else if (finishedKeys.has(key)) {
          void srvEl.offsetWidth;
          srvEl.classList.add("flash-free");
        }
      }

      const runStr = sv.running.length
        ? sv.running.map(r => `Job #${r.job_id} (${r.gpus}G, ${r.remaining}s left)`).join(" · ")
        : "Idle";
      srvEl.title = `Cluster ${c.cluster_id} · Server ${sv.server_id} (${sv.capacity}G) | ${sv.used}/${sv.capacity} GPUs | ${runStr}` +
        (isRes ? ` | RESERVED for Job #${sv.reserved_for}` : "");
    });

    if (tileEl) tileEl.classList.toggle("has-reserve", hasRes);
  });
}

// ---------------------------------------------------------------------------
// 60 FPS Flying Job Packet Canvas Animation
// ---------------------------------------------------------------------------
function spawnDispatchParticles() {
  const stage = document.getElementById("sim-stage");
  const qBox = document.getElementById("queue-blocks");
  if (!stage || !qBox || !SNAPSHOT) return;

  const started = (SNAPSHOT.step_started || []).filter(x => x.placement).slice(0, 6);
  if (started.length === 0) return;

  const stageRect = stage.getBoundingClientRect();
  const qRect = qBox.getBoundingClientRect();
  const sx = qRect.right - stageRect.left - 10;
  const sy = qRect.top - stageRect.top + 38;

  started.forEach((item, idx) => {
    const [cid, sid] = item.placement;
    const srv = document.getElementById(`srv-${cid}_${sid}`);
    if (!srv) return;
    const sRect = srv.getBoundingClientRect();
    const tx = sRect.left - stageRect.left + sRect.width / 2;
    const ty = sRect.top - stageRect.top + sRect.height / 2;

    const col = item.gpus >= 8 ? "#b8532f" : item.gpus >= 4 ? "#2c6e49" : "#b77213";
    PARTICLES.push({
      sx, sy: sy + (idx % 3) * 18,
      tx, ty,
      cx: (sx + tx) * 0.5,
      cy: Math.min(sy, ty) - 32 - (idx % 2) * 14,
      progress: 0,
      speed: 0.048,
      color: col,
      gpus: item.gpus,
    });
  });
}

function drawFlowCanvas() {
  const canvas = document.getElementById("flow-canvas");
  const stage = document.getElementById("sim-stage");
  if (!canvas || !stage) return;

  const w = stage.clientWidth;
  const h = stage.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  for (let i = PARTICLES.length - 1; i >= 0; i--) {
    const p = PARTICLES[i];
    p.progress += p.speed;
    if (p.progress >= 1) {
      RIPPLES.push({ x: p.tx, y: p.ty, r: 4, alpha: 0.85, color: p.color });
      PARTICLES.splice(i, 1);
      continue;
    }

    const t = 1 - Math.pow(1 - p.progress, 3);
    const omt = 1 - t;
    const x = omt * omt * p.sx + 2 * omt * t * p.cx + t * t * p.tx;
    const y = omt * omt * p.sy + 2 * omt * t * p.cy + t * t * p.ty;

    ctx.beginPath();
    ctx.moveTo(p.sx, p.sy);
    ctx.quadraticCurveTo(p.cx, p.cy, x, y);
    ctx.strokeStyle = p.color + "55";
    ctx.lineWidth = 1.8;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y, p.gpus >= 4 ? 5 : 4, 0, Math.PI * 2);
    ctx.fillStyle = p.color;
    ctx.fill();
  }

  for (let i = RIPPLES.length - 1; i >= 0; i--) {
    const r = RIPPLES[i];
    r.r += 1.1;
    r.alpha -= 0.05;
    if (r.alpha <= 0) {
      RIPPLES.splice(i, 1);
      continue;
    }
    ctx.beginPath();
    ctx.arc(r.x, r.y, r.r, 0, Math.PI * 2);
    ctx.strokeStyle = r.color;
    ctx.globalAlpha = r.alpha;
    ctx.lineWidth = 1.8;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
}

// ---------------------------------------------------------------------------
// Queue, Decision Spotlight & Sparkline
// ---------------------------------------------------------------------------
function renderQueueBlocks() {
  const s = SNAPSHOT;
  document.getElementById("q-count-pill").textContent = `${s.metrics.waiting} waiting`;
  const box = document.getElementById("queue-blocks");

  if (s.queue.length === 0) {
    box.innerHTML = `<div class="py-10 text-center text-xs text-[#847d72]">Queue empty — all arrived jobs placed.</div>`;
    return;
  }

  box.innerHTML = s.queue.slice(0, 5).map(j => {
    const gCls = j.gpus >= 8 ? "g8" : j.gpus >= 4 ? "g4" : j.gpus >= 2 ? "g2" : "g1";
    const badge = j.protected
      ? `<span class="tag tag-ochre">PROTECTED</span>`
      : `<span class="mono text-[11px] text-[#5c574e]">${Math.round(j.waited)}s</span>`;

    return `<div class="q-item ${j.protected ? "protected" : ""}">
      <div class="flex items-center gap-2.5">
        <span class="gpu-pill-bar ${gCls} mono">${j.gpus}G</span>
        <div>
          <div class="mono text-xs font-semibold text-[#1d1b18]">Job #${j.job_id}</div>
          <div class="text-[11px] text-[#847d72]">dur ${Math.round(j.duration)}s</div>
        </div>
      </div>
      ${badge}
    </div>`;
  }).join("");
}

function renderDecisionSpotlight() {
  const d = SNAPSHOT?.last_decision;
  const box = document.getElementById("decision-spotlight");
  if (!d) {
    box.innerHTML = `<span class="text-[#847d72]">Waiting for placement...</span>`;
    return;
  }
  const placed = d.placements.find(p => p.job_id === d.focal_job_id);
  box.innerHTML = placed
    ? `<div class="flex items-center justify-between">
        <span class="mono font-semibold text-[#1d1b18]">Job #${d.focal_job_id} (${d.focal_gpus}G)</span>
        <span class="tag tag-sage mono">→ C${placed.cluster_id} · S${placed.server_id}</span>
      </div>`
    : `<div class="flex items-center justify-between">
        <span class="mono font-semibold text-[#1d1b18]">Job #${d.focal_job_id} (${d.focal_gpus}G)</span>
        <span class="tag tag-ochre">Waiting for free node</span>
      </div>`;
}

function renderLiveChart() {
  const tl = SNAPSHOT?.timeline || [];
  const recent = tl.slice(-80);
  const labels = recent.map(p => `${Math.round(p.t)}s`);
  const utils = recent.map(p => p.util);

  if (!CHARTS.live) {
    CHARTS.live = new Chart(document.getElementById("chart-live-telemetry"), {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Util %", data: utils, borderColor: "#b8532f", backgroundColor: "rgba(184,83,47,0.12)", fill: true, tension: 0.25, pointRadius: 0, borderWidth: 2 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#847d72", maxTicksLimit: 5, font: { size: 10 } }, grid: { display: false } },
          y: { min: 0, max: 100, ticks: { color: "#847d72", font: { size: 10 } }, grid: { color: "rgba(29,27,24,0.06)" } },
        },
      },
    });
  } else {
    CHARTS.live.data.labels = labels;
    CHARTS.live.data.datasets[0].data = utils;
    CHARTS.live.update("none");
  }
}

// ---------------------------------------------------------------------------
// TAB 2: Compare Policies
// ---------------------------------------------------------------------------
async function runComparisonArena() {
  const selected = Array.from(document.querySelectorAll("#cmp-sched-chips .policy-chip.selected"))
    .map(b => b.dataset.sched);
  if (!selected.length) return;

  const btn = document.getElementById("btn-run-cmp");
  btn.disabled = true;
  btn.textContent = `Running ${selected.length}...`;
  try {
    const data = await apiPost("/api/compare", {
      trace_id: document.getElementById("cmp-trace").value,
      topo_id: document.getElementById("cmp-topo").value,
      sched_ids: selected,
      n_jobs: parseInt(document.getElementById("cmp-njobs").value || "400", 10),
      eta: parseFloat(document.getElementById("cmp-eta").value || "10"),
      burst: document.getElementById("cmp-burst").checked,
      seed: 0,
    });
    LAST_COMPARE_DATA = data;
    renderComparison(data);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Comparison";
  }
}

function renderComparison(data) {
  document.getElementById("cmp-placeholder").classList.add("hidden");
  document.getElementById("cmp-results-wrap").classList.remove("hidden");
  const meta = data.meta;
  document.getElementById("cmp-meta-badge").textContent = `${meta.trace_id} · ${meta.topo_id} · ${meta.n_jobs} jobs · η=${meta.eta}`;

  const rows = data.results;
  const bestW = Math.min(...rows.map(r => r.avg_wait));
  const bestLB = Math.min(...rows.map(r => r.avg_lb_index));

  document.getElementById("cmp-table-wrap").innerHTML = `<table class="tbl">
    <thead><tr><th>Policy</th><th>Avg Wait (s)</th><th>P99 Wait (s)</th><th>8-GPU Wait</th><th>Starved (&gt;3×)</th><th>GPU Util</th><th>LB σ²</th><th>Done</th></tr></thead>
    <tbody>
      ${rows.map(r => `<tr>
        <td class="mono font-bold text-[#1d1b18]">${r.id}</td>
        <td class="mono ${r.avg_wait === bestW ? "text-[#2c6e49] font-bold" : ""}">${r.avg_wait.toLocaleString()}</td>
        <td class="mono">${r.p99_wait.toLocaleString()}</td>
        <td class="mono">${r.avg_wait_8gpu.toLocaleString()}</td>
        <td class="mono">${r.starved_3x}</td>
        <td class="mono">${r.gpu_utilization.toFixed(2)}%</td>
        <td class="mono ${r.avg_lb_index === bestLB ? "text-[#b8532f] font-bold" : ""}">${r.avg_lb_index.toFixed(5)}</td>
        <td class="mono">${r.jobs_completed}</td>
      </tr>`).join("")}
    </tbody>
  </table>`;

  ["cmpWait", "cmpBySize"].forEach(k => { if (CHARTS[k]) { CHARTS[k].destroy(); CHARTS[k] = null; } });
  const opts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#5c574e", font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: "#5c574e", font: { size: 11 } }, grid: { display: false } },
      y: { ticks: { color: "#5c574e", font: { size: 11 } }, grid: { color: "rgba(29,27,24,0.06)" } },
    },
  };

  CHARTS.cmpWait = new Chart(document.getElementById("cmp-chart-wait"), {
    type: "bar",
    data: {
      labels: rows.map(r => r.id),
      datasets: [
        { label: "Avg Wait (s)", data: rows.map(r => r.avg_wait), backgroundColor: "#b8532f", borderRadius: 4 },
        { label: "P99 Wait (s)", data: rows.map(r => r.p99_wait), backgroundColor: "#475569", borderRadius: 4 },
      ],
    },
    options: opts,
  });

  CHARTS.cmpBySize = new Chart(document.getElementById("cmp-chart-bysize"), {
    type: "bar",
    data: {
      labels: rows.map(r => r.id),
      datasets: ["1", "2", "4", "8"].map((sz, i) => ({
        label: `${sz}-GPU`,
        data: rows.map(r => r.avg_wait_by_gpus[sz] || 0),
        backgroundColor: CHART_PALETTE[i],
        borderRadius: 4,
      })),
    },
    options: opts,
  });
}

function exportComparisonCSV() {
  if (!LAST_COMPARE_DATA) return;
  const cols = ["id", "avg_wait", "p99_wait", "avg_wait_8gpu", "starved_3x", "gpu_utilization", "avg_lb_index", "jobs_completed"];
  const lines = [cols.join(","), ...LAST_COMPARE_DATA.results.map(r => cols.map(c => r[c]).join(","))];
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
  a.download = `benchmark_${LAST_COMPARE_DATA.meta.trace_id}.csv`;
  a.click();
}

// ---------------------------------------------------------------------------
// TAB 3: Fairness & Mixed-GPU
// ---------------------------------------------------------------------------
async function runN2Inspector() {
  const data = await apiPost("/api/experiments/inspect_n2", {
    demand_weights: {
      "1": parseFloat(document.getElementById("n2-w1").value || "0"),
      "2": parseFloat(document.getElementById("n2-w2").value || "0"),
      "4": parseFloat(document.getElementById("n2-w4").value || "0"),
      "8": parseFloat(document.getElementById("n2-w8").value || "0"),
    },
    job_gpus: parseInt(document.getElementById("n2-jobgpus").value || "2", 10),
  });

  const eg = data.empty_gradients;
  document.getElementById("n2-gradient-cards").innerHTML = `
    <div class="bg-[#fffdf9] border border-[#d5cfc2] rounded-lg p-3 text-xs mono space-y-1">
      <div class="font-bold text-[#1d1b18] font-sans mb-1">4-GPU Server Gradient</div>
      <div class="flex justify-between"><span class="text-[#5c574e]">Original G:</span><span class="text-[#b8532f] font-semibold">${eg.server_4g.G_orig}</span></div>
      <div class="flex justify-between"><span class="text-[#5c574e]">F_C (Truncated):</span><span class="text-[#b77213] font-semibold">${eg.server_4g.G_fc}</span></div>
      <div class="flex justify-between"><span class="text-[#5c574e]">F_C (Renormed):</span><span class="text-[#2c6e49] font-semibold">${eg.server_4g.G_fc_norm}</span></div>
    </div>
    <div class="bg-[#fffdf9] border border-[#d5cfc2] rounded-lg p-3 text-xs mono space-y-1">
      <div class="font-bold text-[#1d1b18] font-sans mb-1">8-GPU Server Gradient</div>
      <div class="flex justify-between"><span class="text-[#5c574e]">Original G:</span><span class="text-[#b8532f] font-semibold">${eg.server_8g.G_orig}</span></div>
      <div class="flex justify-between"><span class="text-[#5c574e]">F_C (Truncated):</span><span class="text-[#b77213] font-semibold">${eg.server_8g.G_fc}</span></div>
      <div class="flex justify-between"><span class="text-[#5c574e]">F_C (Renormed):</span><span class="text-[#2c6e49] font-semibold">${eg.server_8g.G_fc_norm}</span></div>
    </div>`;

  if (CHARTS.n2) CHARTS.n2.destroy();
  const t4 = data.table_4g;
  CHARTS.n2 = new Chart(document.getElementById("n2-chart"), {
    type: "line",
    data: {
      labels: t4.map(r => `${r.free} free`),
      datasets: [
        { label: "Original F(r)", data: t4.map(r => r.F_orig), borderColor: "#b8532f", borderWidth: 2, tension: 0.2 },
        { label: "F_C(r) [4G]", data: t4.map(r => r.F_C), borderColor: "#b77213", borderWidth: 2, tension: 0.2 },
        { label: "F_C Norm(r) [4G]", data: t4.map(r => r.F_C_norm), borderColor: "#2c6e49", borderWidth: 2, tension: 0.2 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#5c574e", font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: "#5c574e", font: { size: 11 } }, grid: { display: false } },
        y: { min: 0, ticks: { color: "#5c574e", font: { size: 11 } }, grid: { color: "rgba(29,27,24,0.06)" } },
      },
    },
  });
}

async function loadSavedResearch() {
  SAVED_RESEARCH = await apiGet("/api/experiments/saved");
  showSavedStudy("e5");
}

function showSavedStudy(kind) {
  if (!SAVED_RESEARCH) return;
  if (CHARTS.saved) { CHARTS.saved.destroy(); CHARTS.saved = null; }
  const chartEl = document.getElementById("saved-study-chart");
  const tblEl = document.getElementById("saved-study-table");

  const opts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#5c574e", font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: "#5c574e", font: { size: 10 } }, grid: { display: false } },
      y: { ticks: { color: "#5c574e", font: { size: 10 } }, grid: { color: "rgba(29,27,24,0.06)" } },
    },
  };

  if (kind === "e5" && SAVED_RESEARCH.e5) {
    const homo = SAVED_RESEARCH.e5.venus_eta15_homo || {};
    const mixed = SAVED_RESEARCH.e5.venus_eta15_mixed || {};
    const fc = SAVED_RESEARCH.e5.venus_eta15_mixed_fc || {};
    const items = [
      { label: "mcg (homo)", val: homo.mcg?.avg_wait || 0, p99: homo.mcg?.p99_wait || 0 },
      { label: "mcg (mixed)", val: mixed.mcg?.avg_wait || 0, p99: mixed.mcg?.p99_wait || 0 },
      { label: "mcg_fc (mixed)", val: fc.mcg_fc?.avg_wait || 0, p99: fc.mcg_fc?.p99_wait || 0 },
      { label: "mcg_fc_norm", val: fc.mcg_fc_norm?.avg_wait || 0, p99: fc.mcg_fc_norm?.p99_wait || 0 },
      { label: "fcfs (mixed)", val: mixed.fcfs?.avg_wait || 0, p99: mixed.fcfs?.p99_wait || 0 },
    ];
    CHARTS.saved = new Chart(chartEl, {
      type: "bar",
      data: {
        labels: items.map(x => x.label),
        datasets: [{ label: "Avg Wait (s) — Venus η=15", data: items.map(x => x.val), backgroundColor: ["#2c6e49", "#475569", "#b77213", "#b8532f", "#991b1b"], borderRadius: 4 }],
      },
      options: opts,
    });
    tblEl.innerHTML = `<table class="tbl"><thead><tr><th>Configuration</th><th>Avg Wait</th><th>P99 Wait</th></tr></thead><tbody>
      ${items.map(x => `<tr><td class="mono font-semibold text-[#1d1b18]">${x.label}</td><td class="mono">${Math.round(x.val).toLocaleString()}s</td><td class="mono">${Math.round(x.p99).toLocaleString()}s</td></tr>`).join("")}
    </tbody></table>`;
  } else if (kind === "e4" && SAVED_RESEARCH.e4) {
    const rows = SAVED_RESEARCH.e4.filter(r => r.trace === "venus" && r.eta === 15).slice(0, 7);
    CHARTS.saved = new Chart(chartEl, {
      type: "bar",
      data: {
        labels: rows.map(r => r.label),
        datasets: [
          { label: "Avg Wait", data: rows.map(r => r.avg_wait), backgroundColor: "#2c6e49", borderRadius: 4 },
          { label: "P99 Wait", data: rows.map(r => r.p99_wait), backgroundColor: "#b8532f", borderRadius: 4 },
        ],
      },
      options: opts,
    });
    tblEl.innerHTML = `<table class="tbl"><thead><tr><th>Policy</th><th>Avg Wait</th><th>P99 Wait</th><th>Starved (&gt;3×)</th></tr></thead><tbody>
      ${rows.map(r => `<tr><td class="mono font-semibold text-[#1d1b18]">${r.label}</td><td class="mono">${Math.round(r.avg_wait).toLocaleString()}s</td><td class="mono">${Math.round(r.p99_wait).toLocaleString()}s</td><td class="mono">${r.starved_3x}</td></tr>`).join("")}
    </tbody></table>`;
  } else if (kind === "e7" && SAVED_RESEARCH.e7_norm) {
    const normRows = SAVED_RESEARCH.e7_norm.filter(r => r.kwargs?.window === "dynamic");
    const raw = normRows.filter(r => r.scheduler === "mcg" || (r.label || "").startsWith("raw"));
    const norm = normRows.filter(r => r.scheduler === "mcg_norm" || (r.label || "").startsWith("norm"));
    CHARTS.saved = new Chart(chartEl, {
      type: "line",
      data: {
        labels: raw.map(r => `w=${r.kwargs?.w}`),
        datasets: [
          { label: "Raw Score", data: raw.map(r => r.avg_wait), borderColor: "#b8532f", borderWidth: 2 },
          { label: "Normalised Score", data: norm.map(r => r.avg_wait), borderColor: "#2c6e49", borderWidth: 2 },
        ],
      },
      options: opts,
    });
    tblEl.innerHTML = `<table class="tbl"><thead><tr><th>w</th><th>Raw Avg Wait</th><th>Normalised Avg Wait</th></tr></thead><tbody>
      ${raw.map((r, i) => `<tr><td class="mono">${r.kwargs?.w}</td><td class="mono">${Math.round(r.avg_wait).toLocaleString()}s</td><td class="mono">${Math.round(norm[i]?.avg_wait || 0).toLocaleString()}s</td></tr>`).join("")}
    </tbody></table>`;
  }
}
