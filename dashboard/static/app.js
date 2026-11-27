/* MCG-Sched Live Animated Simulator Frontend */

const SESSION_ID = "sim_" + Math.random().toString(36).slice(2, 10);

let META = { traces: [], topologies: [], schedulers: [] };
let SNAPSHOT = null;
let PLAY_TIMER = null;
let IS_PLAYING = false;
let LAST_COMPARE_DATA = null;
let SAVED_RESEARCH = null;
let CHARTS = {};

const PALETTE = ["#3b82f6", "#14b8a6", "#f59e0b", "#8b5cf6", "#ef4444", "#06b6d4"];

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  lucide.createIcons();
  await loadMetadata();
  await onScenarioChange(false);
  // Start with a burst step so the cluster immediately shows live activity
  await stepSimulation(1);
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

  const defCompare = new Set(["fcfs", "consolidate", "fgd", "mcg", "mcg_fair", "mcg_fc"]);
  document.getElementById("cmp-sched-chips").innerHTML = META.schedulers
    .map(s => `<button type="button" class="policy-chip ${defCompare.has(s.id) ? "selected" : ""}" data-sched="${s.id}" onclick="this.classList.toggle('selected')">${s.id}</button>`)
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
    k: { label: "Guard k", val: 3.0, step: 0.5, min: 0.2, max: 10 },
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
// Tabs & Drawers
// ---------------------------------------------------------------------------
function switchTab(tabId) {
  document.querySelectorAll(".top-tab").forEach(b => b.classList.toggle("active", b.dataset.tab === tabId));
  document.querySelectorAll(".tab-view").forEach(v => v.classList.toggle("hidden", v.id !== `tab-${tabId}`));
  if (tabId === "research" && SAVED_RESEARCH) showSavedStudy("e5");
}

function toggleAdvancedDrawer() {
  document.getElementById("advanced-drawer").classList.toggle("hidden");
}

function toggleScoreDrawer() {
  document.getElementById("score-drawer").classList.toggle("hidden");
}

// ---------------------------------------------------------------------------
// Scenarios & Simulation Control
// ---------------------------------------------------------------------------
async function onScenarioChange(autoStep = true) {
  stopPlay();
  const sc = document.getElementById("live-scenario").value;
  if (sc === "venus_std") {
    document.getElementById("live-trace").value = "venus";
    document.getElementById("live-njobs").value = 600;
    document.getElementById("live-eta").value = 15;
    document.getElementById("live-burst").checked = false;
    document.getElementById("live-topo").value = "venus15";
    document.getElementById("live-sched").value = "mcg";
  } else if (sc === "burst_800") {
    document.getElementById("live-trace").value = "synthetic";
    document.getElementById("live-njobs").value = 800;
    document.getElementById("live-eta").value = 10;
    document.getElementById("live-burst").checked = true;
    document.getElementById("live-topo").value = "venus15";
    document.getElementById("live-sched").value = "mcg";
  } else if (sc === "fairness_n1") {
    document.getElementById("live-trace").value = "venus";
    document.getElementById("live-njobs").value = 1000;
    document.getElementById("live-eta").value = 15;
    document.getElementById("live-burst").checked = false;
    document.getElementById("live-topo").value = "venus15";
    document.getElementById("live-sched").value = "mcg_fair";
  } else if (sc === "mixed_n2") {
    document.getElementById("live-trace").value = "venus";
    document.getElementById("live-njobs").value = 800;
    document.getElementById("live-eta").value = 15;
    document.getElementById("live-burst").checked = false;
    document.getElementById("live-topo").value = "venus15_mixed";
    document.getElementById("live-sched").value = "mcg_fc";
  } else if (sc === "philly_std") {
    document.getElementById("live-trace").value = "philly";
    document.getElementById("live-njobs").value = 600;
    document.getElementById("live-eta").value = 4;
    document.getElementById("live-burst").checked = false;
    document.getElementById("live-topo").value = "paper_8x16";
    document.getElementById("live-sched").value = "mcg";
  }
  renderPolicyParams();
  await initSimulation();
  if (autoStep) {
    await stepSimulation(sc === "burst_800" ? 1 : 25);
  }
}

async function onPolicyChange() {
  renderPolicyParams();
  await initSimulation();
  await stepSimulation(10);
}

function setStatus(state, text) {
  const dot = document.getElementById("status-dot");
  dot.className = "w-2 h-2 rounded-full " + (
    state === "running" ? "bg-blue-400 animate-ping" :
    state === "done" ? "bg-teal-400" : "bg-amber-400"
  );
  document.getElementById("status-text").textContent = text;
}

async function initSimulation() {
  stopPlay();
  SNAPSHOT = await apiPost("/api/sim/init", {
    session_id: SESSION_ID,
    trace_id: document.getElementById("live-trace").value,
    topo_id: document.getElementById("live-topo").value,
    sched_id: document.getElementById("live-sched").value,
    n_jobs: parseInt(document.getElementById("live-njobs").value || "400", 10),
    eta: parseFloat(document.getElementById("live-eta").value || "10"),
    burst: document.getElementById("live-burst").checked,
    seed: 0,
    params: readParams(),
  });
  setStatus("idle", "Ready");
  renderSnapshot();
}

async function stepSimulation(count = 1) {
  if (!SNAPSHOT || SNAPSHOT.done) {
    stopPlay();
    return;
  }
  SNAPSHOT = await apiPost("/api/sim/step", { session_id: SESSION_ID, count });
  renderSnapshot();
  if (SNAPSHOT.done) {
    stopPlay();
    setStatus("done", "Complete");
  }
}

async function runToEnd() {
  stopPlay();
  if (!SNAPSHOT) return;
  setStatus("running", "Completing...");
  document.getElementById("btn-run-end").disabled = true;
  try {
    SNAPSHOT = await apiPost("/api/sim/run", { session_id: SESSION_ID });
    renderSnapshot();
    setStatus("done", "Complete");
  } finally {
    document.getElementById("btn-run-end").disabled = SNAPSHOT?.done ?? false;
  }
}

function togglePlay() {
  if (IS_PLAYING) stopPlay();
  else startPlay();
}

function startPlay() {
  if (!SNAPSHOT || SNAPSHOT.done) return;
  IS_PLAYING = true;
  const btn = document.getElementById("btn-play");
  btn.innerHTML = `<i data-lucide="pause" class="w-3.5 h-3.5"></i> Pause`;
  btn.classList.remove("btn-blue");
  btn.classList.add("btn-amber");
  lucide.createIcons();
  setStatus("running", "Simulating");

  const tick = async () => {
    if (!IS_PLAYING || !SNAPSHOT || SNAPSHOT.done) {
      stopPlay();
      return;
    }
    const delay = parseInt(document.getElementById("play-speed").value, 10);
    const stepSize = delay <= 80 ? 4 : 1;
    await stepSimulation(stepSize);
    if (IS_PLAYING && !SNAPSHOT.done) {
      PLAY_TIMER = setTimeout(tick, delay);
    }
  };
  tick();
}

function stopPlay() {
  IS_PLAYING = false;
  if (PLAY_TIMER) {
    clearTimeout(PLAY_TIMER);
    PLAY_TIMER = null;
  }
  const btn = document.getElementById("btn-play");
  if (btn) {
    btn.innerHTML = `<i data-lucide="play" class="w-3.5 h-3.5"></i> Play`;
    btn.classList.add("btn-blue");
    btn.classList.remove("btn-amber");
    lucide.createIcons();
  }
  if (SNAPSHOT && !SNAPSHOT.done) setStatus("idle", "Paused");
}

// ---------------------------------------------------------------------------
// Render Live Stage + Animations
// ---------------------------------------------------------------------------
function renderSnapshot() {
  const s = SNAPSHOT;
  if (!s) return;
  const m = s.metrics;

  document.getElementById("header-clock").textContent = `t = ${s.now.toLocaleString()} s`;
  const pct = s.total_jobs > 0 ? Math.round((m.completed / s.total_jobs) * 100) : 0;

  document.getElementById("kpi-pct").textContent = `${pct}%`;
  document.getElementById("kpi-done").textContent = m.completed.toLocaleString();
  document.getElementById("kpi-total").textContent = `/ ${s.total_jobs.toLocaleString()} jobs`;
  document.getElementById("sim-progress-bar").style.width = `${pct}%`;

  document.getElementById("kpi-util").textContent = `${m.utilization_pct.toFixed(1)}%`;
  document.getElementById("kpi-gpus-sub").textContent = `${m.used_gpus} / ${m.total_gpus} GPUs`;
  document.getElementById("kpi-stranded").textContent = `· ${m.stranded_gpus} stranded`;
  document.getElementById("util-progress-bar").style.width = `${m.utilization_pct}%`;

  document.getElementById("kpi-wait-cnt").textContent = `${m.waiting} waiting`;
  document.getElementById("kpi-avg-wait").textContent = fmtShortTime(s.summary ? s.summary.avg_wait : m.avg_wait);
  document.getElementById("kpi-p99-wait").textContent = `P99: ${fmtShortTime(s.summary ? s.summary.p99_wait : m.p99_wait)}`;
  document.getElementById("kpi-running-sub").textContent = `${m.running} running jobs`;

  document.getElementById("kpi-res-badge").textContent = `${m.reserved_count} reserved`;
  document.getElementById("kpi-lb").textContent = (s.summary ? s.summary.avg_lb_index : m.lb_index).toFixed(5);
  document.getElementById("kpi-fair-sub").textContent = `${m.protected_count} protected · ${m.starved_completed + m.starved_waiting} starved`;

  document.getElementById("btn-step1").disabled = s.done;
  document.getElementById("btn-step10").disabled = s.done;
  document.getElementById("btn-play").disabled = s.done;
  document.getElementById("btn-run-end").disabled = s.done;

  renderQueueBlocks();
  renderDecisionSpotlight();
  renderClusterRack();
  renderLiveChart();
  renderEventTicker();
  requestAnimationFrame(drawDispatchBeams);
}

function renderQueueBlocks() {
  const s = SNAPSHOT;
  document.getElementById("q-count-pill").textContent = s.metrics.waiting;
  const focalId = s.last_decision?.focal_job_id;
  const box = document.getElementById("queue-blocks");

  if (s.queue.length === 0) {
    box.innerHTML = `<div class="py-12 text-center text-xs text-slate-500">Queue empty — all arrived jobs placed.</div>`;
    return;
  }

  box.innerHTML = s.queue.slice(0, 14).map(j => {
    const cls = j.protected ? "protected" : (j.job_id === focalId ? "focal" : "");
    const gCls = j.gpus >= 8 ? "g8" : j.gpus >= 4 ? "g4" : j.gpus >= 2 ? "g2" : "g1";
    let pips = "";
    for (let i = 0; i < j.gpus; i++) pips += `<span class="gpu-pip ${gCls}"></span>`;

    const badge = j.protected
      ? `<span class="tag tag-amber">PROTECTED</span>`
      : j.starved
      ? `<span class="tag tag-red">STARVED</span>`
      : `<span class="mono text-[11px] text-slate-400">${j.waited}s</span>`;

    return `<div class="q-block ${cls}" id="qjob-${j.job_id}">
      <div class="flex items-center gap-2.5">
        <div class="gpu-bar-mini">${pips}</div>
        <div>
          <div class="mono text-xs font-semibold text-white">Job #${j.job_id}</div>
          <div class="text-[10.5px] text-slate-400">${j.gpus} GPU${j.gpus > 1 ? "s" : ""} · dur ${Math.round(j.duration)}s</div>
        </div>
      </div>
      <div>${badge}</div>
    </div>`;
  }).join("");
}

function renderDecisionSpotlight() {
  const d = SNAPSHOT?.last_decision;
  const box = document.getElementById("decision-spotlight");
  const drawer = document.getElementById("score-drawer");

  if (!d) {
    box.innerHTML = `<div class="text-slate-500">No scheduling round evaluated yet.</div>`;
    drawer.innerHTML = "";
    return;
  }

  const placed = d.placements.find(p => p.job_id === d.focal_job_id);
  const chosenCand = d.candidates.find(c => c.selected);

  box.innerHTML = `
    <div class="flex items-center justify-between">
      <span class="mono font-semibold text-white">Job #${d.focal_job_id} (${d.focal_gpus}G)</span>
      ${placed
        ? `<span class="tag tag-teal">→ Cluster ${placed.cluster_id} · Server ${placed.server_id}</span>`
        : `<span class="tag tag-amber">Waiting for capacity</span>`}
    </div>
    ${chosenCand ? `
      <div class="grid grid-cols-3 gap-2 pt-1 text-center bg-[#0b0d11] p-2 rounded-md border border-[#252a36]">
        <div>
          <div class="text-[10px] text-slate-500 uppercase">Frag Grad G</div>
          <div class="mono text-xs text-blue-400 font-semibold">${chosenCand.g_orig.toFixed(4)}</div>
        </div>
        <div>
          <div class="text-[10px] text-slate-500 uppercase">Variance σ²</div>
          <div class="mono text-xs text-teal-400 font-semibold">${chosenCand.sigma2.toFixed(5)}</div>
        </div>
        <div>
          <div class="text-[10px] text-slate-500 uppercase">Score v</div>
          <div class="mono text-xs text-white font-semibold">${chosenCand.score.toFixed(4)}</div>
        </div>
      </div>` : ""}`;

  if (d.candidates.length > 0) {
    drawer.innerHTML = `<table class="tbl">
      <thead><tr><th>Node</th><th>Cap</th><th>Free</th><th>G</th><th>G_C</th><th>σ²</th><th>Score</th></tr></thead>
      <tbody>
        ${d.candidates.slice(0, 12).map(c => `<tr class="${c.selected ? "win" : ""}">
          <td class="mono font-medium">C${c.cluster_id}·S${c.server_id}</td>
          <td class="mono">${c.capacity}G</td>
          <td class="mono">${c.free}</td>
          <td class="mono text-blue-400">${c.g_orig.toFixed(3)}</td>
          <td class="mono text-slate-300">${c.g_fc.toFixed(3)}</td>
          <td class="mono text-teal-400">${c.sigma2.toFixed(4)}</td>
          <td class="mono font-semibold">${c.score.toFixed(3)}</td>
        </tr>`).join("")}
      </tbody>
    </table>`;
  }
}

function renderClusterRack() {
  const s = SNAPSHOT;
  if (!s) return;

  const startedKeys = new Set(
    (s.step_started || []).filter(x => x.placement).map(x => `${x.placement[0]}_${x.placement[1]}`)
  );
  const finishedKeys = new Set(
    (s.step_finished || []).filter(x => x.placement).map(x => `${x.placement[0]}_${x.placement[1]}`)
  );

  let cap4 = 0, cap8 = 0;
  s.clusters.forEach(c => c.servers.forEach(sv => { if (sv.capacity <= 4) cap4++; else cap8++; }));
  document.getElementById("topo-mix-tag").textContent =
    cap4 > 0 ? `${cap4}×4G + ${cap8}×8G Mixed` : `${cap8}×8G Homogeneous`;

  document.getElementById("cluster-rack").innerHTML = s.clusters.map(c => {
    const uPct = Math.round(c.utilization * 100);
    const nodesHtml = c.servers.map(sv => {
      const key = `${c.cluster_id}_${sv.server_id}`;
      const isRes = sv.reserved_for !== null;
      const pulse = startedKeys.has(key) ? "placed-pulse" : (finishedKeys.has(key) ? "freed-flash" : "");
      const capCls = sv.capacity <= 4 ? "cap4" : "";
      const resCls = isRes ? "reserved" : "";

      let slots = "";
      for (let i = 0; i < sv.used; i++) slots += `<span class="nslot used"></span>`;
      for (let i = 0; i < sv.free; i++) slots += `<span class="nslot ${isRes ? "res-free" : "free"}"></span>`;

      const tip = `C${c.cluster_id}·S${sv.server_id} (${sv.capacity}G) | ${sv.used}/${sv.capacity} used` +
        (sv.running.length ? " | " + sv.running.map(r => `#${r.job_id}(${r.gpus}G, ${r.remaining}s left)`).join(", ") : "");

      return `<div id="node-${key}" class="node-box ${capCls} ${resCls} ${pulse}" title="${tip}">
        <div class="flex items-center justify-between text-[10px]">
          <span class="mono text-slate-400">S${sv.server_id} <span class="${sv.capacity <= 4 ? "text-purple-300" : "text-slate-500"}">${sv.capacity}G</span></span>
          ${isRes
            ? `<span class="mono font-bold text-amber-400">LOCK#${sv.reserved_for}</span>`
            : `<span class="mono text-slate-300">${sv.used}/${sv.capacity}</span>`}
        </div>
        <div class="node-slots">${slots}</div>
        ${sv.used > 0 ? `<div class="node-exec-bar" style="width:${sv.max_progress_pct || 8}%"></div>` : ""}
      </div>`;
    }).join("");

    return `<div class="rack-cluster">
      <div class="flex items-center justify-between mb-1.5">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-slate-200">Cluster ${c.cluster_id}</span>
          <span class="mono text-[11px] text-slate-500">${c.used}/${c.capacity} GPUs</span>
        </div>
        <div class="flex items-center gap-2 w-32">
          <div class="flex-1 h-1.5 bg-[#12151c] rounded-full overflow-hidden">
            <div class="h-full bg-blue-500 transition-all duration-200" style="width:${uPct}%"></div>
          </div>
          <span class="mono text-[11px] text-slate-300 w-8 text-right">${uPct}%</span>
        </div>
      </div>
      <div class="flex flex-wrap gap-1.5">${nodesHtml}</div>
    </div>`;
  }).join("");
}

function drawDispatchBeams() {
  const svg = document.getElementById("dispatch-svg");
  const stage = document.getElementById("sim-stage");
  if (!svg || !stage || !SNAPSHOT) return;

  const started = (SNAPSHOT.step_started || []).filter(x => x.placement).slice(0, 6);
  if (started.length === 0) {
    svg.innerHTML = "";
    return;
  }

  const stageRect = stage.getBoundingClientRect();
  const qBox = document.getElementById("queue-blocks");
  const qRect = qBox.getBoundingClientRect();
  const startX = qRect.right - stageRect.left - 8;
  const startY = qRect.top - stageRect.top + 36;

  let paths = "";
  started.forEach(item => {
    const [cid, sid] = item.placement;
    const node = document.getElementById(`node-${cid}_${sid}`);
    if (!node) return;
    const nRect = node.getBoundingClientRect();
    // Skip beams for scrolled-out-of-view nodes
    if (nRect.top < stageRect.top || nRect.bottom > stageRect.bottom + 20) return;

    const endX = nRect.left - stageRect.left + 8;
    const endY = nRect.top - stageRect.top + nRect.height / 2;
    const midX = (startX + endX) / 2;
    paths += `<path class="beam-path" d="M ${startX} ${startY} C ${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}" />`;
    paths += `<circle cx="${endX}" cy="${endY}" r="3.5" fill="#60a5fa" />`;
  });
  svg.innerHTML = paths;
}

function renderLiveChart() {
  const tl = SNAPSHOT?.timeline || [];
  const labels = tl.map(p => `${Math.round(p.t)}s`);
  const utils = tl.map(p => p.util);
  const qlens = tl.map(p => p.queue_len);

  if (!CHARTS.live) {
    CHARTS.live = new Chart(document.getElementById("chart-live-telemetry"), {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "GPU Util (%)", data: utils, borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,0.1)", fill: true, tension: 0.25, pointRadius: 0, borderWidth: 2, yAxisID: "y" },
          { label: "Queue Length", data: qlens, borderColor: "#f59e0b", tension: 0.25, pointRadius: 0, borderWidth: 1.8, yAxisID: "y1" },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { labels: { color: "#9aa4b5", boxWidth: 10, font: { size: 11 } } } },
        scales: {
          x: { ticks: { color: "#667085", maxTicksLimit: 7, font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.04)" } },
          y: { position: "left", min: 0, max: 100, ticks: { color: "#60a5fa", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.05)" } },
          y1: { position: "right", min: 0, ticks: { color: "#fbbf24", font: { size: 10 } }, grid: { drawOnChartArea: false } },
        },
      },
    });
  } else {
    CHARTS.live.data.labels = labels;
    CHARTS.live.data.datasets[0].data = utils;
    CHARTS.live.data.datasets[1].data = qlens;
    CHARTS.live.update("none");
  }
}

function renderEventTicker() {
  const evs = SNAPSHOT?.recent_events || [];
  const n1 = SNAPSHOT?.n1_log || [];
  document.getElementById("n1-event-count").textContent = n1.length ? `${n1.length} N1 actions` : "";

  const box = document.getElementById("event-ticker");
  if (evs.length === 0) {
    box.innerHTML = `<div class="text-slate-500 py-6 text-center">Waiting for events...</div>`;
    return;
  }
  box.innerHTML = evs.slice(0, 12).map(ev => {
    const cls = ev.event === "start" ? "tag-blue" : ev.event === "finish" ? "tag-teal" : ev.event === "reject" ? "tag-red" : "tag-dim";
    const loc = ev.placement ? `→ C${ev.placement[0]}·S${ev.placement[1]}` : (ev.wait !== null ? `waited ${ev.wait}s` : "");
    return `<div class="flex items-center justify-between py-1 border-b border-white/5">
      <div class="flex items-center gap-2">
        <span class="tag ${cls}">${ev.event.toUpperCase()}</span>
        <span class="mono text-white">Job #${ev.job_id}</span>
        <span class="mono text-slate-400">${ev.gpus}G</span>
      </div>
      <div class="mono text-[11px] text-slate-400">${loc} @ ${ev.time}s</div>
    </div>`;
  }).join("");
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
  btn.textContent = `Running ${selected.length} policies...`;
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
        <td class="mono font-semibold text-white">${r.id}</td>
        <td class="mono ${r.avg_wait === bestW ? "text-teal-400 font-bold" : ""}">${r.avg_wait.toLocaleString()}</td>
        <td class="mono">${r.p99_wait.toLocaleString()}</td>
        <td class="mono">${r.avg_wait_8gpu.toLocaleString()}</td>
        <td class="mono">${r.starved_3x}</td>
        <td class="mono">${r.gpu_utilization.toFixed(2)}%</td>
        <td class="mono ${r.avg_lb_index === bestLB ? "text-blue-400 font-bold" : ""}">${r.avg_lb_index.toFixed(5)}</td>
        <td class="mono">${r.jobs_completed}</td>
      </tr>`).join("")}
    </tbody>
  </table>`;

  ["cmpWait", "cmpBySize"].forEach(k => { if (CHARTS[k]) { CHARTS[k].destroy(); CHARTS[k] = null; } });
  const opts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#9aa4b5", font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: "#9aa4b5", font: { size: 11 } }, grid: { color: "rgba(255,255,255,0.04)" } },
      y: { ticks: { color: "#9aa4b5", font: { size: 11 } }, grid: { color: "rgba(255,255,255,0.06)" } },
    },
  };

  CHARTS.cmpWait = new Chart(document.getElementById("cmp-chart-wait"), {
    type: "bar",
    data: {
      labels: rows.map(r => r.id),
      datasets: [
        { label: "Avg Wait (s)", data: rows.map(r => r.avg_wait), backgroundColor: "#3b82f6", borderRadius: 4 },
        { label: "P99 Wait (s)", data: rows.map(r => r.p99_wait), backgroundColor: "#8b5cf6", borderRadius: 4 },
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
        backgroundColor: PALETTE[i],
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
// TAB 3: Fairness & Heterogeneity (N1 & N2)
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
    <div class="bg-[#0b0d11] border border-[#252a36] rounded-lg p-3 text-xs mono space-y-1">
      <div class="font-semibold text-slate-200 font-sans mb-1">4-GPU Server Gradient</div>
      <div class="flex justify-between"><span class="text-slate-400">Original G:</span><span class="text-blue-400">${eg.server_4g.G_orig}</span></div>
      <div class="flex justify-between"><span class="text-slate-400">F_C (Truncated):</span><span class="text-amber-400">${eg.server_4g.G_fc}</span></div>
      <div class="flex justify-between"><span class="text-slate-400">F_C (Renormed):</span><span class="text-purple-400">${eg.server_4g.G_fc_norm}</span></div>
    </div>
    <div class="bg-[#0b0d11] border border-[#252a36] rounded-lg p-3 text-xs mono space-y-1">
      <div class="font-semibold text-slate-200 font-sans mb-1">8-GPU Server Gradient</div>
      <div class="flex justify-between"><span class="text-slate-400">Original G:</span><span class="text-blue-400">${eg.server_8g.G_orig}</span></div>
      <div class="flex justify-between"><span class="text-slate-400">F_C (Truncated):</span><span class="text-amber-400">${eg.server_8g.G_fc}</span></div>
      <div class="flex justify-between"><span class="text-slate-400">F_C (Renormed):</span><span class="text-purple-400">${eg.server_8g.G_fc_norm}</span></div>
    </div>`;

  if (CHARTS.n2) CHARTS.n2.destroy();
  const t4 = data.table_4g;
  CHARTS.n2 = new Chart(document.getElementById("n2-chart"), {
    type: "line",
    data: {
      labels: t4.map(r => `${r.free} free`),
      datasets: [
        { label: "Original F(r)", data: t4.map(r => r.F_orig), borderColor: "#3b82f6", borderWidth: 2, tension: 0.2 },
        { label: "F_C(r) [4G]", data: t4.map(r => r.F_C), borderColor: "#f59e0b", borderWidth: 2, tension: 0.2 },
        { label: "F_C Norm(r) [4G]", data: t4.map(r => r.F_C_norm), borderColor: "#8b5cf6", borderWidth: 2, tension: 0.2 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#9aa4b5", font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: "#9aa4b5", font: { size: 11 } }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { min: 0, ticks: { color: "#9aa4b5", font: { size: 11 } }, grid: { color: "rgba(255,255,255,0.06)" } },
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
    plugins: { legend: { labels: { color: "#9aa4b5", font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: "#9aa4b5", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.04)" } },
      y: { ticks: { color: "#9aa4b5", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.06)" } },
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
        datasets: [{ label: "Avg Wait (s) — Venus η=15", data: items.map(x => x.val), backgroundColor: ["#3b82f6", "#14b8a6", "#f59e0b", "#8b5cf6", "#ef4444"], borderRadius: 4 }],
      },
      options: opts,
    });
    tblEl.innerHTML = `<table class="tbl"><thead><tr><th>Configuration</th><th>Avg Wait</th><th>P99 Wait</th></tr></thead><tbody>
      ${items.map(x => `<tr><td class="mono text-white">${x.label}</td><td class="mono">${Math.round(x.val).toLocaleString()}s</td><td class="mono">${Math.round(x.p99).toLocaleString()}s</td></tr>`).join("")}
    </tbody></table>`;
  } else if (kind === "e4" && SAVED_RESEARCH.e4) {
    const rows = SAVED_RESEARCH.e4.filter(r => r.trace === "venus" && r.eta === 15).slice(0, 7);
    CHARTS.saved = new Chart(chartEl, {
      type: "bar",
      data: {
        labels: rows.map(r => r.label),
        datasets: [
          { label: "Avg Wait", data: rows.map(r => r.avg_wait), backgroundColor: "#3b82f6", borderRadius: 4 },
          { label: "P99 Wait", data: rows.map(r => r.p99_wait), backgroundColor: "#f59e0b", borderRadius: 4 },
        ],
      },
      options: opts,
    });
    tblEl.innerHTML = `<table class="tbl"><thead><tr><th>Policy</th><th>Avg Wait</th><th>P99 Wait</th><th>Starved (&gt;3×)</th></tr></thead><tbody>
      ${rows.map(r => `<tr><td class="mono text-white">${r.label}</td><td class="mono">${Math.round(r.avg_wait).toLocaleString()}s</td><td class="mono">${Math.round(r.p99_wait).toLocaleString()}s</td><td class="mono">${r.starved_3x}</td></tr>`).join("")}
    </tbody></table>`;
  } else if (kind === "e7" && SAVED_RESEARCH.e7_norm) {
    const normRows = SAVED_RESEARCH.e7_norm.filter(r => r.window === "dynamic");
    const raw = normRows.filter(r => !r.normalize);
    const norm = normRows.filter(r => r.normalize);
    CHARTS.saved = new Chart(chartEl, {
      type: "line",
      data: {
        labels: raw.map(r => `w=${r.w}`),
        datasets: [
          { label: "Raw Score", data: raw.map(r => r.avg_wait), borderColor: "#3b82f6", borderWidth: 2 },
          { label: "Normalised Score", data: norm.map(r => r.avg_wait), borderColor: "#8b5cf6", borderWidth: 2 },
        ],
      },
      options: opts,
    });
    tblEl.innerHTML = `<table class="tbl"><thead><tr><th>w</th><th>Raw Avg Wait</th><th>Normalised Avg Wait</th></tr></thead><tbody>
      ${raw.map((r, i) => `<tr><td class="mono">${r.w}</td><td class="mono">${Math.round(r.avg_wait).toLocaleString()}s</td><td class="mono">${Math.round(norm[i]?.avg_wait || 0).toLocaleString()}s</td></tr>`).join("")}
    </tbody></table>`;
  }
}
