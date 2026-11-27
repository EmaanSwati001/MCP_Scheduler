/* MCG-Sched Control Plane Frontend Logic */

const SESSION_ID = "sim_" + Math.random().toString(36).slice(2, 10);

let META = { traces: [], topologies: [], schedulers: [] };
let SNAPSHOT = null;
let PLAY_TIMER = null;
let IS_PLAYING = false;
let LAST_COMPARE_DATA = null;
let SAVED_RESEARCH = null;
let ACTIVE_RESEARCH_SUB = "e5";

// Chart instances
const CHARTS = {};

// Palette for multi-scheduler charts
const PALETTE = [
  "#38bdf8", "#10b981", "#f59e0b", "#a855f7",
  "#f43f5e", "#6366f1", "#14b8a6", "#fb7185",
  "#84cc16", "#eab308", "#ec4899", "#60a5fa"
];

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  lucide.createIcons();
  await loadMetadata();
  await initSimulation();
  // Pre-load N2 inspector & saved research in background
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
  if (!res.ok) {
    let msg = await res.text();
    try { msg = JSON.parse(msg).detail || msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
}

function fmtTime(sec) {
  if (sec === null || sec === undefined || isNaN(sec)) return "0s";
  const s = Math.round(sec);
  if (s < 1000) return `${s}s`;
  if (s < 3600) return `${(sec / 60).toFixed(1)}m (${s}s)`;
  return `${(sec / 3600).toFixed(1)}h (${s.toLocaleString()}s)`;
}

function fmtShortTime(sec) {
  if (sec === null || sec === undefined || isNaN(sec)) return "0s";
  if (sec >= 10000) return `${(sec / 1000).toFixed(1)}k s`;
  return `${Math.round(sec).toLocaleString()}s`;
}

// ---------------------------------------------------------------------------
// Metadata hydration
// ---------------------------------------------------------------------------
async function loadMetadata() {
  META = await apiGet("/api/meta");

  // Populate trace selects
  const traceOpts = META.traces
    .filter(t => t.available)
    .map(t => {
      const cnt = t.total_jobs ? ` (${t.total_jobs.toLocaleString()} jobs)` : "";
      return `<option value="${t.id}">${t.name}${cnt}</option>`;
    })
    .join("");
  document.getElementById("live-trace").innerHTML = traceOpts;
  document.getElementById("cmp-trace").innerHTML = traceOpts;
  document.getElementById("e5-trace").innerHTML = META.traces
    .filter(t => t.available && t.id !== "synthetic")
    .map(t => `<option value="${t.id}">${t.name}</option>`)
    .join("");

  // Populate topology selects
  const topoOpts = META.topologies
    .map(t => `<option value="${t.id}">${t.name}</option>`)
    .join("");
  document.getElementById("live-topo").innerHTML = topoOpts;
  document.getElementById("cmp-topo").innerHTML = topoOpts;

  // Grouped scheduler select for Live Simulator
  const byCat = {};
  META.schedulers.forEach(s => {
    if (!byCat[s.category]) byCat[s.category] = [];
    byCat[s.category].push(s);
  });
  let schedHtml = "";
  for (const [cat, items] of Object.entries(byCat)) {
    schedHtml += `<optgroup label="${cat}">`;
    items.forEach(s => {
      const sel = s.id === "mcg" ? "selected" : "";
      schedHtml += `<option value="${s.id}" ${sel}>${s.name} [${s.id}]</option>`;
    });
    schedHtml += `</optgroup>`;
  }
  document.getElementById("live-sched").innerHTML = schedHtml;

  // Populate Benchmark Arena chips
  const defaultSelected = new Set(["fcfs", "consolidate", "fgd", "mcg", "mcg_fair", "mcg_fc"]);
  document.getElementById("cmp-sched-chips").innerHTML = META.schedulers
    .map(s => {
      const active = defaultSelected.has(s.id) ? "selected" : "";
      return `<button type="button" class="sched-chip ${active}" data-sched="${s.id}" onclick="this.classList.toggle('selected')">
        <span class="mono text-[11px]">${s.id}</span>
      </button>`;
    })
    .join("");

  updateTopologyBadge();
  updateSchedulerMeta();
}

function updateTopologyBadge() {
  const id = document.getElementById("live-topo").value;
  const topo = META.topologies.find(t => t.id === id);
  if (!topo) return;
  const mixBadges = Object.entries(topo.server_mix)
    .map(([cap, cnt]) => {
      const cls = cap === "4" ? "badge-purple" : "badge-cyan";
      return `<span class="badge ${cls}">${cnt} × ${cap}-GPU</span>`;
    })
    .join("");
  document.getElementById("live-topo-info").innerHTML =
    `<span class="badge badge-slate">${topo.clusters} clusters</span>
     <span class="badge badge-emerald">${topo.total_gpus.toLocaleString()} GPUs</span>
     ${mixBadges}`;
}

function updateSchedulerMeta() {
  const id = document.getElementById("live-sched").value;
  const s = META.schedulers.find(x => x.id === id);
  if (!s) return;
  document.getElementById("live-sched-desc").textContent = s.description;

  const paramDefs = {
    w: { label: "Weight w (0..1)", val: 0.5, step: 0.05, min: 0, max: 1 },
    lam: { label: "Queue Blend λ", val: 1.0, step: 0.1, min: 0, max: 10 },
    l_star: { label: "Window L*", val: 10, step: 1, min: 1, max: 50 },
    alpha: { label: "Decay α", val: 0.05, step: 0.01, min: 0, max: 1 },
    beta: { label: "Aging β", val: 1.0, step: 0.25, min: 0, max: 5 },
    k: { label: "Guard k (×W_ref)", val: 3.0, step: 0.5, min: 0.2, max: 10 },
    max_reservations: { label: "Max Reserves", val: 1, step: 1, min: 1, max: 8 },
  };

  const box = document.getElementById("live-params-box");
  if (!s.params || s.params.length === 0) {
    box.innerHTML = `<div class="col-span-2 text-[11px] text-slate-500 italic">No tunable hyperparameters for this baseline.</div>`;
    return;
  }
  box.innerHTML = s.params
    .filter(p => paramDefs[p])
    .map(p => {
      const d = paramDefs[p];
      return `<div>
        <label class="ctrl-label">${d.label}</label>
        <input type="number" id="param-${p}" value="${d.val}" step="${d.step}" min="${d.min}" max="${d.max}" class="ctrl-input mono !py-1 text-xs" />
      </div>`;
    })
    .join("");
}

function readSchedulerParams() {
  const id = document.getElementById("live-sched").value;
  const s = META.schedulers.find(x => x.id === id);
  if (!s || !s.params) return {};
  const params = {};
  s.params.forEach(p => {
    const el = document.getElementById(`param-${p}`);
    if (el && el.value !== "") {
      params[p] = parseFloat(el.value);
    }
  });
  return params;
}

// ---------------------------------------------------------------------------
// Tab Switching
// ---------------------------------------------------------------------------
function switchTab(tabId) {
  document.querySelectorAll(".nav-tab").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-view").forEach(sec => {
    sec.classList.toggle("hidden", sec.id !== `tab-${tabId}`);
  });
  if (tabId === "research" && SAVED_RESEARCH) {
    renderResearchTab();
  }
}

function switchInspectorSub(sub) {
  document.querySelectorAll("[data-sub]").forEach(b => b.classList.toggle("active", b.dataset.sub === sub));
  ["scores", "window", "n1log"].forEach(k => {
    document.getElementById(`inspector-${k}`).classList.toggle("hidden", k !== sub);
  });
}

function switchQueueSub(sub) {
  document.querySelectorAll("[data-qsub]").forEach(b => b.classList.toggle("active", b.dataset.qsub === sub));
  ["waiting", "running", "events"].forEach(k => {
    document.getElementById(`queue-${k}`).classList.toggle("hidden", k !== sub);
  });
}

// ---------------------------------------------------------------------------
// Quick Presets
// ---------------------------------------------------------------------------
async function applyPreset(kind) {
  stopPlay();
  if (kind === "burst") {
    document.getElementById("live-trace").value = "synthetic";
    document.getElementById("live-njobs").value = 800;
    document.getElementById("live-eta").value = 10;
    document.getElementById("live-eta-val").textContent = "10.0×";
    document.getElementById("live-burst").checked = true;
    document.getElementById("live-topo").value = "venus15";
    document.getElementById("live-sched").value = "mcg";
  } else if (kind === "fairness") {
    document.getElementById("live-trace").value = "venus";
    document.getElementById("live-njobs").value = 800;
    document.getElementById("live-eta").value = 15;
    document.getElementById("live-eta-val").textContent = "15.0×";
    document.getElementById("live-burst").checked = false;
    document.getElementById("live-topo").value = "venus15";
    document.getElementById("live-sched").value = "mcg_fair";
  } else if (kind === "mixed") {
    document.getElementById("live-trace").value = "venus";
    document.getElementById("live-njobs").value = 600;
    document.getElementById("live-eta").value = 15;
    document.getElementById("live-eta-val").textContent = "15.0×";
    document.getElementById("live-burst").checked = false;
    document.getElementById("live-topo").value = "venus15_mixed";
    document.getElementById("live-sched").value = "mcg_fc";
  } else if (kind === "philly") {
    document.getElementById("live-trace").value = "philly";
    document.getElementById("live-njobs").value = 500;
    document.getElementById("live-eta").value = 4;
    document.getElementById("live-eta-val").textContent = "4.0×";
    document.getElementById("live-burst").checked = false;
    document.getElementById("live-topo").value = "paper_8x16";
    document.getElementById("live-sched").value = "mcg";
  }
  updateTopologyBadge();
  updateSchedulerMeta();
  await initSimulation();
  // Advance a few steps so something interesting is immediately visible on screen
  await stepSimulation(kind === "burst" ? 1 : 20);
}

// ---------------------------------------------------------------------------
// Live Simulator Actions
// ---------------------------------------------------------------------------
function setGlobalStatus(state, text) {
  const dot = document.getElementById("status-dot");
  const txt = document.getElementById("status-text");
  dot.className = "w-2 h-2 rounded-full " + (
    state === "running" ? "bg-sky-400 animate-ping" :
    state === "done" ? "bg-emerald-400" : "bg-amber-400"
  );
  txt.textContent = text;
}

async function initSimulation() {
  stopPlay();
  setGlobalStatus("idle", "Initializing...");
  const payload = {
    session_id: SESSION_ID,
    trace_id: document.getElementById("live-trace").value,
    topo_id: document.getElementById("live-topo").value,
    sched_id: document.getElementById("live-sched").value,
    n_jobs: parseInt(document.getElementById("live-njobs").value || "0", 10),
    eta: parseFloat(document.getElementById("live-eta").value || "10"),
    burst: document.getElementById("live-burst").checked,
    seed: parseInt(document.getElementById("live-seed").value || "0", 10),
    params: readSchedulerParams(),
  };
  SNAPSHOT = await apiPost("/api/sim/init", payload);
  renderSnapshot();
  setGlobalStatus("idle", "Initialized");
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
    setGlobalStatus("done", "Complete");
  }
}

async function runToEnd() {
  stopPlay();
  if (!SNAPSHOT) return;
  setGlobalStatus("running", "Running all events...");
  document.getElementById("btn-run-end").disabled = true;
  try {
    SNAPSHOT = await apiPost("/api/sim/run", { session_id: SESSION_ID });
    renderSnapshot();
    setGlobalStatus("done", "Complete");
  } finally {
    document.getElementById("btn-run-end").disabled = SNAPSHOT?.done ?? false;
  }
}

function togglePlay() {
  if (IS_PLAYING) {
    stopPlay();
  } else {
    startPlay();
  }
}

function startPlay() {
  if (!SNAPSHOT || SNAPSHOT.done) return;
  IS_PLAYING = true;
  const btn = document.getElementById("btn-play");
  btn.innerHTML = `<i data-lucide="pause" class="w-3.5 h-3.5"></i> Pause`;
  btn.classList.remove("btn-emerald");
  btn.classList.add("btn-amber");
  lucide.createIcons();
  setGlobalStatus("running", "Simulating...");

  const tick = async () => {
    if (!IS_PLAYING || !SNAPSHOT || SNAPSHOT.done) {
      stopPlay();
      return;
    }
    const delay = parseInt(document.getElementById("play-speed").value, 10);
    const stepSize = delay <= 80 ? 5 : 1;
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
    btn.classList.add("btn-emerald");
    btn.classList.remove("btn-amber");
    lucide.createIcons();
  }
  if (SNAPSHOT && !SNAPSHOT.done) {
    setGlobalStatus("idle", "Paused");
  }
}

// ---------------------------------------------------------------------------
// Render Snapshot
// ---------------------------------------------------------------------------
function renderSnapshot() {
  const s = SNAPSHOT;
  if (!s) return;
  const m = s.metrics;

  document.getElementById("header-clock").textContent = `t = ${s.now.toLocaleString()} s`;

  // Progress & button states
  const pct = s.total_jobs > 0 ? Math.round((m.completed / s.total_jobs) * 100) : 0;
  document.getElementById("sim-progress-pct").textContent = `${pct}%`;
  document.getElementById("sim-progress-bar").style.width = `${pct}%`;

  document.getElementById("btn-step1").disabled = s.done;
  document.getElementById("btn-step10").disabled = s.done;
  document.getElementById("btn-play").disabled = s.done;
  document.getElementById("btn-run-end").disabled = s.done;

  // KPI cards
  document.getElementById("kpi-done").textContent = m.completed.toLocaleString();
  document.getElementById("kpi-total").textContent = `/ ${s.total_jobs.toLocaleString()}`;
  document.getElementById("kpi-job-sub").textContent = `run ${m.running} · wait ${m.waiting}`;

  document.getElementById("kpi-util").textContent = `${m.utilization_pct.toFixed(1)}%`;
  document.getElementById("kpi-gpus-sub").textContent = `${m.used_gpus} / ${m.total_gpus} GPUs`;

  document.getElementById("kpi-avg-wait").textContent = fmtShortTime(s.summary ? s.summary.avg_wait : m.avg_wait);
  document.getElementById("kpi-p99-wait").textContent = `P99: ${fmtShortTime(s.summary ? s.summary.p99_wait : m.p99_wait)}`;

  document.getElementById("kpi-lb").textContent = (s.summary ? s.summary.avg_lb_index : m.lb_index).toFixed(5);

  document.getElementById("kpi-stranded").textContent = m.stranded_gpus;
  document.getElementById("kpi-free-sub").textContent = `${m.free_gpus} free GPUs`;

  document.getElementById("kpi-protected").textContent = m.protected_count;
  document.getElementById("kpi-reserved-sub").textContent =
    `${m.reserved_count} reserved · ${m.starved_completed + m.starved_waiting} starved`;

  renderClusters();
  renderLiveCharts();
  renderDecisionInspector();
  renderJobMonitor();
}

function renderClusters() {
  const s = SNAPSHOT;
  if (!s) return;
  const filter = document.getElementById("grid-filter").value;
  let totalServers = 0;
  let cap4 = 0, cap8 = 0;
  s.clusters.forEach(c => {
    c.servers.forEach(sv => {
      totalServers++;
      if (sv.capacity === 4) cap4++;
      else if (sv.capacity === 8) cap8++;
    });
  });
  document.getElementById("grid-server-summary").textContent =
    cap4 > 0 ? `${cap4}×4G + ${cap8}×8G (${totalServers} nodes)` : `${totalServers} nodes`;

  const html = s.clusters
    .filter(c => {
      if (filter === "active") return c.used > 0;
      if (filter === "reserved") return c.servers.some(sv => sv.reserved_for !== null);
      return true;
    })
    .map(c => {
      const utilPct = (c.utilization * 100).toFixed(0);
      const serversHtml = c.servers
        .filter(sv => filter !== "reserved" || sv.reserved_for !== null)
        .map(sv => {
          const isRes = sv.reserved_for !== null;
          const capCls = sv.capacity <= 4 ? "cap-4" : "";
          const resCls = isRes ? "reserved" : "";
          const runTip = sv.running.length
            ? sv.running.map(r => `Job #${r.job_id} (${r.gpus}G)`).join(", ")
            : "Idle";
          const tip = `C${c.cluster_id}·S${sv.server_id} (${sv.capacity}G) | Free: ${sv.free} | ${runTip}` +
            (isRes ? ` | RESERVED for Job #${sv.reserved_for}` : "");

          let slots = "";
          for (let i = 0; i < sv.used; i++) {
            slots += `<span class="gpu-slot used"></span>`;
          }
          for (let i = 0; i < sv.free; i++) {
            slots += `<span class="gpu-slot ${isRes ? "reserved-free" : "free"}"></span>`;
          }

          const capTag = sv.capacity <= 4
            ? `<span class="text-[9.5px] font-bold text-purple-300 bg-purple-500/20 px-1 rounded">4G</span>`
            : `<span class="text-[9.5px] font-bold text-sky-300 bg-sky-500/15 px-1 rounded">8G</span>`;

          const resTag = isRes
            ? `<span class="text-[9px] font-bold text-amber-300 ml-auto">🔒#${sv.reserved_for}</span>`
            : `<span class="text-[9.5px] mono text-slate-400 ml-auto">${sv.used}/${sv.capacity}</span>`;

          return `<div class="server-card ${capCls} ${resCls}" title="${tip}">
            <div class="flex items-center gap-1">
              <span class="mono text-[10px] text-slate-300 font-semibold">S${sv.server_id}</span>
              ${capTag}
              ${resTag}
            </div>
            <div class="gpu-slots">${slots}</div>
          </div>`;
        })
        .join("");

      return `<div class="cluster-row">
        <div class="flex items-center justify-between mb-2">
          <div class="flex items-center gap-2">
            <span class="text-xs font-semibold text-slate-200">Cluster ${c.cluster_id}</span>
            <span class="mono text-[11px] text-slate-400">${c.used}/${c.capacity} GPUs</span>
          </div>
          <div class="flex items-center gap-2 w-36">
            <div class="flex-1 bg-slate-900 h-1.5 rounded-full overflow-hidden">
              <div class="bg-sky-400 h-full" style="width:${utilPct}%"></div>
            </div>
            <span class="mono text-[11px] text-sky-400 w-9 text-right">${utilPct}%</span>
          </div>
        </div>
        <div class="flex flex-wrap gap-1.5">${serversHtml}</div>
      </div>`;
    })
    .join("");

  document.getElementById("cluster-grid").innerHTML =
    html || `<div class="text-center text-xs text-slate-500 py-10">No clusters match the current filter.</div>`;
}

function renderLiveCharts() {
  const tl = SNAPSHOT?.timeline || [];
  const labels = tl.map(p => `${Math.round(p.t)}s`);
  const utils = tl.map(p => p.util);
  const qlens = tl.map(p => p.queue_len);
  const lbs = tl.map(p => p.lb_index);

  // Chart 1: Utilization & Queue Length
  if (!CHARTS.utilQueue) {
    const ctx = document.getElementById("chart-util-queue").getContext("2d");
    CHARTS.utilQueue = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "GPU Util (%)",
            data: utils,
            borderColor: "#38bdf8",
            backgroundColor: "rgba(56,189,248,0.12)",
            fill: true,
            tension: 0.25,
            pointRadius: 0,
            borderWidth: 2,
            yAxisID: "y",
          },
          {
            label: "Queue Len",
            data: qlens,
            borderColor: "#f43f5e",
            backgroundColor: "rgba(244,63,94,0.08)",
            fill: true,
            tension: 0.25,
            pointRadius: 0,
            borderWidth: 1.8,
            yAxisID: "y1",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { labels: { color: "#cbd5e1", boxWidth: 10, font: { size: 11 } } } },
        scales: {
          x: { ticks: { color: "#64748b", maxTicksLimit: 6, font: { size: 10 } }, grid: { color: "rgba(148,163,184,0.06)" } },
          y: { position: "left", min: 0, max: 100, ticks: { color: "#38bdf8", font: { size: 10 } }, grid: { color: "rgba(148,163,184,0.08)" } },
          y1: { position: "right", min: 0, ticks: { color: "#fb7185", font: { size: 10 } }, grid: { drawOnChartArea: false } },
        },
      },
    });
  } else {
    CHARTS.utilQueue.data.labels = labels;
    CHARTS.utilQueue.data.datasets[0].data = utils;
    CHARTS.utilQueue.data.datasets[1].data = qlens;
    CHARTS.utilQueue.update("none");
  }

  // Chart 2: Load Balance Index
  if (!CHARTS.lb) {
    const ctx = document.getElementById("chart-lb").getContext("2d");
    CHARTS.lb = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Load-Balance Variance (σ²)",
            data: lbs,
            borderColor: "#818cf8",
            backgroundColor: "rgba(129,140,248,0.14)",
            fill: true,
            tension: 0.25,
            pointRadius: 0,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { labels: { color: "#cbd5e1", boxWidth: 10, font: { size: 11 } } } },
        scales: {
          x: { ticks: { color: "#64748b", maxTicksLimit: 6, font: { size: 10 } }, grid: { color: "rgba(148,163,184,0.06)" } },
          y: { min: 0, ticks: { color: "#a5b4fc", font: { size: 10 } }, grid: { color: "rgba(148,163,184,0.08)" } },
        },
      },
    });
  } else {
    CHARTS.lb.data.labels = labels;
    CHARTS.lb.data.datasets[0].data = lbs;
    CHARTS.lb.update("none");
  }
}

function renderDecisionInspector() {
  const d = SNAPSHOT?.last_decision;
  const banner = document.getElementById("decision-banner");
  const scoresDiv = document.getElementById("inspector-scores");
  const winDiv = document.getElementById("inspector-window");
  const n1Div = document.getElementById("inspector-n1log");

  if (!d) {
    banner.innerHTML = `<span class="text-slate-400">No active scheduling round captured yet. Click <span class="text-sky-400 font-semibold">+1</span> or <span class="text-emerald-400 font-semibold">Play</span> to inspect live placement scores.</span>`;
    scoresDiv.innerHTML = `<div class="p-6 text-center text-xs text-slate-500">Waiting for first scheduling decision...</div>`;
    winDiv.innerHTML = `<div class="p-6 text-center text-xs text-slate-500">Queue window empty.</div>`;
  } else {
    const selPlacement = d.placements.find(p => p.job_id === d.focal_job_id);
    const outcomeBadge = selPlacement
      ? `<span class="badge badge-emerald">Placed → C${selPlacement.cluster_id}·S${selPlacement.server_id}</span>`
      : `<span class="badge badge-amber">Waiting / Blocked</span>`;

    banner.innerHTML = `
      <div class="flex items-center gap-2 flex-wrap">
        <span class="mono text-slate-400">@ t=${d.time}s</span>
        <span class="font-semibold text-white">Focal Job #${d.focal_job_id}</span>
        <span class="badge badge-cyan">${d.focal_gpus} GPUs</span>
        <span class="text-slate-400">waited ${d.focal_waited}s</span>
        ${outcomeBadge}
      </div>
      <span class="text-[11px] text-slate-400">${d.candidates_total} feasible servers · w=${d.w}</span>`;

    if (d.candidates.length === 0) {
      scoresDiv.innerHTML = `<div class="p-6 text-center text-xs text-amber-400">No server currently has ≥ ${d.focal_gpus} free GPUs.</div>`;
    } else {
      scoresDiv.innerHTML = `<table class="data-table">
        <thead>
          <tr>
            <th>Server</th>
            <th>Cap</th>
            <th>Free</th>
            <th>G (Orig)</th>
            <th>G_C (N2)</th>
            <th>σ²</th>
            <th>Score v</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          ${d.candidates.map(c => {
            const cls = c.selected ? "row-selected" : (c.reserved_excluded ? "row-excluded" : "");
            const status = c.selected
              ? `<span class="badge badge-emerald">CHOSEN</span>`
              : c.reserved_excluded
              ? `<span class="badge badge-amber">RESERVED</span>`
              : `<span class="text-slate-500 text-[11px]">candidate</span>`;
            return `<tr class="${cls}">
              <td class="mono font-semibold">C${c.cluster_id}·S${c.server_id}</td>
              <td><span class="badge ${c.capacity <= 4 ? "badge-purple" : "badge-slate"}">${c.capacity}G</span></td>
              <td class="mono">${c.free}</td>
              <td class="mono text-sky-300">${c.g_orig.toFixed(4)}</td>
              <td class="mono text-purple-300">${c.g_fc.toFixed(4)}</td>
              <td class="mono text-indigo-300">${c.sigma2.toFixed(5)}</td>
              <td class="mono font-semibold text-white">${c.score.toFixed(4)}</td>
              <td>${status}</td>
            </tr>`;
          }).join("")}
        </tbody>
      </table>`;
    }

    // Window order preview
    winDiv.innerHTML = `<table class="data-table">
      <thead>
        <tr><th>Pos</th><th>Job</th><th>GPUs</th><th>Waited</th><th>Wait / W_ref</th><th>Aging Bonus</th><th>State</th></tr>
      </thead>
      <tbody>
        ${d.window_preview.map(j => {
          const stateBadge = j.placed
            ? `<span class="badge badge-emerald">Started C${j.placed[0]}·S${j.placed[1]}</span>`
            : j.protected
            ? `<span class="badge badge-amber">🛡️ PROTECTED</span>`
            : `<span class="text-slate-400 text-[11px]">in queue</span>`;
          return `<tr>
            <td class="mono text-slate-400">#${j.pos + 1}</td>
            <td class="mono font-semibold">Job #${j.job_id}</td>
            <td><span class="badge badge-cyan">${j.gpus}G</span></td>
            <td class="mono">${j.waited}s</td>
            <td class="mono">${j.wait_ratio}×</td>
            <td class="mono text-emerald-400">-${j.aging_bonus.toFixed(3)}</td>
            <td>${stateBadge}</td>
          </tr>`;
        }).join("")}
      </tbody>
    </table>`;
  }

  // N1 Log
  const log = SNAPSHOT?.n1_log || [];
  if (log.length === 0) {
    n1Div.innerHTML = `<div class="p-6 text-center text-xs text-slate-500">No N1 guard or drain reservation events recorded yet (use <span class="mono text-amber-300">mcg_fair</span> or <span class="mono text-amber-300">mcg_guard</span>).</div>`;
  } else {
    n1Div.innerHTML = `<table class="data-table">
      <thead><tr><th>Time</th><th>Action</th><th>Job</th><th>Target Server</th></tr></thead>
      <tbody>
        ${[...log].reverse().map(e => {
          const bCls = e.action === "reserve" ? "badge-amber" : e.action === "protect" ? "badge-rose" : "badge-emerald";
          const srv = e.server ? `C${e.server[0]}·S${e.server[1]}` : "—";
          return `<tr>
            <td class="mono">${e.time}s</td>
            <td><span class="badge ${bCls}">${e.action.toUpperCase()}</span></td>
            <td class="mono">Job #${e.job_id}</td>
            <td class="mono">${srv}</td>
          </tr>`;
        }).join("")}
      </tbody>
    </table>`;
  }
}

function renderJobMonitor() {
  const s = SNAPSHOT;
  if (!s) return;
  document.getElementById("qtab-wait-cnt").textContent = s.metrics.waiting;
  document.getElementById("qtab-run-cnt").textContent = s.metrics.running;

  // Waiting queue
  const qDiv = document.getElementById("queue-waiting");
  if (s.queue.length === 0) {
    qDiv.innerHTML = `<div class="p-6 text-center text-xs text-slate-500">Waiting queue is empty.</div>`;
  } else {
    qDiv.innerHTML = `<table class="data-table">
      <thead><tr><th>Job</th><th>Demand</th><th>Arrival</th><th>Waited</th><th>Wait / W_ref</th><th>Status</th></tr></thead>
      <tbody>
        ${s.queue.map(j => {
          const tag = j.protected
            ? `<span class="badge badge-amber">🛡️ PROTECTED</span>`
            : j.starved
            ? `<span class="badge badge-rose">STARVED &gt;3×</span>`
            : `<span class="badge badge-slate">Waiting</span>`;
          return `<tr>
            <td class="mono font-semibold">#${j.job_id}</td>
            <td><span class="badge ${j.gpus >= 8 ? "badge-purple" : "badge-cyan"}">${j.gpus}G</span></td>
            <td class="mono text-slate-400">${j.arrival_time}s</td>
            <td class="mono">${j.waited}s</td>
            <td class="mono">${j.wait_ratio}×</td>
            <td>${tag}</td>
          </tr>`;
        }).join("")}
      </tbody>
    </table>`;
  }

  // Running jobs
  const rDiv = document.getElementById("queue-running");
  if (s.running_jobs.length === 0) {
    rDiv.innerHTML = `<div class="p-6 text-center text-xs text-slate-500">No jobs currently running.</div>`;
  } else {
    rDiv.innerHTML = `<table class="data-table">
      <thead><tr><th>Job</th><th>GPUs</th><th>Node</th><th>Started</th><th>Remaining</th></tr></thead>
      <tbody>
        ${s.running_jobs.map(j => `<tr>
          <td class="mono font-semibold">#${j.job_id}</td>
          <td><span class="badge badge-cyan">${j.gpus}G</span></td>
          <td class="mono">${j.placement ? `C${j.placement[0]}·S${j.placement[1]}` : "—"}</td>
          <td class="mono text-slate-400">${j.start_time}s</td>
          <td class="mono text-emerald-400">${j.remaining}s</td>
        </tr>`).join("")}
      </tbody>
    </table>`;
  }

  // Event feed
  const eDiv = document.getElementById("queue-events");
  if (s.recent_events.length === 0) {
    eDiv.innerHTML = `<div class="p-6 text-center text-xs text-slate-500">No simulation events yet.</div>`;
  } else {
    eDiv.innerHTML = `<table class="data-table">
      <thead><tr><th>Time</th><th>Event</th><th>Job</th><th>GPUs</th><th>Info</th></tr></thead>
      <tbody>
        ${s.recent_events.map(ev => {
          const b = ev.event === "start" ? "badge-emerald" :
                    ev.event === "finish" ? "badge-cyan" :
                    ev.event === "reject" ? "badge-rose" : "badge-slate";
          const info = ev.placement ? `C${ev.placement[0]}·S${ev.placement[1]}` :
                       ev.wait !== null ? `wait ${ev.wait}s` : "";
          return `<tr>
            <td class="mono text-slate-400">${ev.time}s</td>
            <td><span class="badge ${b}">${ev.event.toUpperCase()}</span></td>
            <td class="mono">#${ev.job_id}</td>
            <td class="mono">${ev.gpus}G</td>
            <td class="mono text-xs text-slate-300">${info}</td>
          </tr>`;
        }).join("")}
      </tbody>
    </table>`;
  }
}

// ---------------------------------------------------------------------------
// TAB 2: Benchmark Arena
// ---------------------------------------------------------------------------
function selectCompareGroup(group) {
  const map = {
    core: new Set(["fcfs", "consolidate", "fgd", "mast", "mcg"]),
    fairness: new Set(["mcg", "mcg_aging", "mcg_guard", "mcg_fair", "mcg_norm"]),
    hetero: new Set(["fcfs", "fgd", "mcg", "mcg_fc", "mcg_fc_norm"]),
  };
  const set = map[group] || new Set();
  document.querySelectorAll("#cmp-sched-chips .sched-chip").forEach(btn => {
    btn.classList.toggle("selected", set.has(btn.dataset.sched));
  });
}

async function runComparisonArena() {
  const selected = Array.from(document.querySelectorAll("#cmp-sched-chips .sched-chip.selected"))
    .map(b => b.dataset.sched);
  if (selected.length === 0) {
    alert("Select at least one scheduler to compare.");
    return;
  }
  const btn = document.getElementById("btn-run-cmp");
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader" class="w-4 h-4 animate-spin"></i> Running ${selected.length} policies...`;
  lucide.createIcons();

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
    renderComparisonResults(data);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="play" class="w-4 h-4"></i> Run Head-to-Head Benchmark`;
    lucide.createIcons();
  }
}

function renderComparisonResults(data) {
  document.getElementById("cmp-placeholder").classList.add("hidden");
  document.getElementById("cmp-results-wrap").classList.remove("hidden");

  const meta = data.meta;
  document.getElementById("cmp-meta-badge").textContent =
    `${meta.trace_id} · ${meta.topo_id} · ${meta.n_jobs} jobs · η=${meta.eta} · W_ref=${meta.w_ref}s`;

  const rows = data.results;
  const bestAvgWait = Math.min(...rows.map(r => r.avg_wait));
  const bestP99 = Math.min(...rows.map(r => r.p99_wait));
  const bestLB = Math.min(...rows.map(r => r.avg_lb_index));

  document.getElementById("cmp-table-wrap").innerHTML = `<table class="data-table">
    <thead>
      <tr>
        <th>Scheduler</th>
        <th>Avg Wait (s)</th>
        <th>P99 Wait (s)</th>
        <th>8-GPU Wait (s)</th>
        <th>Starved (&gt;3×)</th>
        <th>GPU Util</th>
        <th>Load Balance σ²</th>
        <th>Unalloc Ratio</th>
        <th>Completed</th>
      </tr>
    </thead>
    <tbody>
      ${rows.map(r => `<tr>
        <td><span class="font-semibold text-white">${r.id}</span> <span class="text-[11px] text-slate-400">(${r.category})</span></td>
        <td class="mono ${r.avg_wait === bestAvgWait ? "text-emerald-400 font-bold" : ""}">${r.avg_wait.toLocaleString()}</td>
        <td class="mono ${r.p99_wait === bestP99 ? "text-emerald-400 font-bold" : ""}">${r.p99_wait.toLocaleString()}</td>
        <td class="mono">${r.avg_wait_8gpu.toLocaleString()}</td>
        <td class="mono">${r.starved_3x}</td>
        <td class="mono">${r.gpu_utilization.toFixed(2)}%</td>
        <td class="mono ${r.avg_lb_index === bestLB ? "text-sky-400 font-bold" : ""}">${r.avg_lb_index.toFixed(5)}</td>
        <td class="mono">${r.unalloc_ratio.toFixed(2)}%</td>
        <td class="mono">${r.jobs_completed}</td>
      </tr>`).join("")}
    </tbody>
  </table>`;

  // Destroy prior comparison charts
  ["cmpWait", "cmpBySize", "cmpLB", "cmpQTL"].forEach(k => {
    if (CHARTS[k]) { CHARTS[k].destroy(); CHARTS[k] = null; }
  });

  const labels = rows.map(r => r.id);
  const darkOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: "#94a3b8", font: { size: 11 } }, grid: { color: "rgba(148,163,184,0.06)" } },
      y: { ticks: { color: "#94a3b8", font: { size: 11 } }, grid: { color: "rgba(148,163,184,0.08)" } },
    },
  };

  CHARTS.cmpWait = new Chart(document.getElementById("cmp-chart-wait"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Avg Wait (s)", data: rows.map(r => r.avg_wait), backgroundColor: "#38bdf8", borderRadius: 5 },
        { label: "P99 Wait (s)", data: rows.map(r => r.p99_wait), backgroundColor: "#818cf8", borderRadius: 5 },
      ],
    },
    options: darkOpts,
  });

  // By size (1, 2, 4, 8)
  const sizes = ["1", "2", "4", "8"];
  CHARTS.cmpBySize = new Chart(document.getElementById("cmp-chart-bysize"), {
    type: "bar",
    data: {
      labels,
      datasets: sizes.map((sz, idx) => ({
        label: `${sz}-GPU Jobs`,
        data: rows.map(r => r.avg_wait_by_gpus[sz] || 0),
        backgroundColor: PALETTE[idx],
        borderRadius: 4,
      })),
    },
    options: darkOpts,
  });

  CHARTS.cmpLB = new Chart(document.getElementById("cmp-chart-lb"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Avg Load-Balance Index σ²", data: rows.map(r => r.avg_lb_index), backgroundColor: "#10b981", borderRadius: 5 },
      ],
    },
    options: darkOpts,
  });

  // Queue length trajectory
  const firstSid = rows[0]?.id;
  const baseTL = data.timelines[firstSid] || [];
  CHARTS.cmpQTL = new Chart(document.getElementById("cmp-chart-qtl"), {
    type: "line",
    data: {
      labels: baseTL.map(pt => `${Math.round(pt.t)}s`),
      datasets: rows.map((r, idx) => ({
        label: r.id,
        data: (data.timelines[r.id] || []).map(pt => pt.queue_len),
        borderColor: PALETTE[idx % PALETTE.length],
        tension: 0.25,
        pointRadius: 0,
        borderWidth: 2,
      })),
    },
    options: darkOpts,
  });
}

function exportComparisonCSV() {
  if (!LAST_COMPARE_DATA) {
    alert("Run a benchmark comparison first.");
    return;
  }
  const cols = ["id", "avg_wait", "p95_wait", "p99_wait", "max_wait", "avg_wait_8gpu", "starved_3x", "gpu_utilization", "avg_lb_index", "unalloc_ratio", "jobs_completed"];
  const lines = [cols.join(",")];
  LAST_COMPARE_DATA.results.forEach(r => {
    lines.push(cols.map(c => r[c]).join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `mcgsched_benchmark_${LAST_COMPARE_DATA.meta.trace_id}.csv`;
  a.click();
}

// ---------------------------------------------------------------------------
// TAB 3: Fairness & Heterogeneity Lab (N2 Inspector + Live E5)
// ---------------------------------------------------------------------------
async function runN2Inspector() {
  const w1 = parseFloat(document.getElementById("n2-w1").value || "0");
  const w2 = parseFloat(document.getElementById("n2-w2").value || "0");
  const w4 = parseFloat(document.getElementById("n2-w4").value || "0");
  const w8 = parseFloat(document.getElementById("n2-w8").value || "0");
  const jobGpus = parseInt(document.getElementById("n2-jobgpus").value || "2", 10);

  const data = await apiPost("/api/experiments/inspect_n2", {
    demand_weights: { "1": w1, "2": w2, "4": w4, "8": w8 },
    job_gpus: jobGpus,
  });

  const eg = data.empty_gradients;
  document.getElementById("n2-gradient-cards").innerHTML = `
    <div class="bg-[#0b1120] border border-purple-500/30 rounded-xl p-3">
      <div class="flex items-center justify-between mb-1.5">
        <span class="text-xs font-semibold text-purple-300">Empty 4-GPU Server</span>
        <span class="badge badge-purple">cap = 4G</span>
      </div>
      <div class="space-y-1 text-xs mono">
        <div class="flex justify-between"><span class="text-slate-400">Original G:</span><span class="text-sky-300 font-semibold">${eg.server_4g.G_orig}</span></div>
        <div class="flex justify-between"><span class="text-slate-400">F_C (Truncated):</span><span class="text-amber-300 font-semibold">${eg.server_4g.G_fc}</span></div>
        <div class="flex justify-between"><span class="text-slate-400">F_C (Renormed):</span><span class="text-purple-300 font-semibold">${eg.server_4g.G_fc_norm}</span></div>
      </div>
    </div>
    <div class="bg-[#0b1120] border border-sky-500/30 rounded-xl p-3">
      <div class="flex items-center justify-between mb-1.5">
        <span class="text-xs font-semibold text-sky-300">Empty 8-GPU Server</span>
        <span class="badge badge-cyan">cap = 8G</span>
      </div>
      <div class="space-y-1 text-xs mono">
        <div class="flex justify-between"><span class="text-slate-400">Original G:</span><span class="text-sky-300 font-semibold">${eg.server_8g.G_orig}</span></div>
        <div class="flex justify-between"><span class="text-slate-400">F_C (Truncated):</span><span class="text-amber-300 font-semibold">${eg.server_8g.G_fc}</span></div>
        <div class="flex justify-between"><span class="text-slate-400">F_C (Renormed):</span><span class="text-purple-300 font-semibold">${eg.server_8g.G_fc_norm}</span></div>
      </div>
    </div>`;

  if (CHARTS.n2) CHARTS.n2.destroy();
  const t4 = data.table_4g;
  CHARTS.n2 = new Chart(document.getElementById("n2-chart"), {
    type: "line",
    data: {
      labels: t4.map(r => `r = ${r.free} free`),
      datasets: [
        { label: "Original F(r) [ignores cap]", data: t4.map(r => r.F_orig), borderColor: "#38bdf8", tension: 0.2, borderWidth: 2.5 },
        { label: "Capacity-Aware F_C(r) [4G server]", data: t4.map(r => r.F_C), borderColor: "#f59e0b", tension: 0.2, borderWidth: 2.5 },
        { label: "Renormalised F_C_norm(r) [4G server]", data: t4.map(r => r.F_C_norm), borderColor: "#c084fc", tension: 0.2, borderWidth: 2.5 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: "#94a3b8", font: { size: 11 } }, grid: { color: "rgba(148,163,184,0.06)" } },
        y: { min: 0, ticks: { color: "#94a3b8", font: { size: 11 } }, grid: { color: "rgba(148,163,184,0.08)" } },
      },
    },
  });
}

async function runLiveE5() {
  const btn = document.getElementById("btn-run-e5");
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader" class="w-4 h-4 animate-spin"></i> Running E5 across 8 configurations...`;
  lucide.createIcons();
  try {
    const data = await apiPost("/api/experiments/run_e5", {
      trace_id: document.getElementById("e5-trace").value,
      n_jobs: parseInt(document.getElementById("e5-njobs").value || "600", 10),
      eta: parseFloat(document.getElementById("e5-eta").value || "15"),
    });
    renderLiveE5(data);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="flask-conical" class="w-4 h-4"></i> Execute Live E5 Experiment`;
    lucide.createIcons();
  }
}

function renderLiveE5(data) {
  const rows = data.rows;
  if (CHARTS.e5Live) CHARTS.e5Live.destroy();
  CHARTS.e5Live = new Chart(document.getElementById("e5-live-chart"), {
    type: "bar",
    data: {
      labels: rows.map(r => `${r.scheduler} (${r.topology === "venus15" ? "homo" : "mixed"})`),
      datasets: [
        {
          label: "Avg Wait (s)",
          data: rows.map(r => r.avg_wait),
          backgroundColor: rows.map(r =>
            r.topology === "venus15" ? "#38bdf8" :
            r.scheduler.includes("fc") ? "#f59e0b" : "#10b981"
          ),
          borderRadius: 5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { color: "rgba(148,163,184,0.06)" } },
        y: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { color: "rgba(148,163,184,0.08)" } },
      },
    },
  });

  document.getElementById("e5-live-table").innerHTML = `<table class="data-table">
    <thead><tr><th>Group</th><th>Policy</th><th>Avg Wait</th><th>8-GPU Wait</th><th>LB σ²</th><th>Unalloc</th></tr></thead>
    <tbody>
      ${rows.map(r => `<tr>
        <td class="text-slate-400 text-[11px]">${r.group}</td>
        <td class="mono font-semibold text-white">${r.scheduler}</td>
        <td class="mono">${r.avg_wait.toLocaleString()}s</td>
        <td class="mono">${r.avg_wait_8gpu.toLocaleString()}s</td>
        <td class="mono">${r.avg_lb_index.toFixed(5)}</td>
        <td class="mono">${r.unalloc_ratio.toFixed(1)}%</td>
      </tr>`).join("")}
    </tbody>
  </table>`;
}

// ---------------------------------------------------------------------------
// TAB 4: Full-Trace Saved Research Results
// ---------------------------------------------------------------------------
async function loadSavedResearch() {
  SAVED_RESEARCH = await apiGet("/api/experiments/saved");
  renderResearchTab();
}

function switchResearchSub(sub) {
  ACTIVE_RESEARCH_SUB = sub;
  document.querySelectorAll("[data-res]").forEach(b => b.classList.toggle("active", b.dataset.res === sub));
  renderResearchTab();
}

function renderResearchTab() {
  const container = document.getElementById("research-content");
  if (!SAVED_RESEARCH) {
    container.innerHTML = `<div class="panel p-8 text-center text-slate-400">Loading saved experiment datasets...</div>`;
    return;
  }

  if (ACTIVE_RESEARCH_SUB === "e5" && SAVED_RESEARCH.e5) {
    const e5 = SAVED_RESEARCH.e5;
    const renderBlock = (title, homoKey, mixedKey, fcKey, chartId) => {
      const homo = e5[homoKey] || {};
      const mixed = e5[mixedKey] || {};
      const fc = e5[fcKey] || {};
      return `<div class="panel p-4 space-y-4">
        <div class="flex flex-wrap justify-between items-center gap-2 border-b border-slate-800 pb-2">
          <h3 class="text-sm font-bold text-white">${title}</h3>
          <span class="badge badge-cyan">6,941 Venus Jobs · 1,080 GPUs</span>
        </div>
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 items-center">
          <div class="h-[240px]"><canvas id="${chartId}"></canvas></div>
          <div class="overflow-x-auto">
            <table class="data-table">
              <thead><tr><th>Setting</th><th>Avg Wait (s)</th><th>P99 Wait (s)</th><th>8-GPU Wait</th><th>LB σ²</th></tr></thead>
              <tbody>
                ${["fcfs", "consolidate", "fgd", "mcg"].map(s => homo[s] ? `<tr>
                  <td><span class="badge badge-cyan">Homo</span> <span class="mono font-semibold">${s}</span></td>
                  <td class="mono">${Math.round(homo[s].avg_wait).toLocaleString()}</td>
                  <td class="mono">${Math.round(homo[s].p99_wait).toLocaleString()}</td>
                  <td class="mono">${Math.round(homo[s].avg_wait_8gpu).toLocaleString()}</td>
                  <td class="mono">${homo[s].avg_lb_index.toFixed(5)}</td>
                </tr>` : "").join("")}
                ${["fcfs", "consolidate", "fgd", "mcg"].map(s => mixed[s] ? `<tr>
                  <td><span class="badge badge-emerald">Mixed</span> <span class="mono font-semibold">${s}</span></td>
                  <td class="mono">${Math.round(mixed[s].avg_wait).toLocaleString()}</td>
                  <td class="mono">${Math.round(mixed[s].p99_wait).toLocaleString()}</td>
                  <td class="mono">${Math.round(mixed[s].avg_wait_8gpu).toLocaleString()}</td>
                  <td class="mono">${mixed[s].avg_lb_index.toFixed(5)}</td>
                </tr>` : "").join("")}
                ${["mcg_fc", "mcg_fc_norm"].map(s => fc[s] ? `<tr>
                  <td><span class="badge badge-amber">Mixed F_C</span> <span class="mono font-semibold">${s}</span></td>
                  <td class="mono text-amber-300">${Math.round(fc[s].avg_wait).toLocaleString()}</td>
                  <td class="mono">${Math.round(fc[s].p99_wait).toLocaleString()}</td>
                  <td class="mono">${Math.round(fc[s].avg_wait_8gpu).toLocaleString()}</td>
                  <td class="mono">${fc[s].avg_lb_index.toFixed(5)}</td>
                </tr>` : "").join("")}
              </tbody>
            </table>
          </div>
        </div>
      </div>`;
    };

    container.innerHTML =
      renderBlock("Venus η = 15: Homogeneous 8-GPU vs Mixed 4+8-GPU & F_C", "venus_eta15_homo", "venus_eta15_mixed", "venus_eta15_mixed_fc", "res-e5-15") +
      renderBlock("Venus η = 10: Homogeneous 8-GPU vs Mixed 4+8-GPU & F_C", "venus_eta10_homo", "venus_eta10_mixed", "venus_eta10_mixed_fc", "res-e5-10");

    const drawE5Chart = (id, homo, mixed, fc) => {
      const labels = ["mcg (homo)", "mcg (mixed)", "mcg_fc (mixed)", "mcg_fc_norm (mixed)", "fcfs (mixed)"];
      const vals = [
        homo.mcg?.avg_wait || 0,
        mixed.mcg?.avg_wait || 0,
        fc.mcg_fc?.avg_wait || 0,
        fc.mcg_fc_norm?.avg_wait || 0,
        mixed.fcfs?.avg_wait || 0,
      ];
      new Chart(document.getElementById(id), {
        type: "bar",
        data: {
          labels,
          datasets: [{
            label: "Average Wait (s)",
            data: vals,
            backgroundColor: ["#38bdf8", "#10b981", "#f59e0b", "#a855f7", "#f43f5e"],
            borderRadius: 6,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { color: "rgba(148,163,184,0.06)" } },
            y: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { color: "rgba(148,163,184,0.08)" } },
          },
        },
      });
    };
    drawE5Chart("res-e5-15", e5.venus_eta15_homo || {}, e5.venus_eta15_mixed || {}, e5.venus_eta15_mixed_fc || {});
    drawE5Chart("res-e5-10", e5.venus_eta10_homo || {}, e5.venus_eta10_mixed || {}, e5.venus_eta10_mixed_fc || {});
  } else if (ACTIVE_RESEARCH_SUB === "e4" && SAVED_RESEARCH.e4) {
    const e4 = SAVED_RESEARCH.e4.filter(r => r.trace === "venus" && r.eta === 15);
    container.innerHTML = `<div class="panel p-4 space-y-4">
      <div class="flex justify-between items-center border-b border-slate-800 pb-2">
        <h3 class="text-sm font-bold text-white">E4: Fairness Protection (Venus η = 15, W_ref = 9,638s)</h3>
        <span class="badge badge-amber">Aging + Starvation Guard + Drain Reservation</span>
      </div>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 items-center">
        <div class="h-[260px]"><canvas id="res-e4-chart"></canvas></div>
        <div class="overflow-x-auto max-h-[300px]">
          <table class="data-table">
            <thead><tr><th>Run Label</th><th>Avg Wait</th><th>P99 Wait</th><th>8-GPU Wait</th><th>Starved (&gt;3×)</th></tr></thead>
            <tbody>
              ${e4.slice(0, 14).map(r => `<tr>
                <td class="mono font-semibold text-white">${r.label}</td>
                <td class="mono">${Math.round(r.avg_wait).toLocaleString()}s</td>
                <td class="mono text-sky-300">${Math.round(r.p99_wait).toLocaleString()}s</td>
                <td class="mono text-emerald-300">${Math.round(r.avg_wait_8gpu).toLocaleString()}s</td>
                <td class="mono">${r.starved_3x}</td>
              </tr>`).join("")}
            </tbody>
          </table>
        </div>
      </div>
    </div>`;
    const coreE4 = e4.slice(0, 8);
    new Chart(document.getElementById("res-e4-chart"), {
      type: "bar",
      data: {
        labels: coreE4.map(r => r.label),
        datasets: [
          { label: "Avg Wait (s)", data: coreE4.map(r => r.avg_wait), backgroundColor: "#38bdf8", borderRadius: 4 },
          { label: "P99 Wait (s)", data: coreE4.map(r => r.p99_wait), backgroundColor: "#f59e0b", borderRadius: 4 },
          { label: "8-GPU Wait (s)", data: coreE4.map(r => r.avg_wait_8gpu), backgroundColor: "#10b981", borderRadius: 4 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 } } } },
        scales: {
          x: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { color: "rgba(148,163,184,0.06)" } },
          y: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { color: "rgba(148,163,184,0.08)" } },
        },
      },
    });
  } else if (ACTIVE_RESEARCH_SUB === "e7" && SAVED_RESEARCH.e7_norm) {
    const scaleRows = SAVED_RESEARCH.e7_scale || [];
    const normRows = SAVED_RESEARCH.e7_norm.filter(r => r.window === "dynamic");
    const rawPts = normRows.filter(r => !r.normalize);
    const normPts = normRows.filter(r => r.normalize);

    container.innerHTML = `<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div class="panel p-4 space-y-3">
        <h3 class="text-sm font-bold text-white border-b border-slate-800 pb-2">Empirical Score Scale Gap (|G| vs σ²)</h3>
        <table class="data-table">
          <thead><tr><th>Trace / η</th><th>Sample Set</th><th>Median |G|</th><th>Median σ²</th><th>Ratio</th></tr></thead>
          <tbody>
            ${scaleRows.map(r => ["all_candidates", "chosen_placements"].map(k => `<tr>
              <td class="mono">${r.trace} η=${r.eta}</td>
              <td class="text-slate-300">${k}</td>
              <td class="mono text-sky-300">${r[k].median_abs_G.toFixed(4)}</td>
              <td class="mono text-indigo-300">${r[k].median_sigma2.toFixed(5)}</td>
              <td class="mono font-bold text-amber-300">${r[k].ratio.toFixed(1)}×</td>
            </tr>`).join("")).join("")}
          </tbody>
        </table>
      </div>
      <div class="panel p-4 space-y-3">
        <h3 class="text-sm font-bold text-white border-b border-slate-800 pb-2">Weight w Sweep: Raw vs Min-Max Normalised (Venus η=10)</h3>
        <div class="h-[240px]"><canvas id="res-e7-chart"></canvas></div>
      </div>
    </div>`;

    new Chart(document.getElementById("res-e7-chart"), {
      type: "line",
      data: {
        labels: rawPts.map(r => `w=${r.w}`),
        datasets: [
          { label: "Raw Score Avg Wait (s)", data: rawPts.map(r => r.avg_wait), borderColor: "#38bdf8", tension: 0.2, borderWidth: 2.5 },
          { label: "Normalised Score Avg Wait (s)", data: normPts.map(r => r.avg_wait), borderColor: "#c084fc", tension: 0.2, borderWidth: 2.5 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 } } } },
        scales: {
          x: { ticks: { color: "#94a3b8", font: { size: 11 } }, grid: { color: "rgba(148,163,184,0.06)" } },
          y: { ticks: { color: "#94a3b8", font: { size: 11 } }, grid: { color: "rgba(148,163,184,0.08)" } },
        },
      },
    });
  }
}
