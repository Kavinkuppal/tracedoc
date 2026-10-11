/* tracedoc report app. Plain JS, no dependencies besides the inlined ECharts.
 *
 * aggregate() mirrors tracedoc/report/data.py:aggregate(); tests/test_report.py runs both on
 * the same rows and compares. Keep them in step.
 */
"use strict";

const SEVERITIES = ["high", "medium", "low"];
const STATUSES = ["completed", "no_answer", "error", "recursion_limit", "interrupted"];
const CONTEXT_GROWTH_MAX_CALLS = 50;

function cmp(a, b) {
  if (a === b) return 0;
  if (a === null || a === undefined) return 1;
  if (b === null || b === undefined) return -1;
  return a < b ? -1 : 1;
}
function cmpList(a, b) {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const c = cmp(a[i], b[i]);
    if (c) return c;
  }
  return a.length - b.length;
}
function r4(x, digits = 4) {
  if (x === null || x === undefined) return null;
  const f = Math.pow(10, digits);
  return Math.round(x * f) / f;
}
function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
const num = (a, b) => a - b;
function counter() {
  const m = new Map();
  m.add = (k, n = 1) => m.set(k, (m.get(k) || 0) + n);
  return m;
}
function setMap() {
  const m = new Map();
  m.add = (k, v) => {
    if (!m.has(k)) m.set(k, new Set());
    m.get(k).add(v);
  };
  m.of = (k) => m.get(k) || new Set();
  return m;
}
const intersectSize = (a, b) => [...a].filter((x) => b.has(x)).length;

function select(rows, task, node, sev) {
  const sevRuns = sev ? new Set(rows.findings.filter((f) => f.severity === sev).map((f) => f.run)) : null;
  const runs = rows.runs.filter(
    (r) =>
      (task == null || r.task === task) &&
      (node == null || r.nodes.includes(node)) &&
      (sevRuns == null || sevRuns.has(r.id))
  );
  const ids = new Set(runs.map((r) => r.id));
  const findings = rows.findings.filter(
    (f) => ids.has(f.run) && (node == null || f.node === node) && (sev == null || f.severity === sev)
  );
  const steps = rows.steps.filter((s) => ids.has(s.run));
  return [runs, findings, steps];
}

function aggregate(rows, task = null, node = null, sev = null) {
  const [runs, findings, allSteps] = select(rows, task, node, sev);
  const steps = allSteps.filter((s) => node == null || s.node === node);
  const n = runs.length;

  // 1. scorecard
  const costs = runs.filter((r) => r.cost !== null).map((r) => r.cost).sort(num);
  const durs = runs.filter((r) => r.dur !== null).map((r) => r.dur).sort(num);
  const bySev = {};
  for (const s of SEVERITIES) bySev[s] = findings.filter((f) => f.severity === s).length;
  const scorecard = {
    runs: n,
    completion_rate: n ? r4(runs.filter((r) => r.status === "completed").length / n) : 0,
    median_steps: r4(quantile(runs.map((r) => r.steps).sort(num), 0.5)),
    total_cost: costs.length ? r4(costs.reduce((a, b) => a + b, 0), 6) : null,
    median_cost: costs.length ? r4(quantile(costs, 0.5), 6) : null,
    median_latency_s: durs.length ? r4(quantile(durs, 0.5)) : null,
    tokens: runs.reduce((a, r) => a + r.tin + r.tout, 0),
    findings_by_severity: bySev,
  };

  // 3. failure frequency
  const byCheck = new Map();
  for (const f of findings) {
    if (!byCheck.has(f.check)) byCheck.set(f.check, { runs: new Set(), severity: "low" });
    const e = byCheck.get(f.check);
    e.runs.add(f.run);
    if (SEVERITIES.indexOf(f.severity) < SEVERITIES.indexOf(e.severity)) e.severity = f.severity;
  }
  const frequency = [...byCheck.entries()]
    .map(([c, e]) => ({ check: c, runs: e.runs.size, severity: e.severity }))
    .sort((a, b) =>
      cmpList([-a.runs, SEVERITIES.indexOf(a.severity), a.check], [-b.runs, SEVERITIES.indexOf(b.severity), b.check])
    );

  // 4. failures by task type
  const taskOf = new Map(runs.map((r) => [r.id, r.task]));
  const cells = setMap();
  for (const f of findings) cells.add(taskOf.get(f.run) + "\u0000" + f.check, f.run);
  const tasks = [...new Set(runs.map((r) => r.task))].sort(cmp);
  const runsPerTask = counter();
  for (const r of runs) runsPerTask.add(r.task);
  const byTask = {
    tasks,
    checks: frequency.map((x) => x.check),
    cells: [...cells.entries()]
      .map(([k, v]) => {
        const [t, c] = k.split("\u0000");
        return [t, c, v.size, r4(v.size / runsPerTask.get(t))];
      })
      .sort(cmpList),
  };

  // 5. how runs end
  const statusByTask = {};
  for (const t of tasks) {
    statusByTask[t] = {};
    for (const s of STATUSES) statusByTask[t][s] = 0;
  }
  for (const r of runs) statusByTask[r.task][r.status] += 1;

  // 6. agent flow map
  const nodeSteps = new Map();
  for (const s of allSteps) {
    if (s.kind === "node" && s.parent === null) {
      if (!nodeSteps.has(s.run)) nodeSteps.set(s.run, []);
      nodeSteps.get(s.run).push(s);
    }
  }
  const statusOf = new Map(runs.map((r) => [r.id, r.status]));
  const edges = counter();
  const visits = counter();
  const nodeRuns = setMap();
  for (const [runId, seq] of nodeSteps) {
    seq.sort((a, b) => cmpList([a.t === null, a.t], [b.t === null, b.t]));
    const path = ["__start__", ...seq.map((s) => s.node)];
    if (statusOf.get(runId) === "completed") path.push("__end__");
    for (let i = 0; i + 1 < path.length; i++) edges.add(path[i] + "\u0000" + path[i + 1]);
    for (const s of seq) {
      visits.add(s.node);
      nodeRuns.add(s.node, runId);
    }
  }
  const findingsInSet = rows.findings.filter((f) => taskOf.has(f.run));
  const failedAt = setMap();
  for (const f of findingsInSet) if (f.node && (sev == null || f.severity === sev)) failedAt.add(f.node, f.run);
  const flow = {
    nodes: [...visits.keys()]
      .map((name) => [name, visits.get(name), r4(intersectSize(failedAt.of(name), nodeRuns.of(name)) / nodeRuns.of(name).size)])
      .sort(cmpList),
    edges: [...edges.entries()].map(([k, c]) => [...k.split("\u0000"), c]).sort(cmpList),
  };

  // 7. handoffs
  const pairCount = counter();
  const pairRuns = setMap();
  for (const s of allSteps) {
    if (s.kind === "handoff") {
      const k = s.from + "\u0000" + s.to;
      pairCount.add(k);
      pairRuns.add(k, s.run);
    }
  }
  const pairFailed = setMap();
  for (const f of findingsInSet) {
    if (f.key.includes("->") && (sev == null || f.severity === sev)) {
      const i = f.key.indexOf("->");
      pairFailed.add(f.key.slice(0, i) + "\u0000" + f.key.slice(i + 2), f.run);
    }
  }
  const handoffs = [...pairCount.entries()]
    .map(([k, c]) => {
      const [a, b] = k.split("\u0000");
      return [a, b, c, r4(intersectSize(pairFailed.of(k), pairRuns.of(k)) / pairRuns.of(k).size)];
    })
    .sort(cmpList);

  // 8. cost by node
  const tok = counter();
  const cost = new Map();
  for (const s of steps) {
    if (s.kind === "llm" && s.node) {
      tok.add(s.node, (s.tin || 0) + (s.tout || 0));
      cost.set(s.node, (cost.get(s.node) || 0) + (s.cost || 0));
    }
  }
  const tokTotal = [...tok.values()].reduce((a, b) => a + b, 0);
  const costTotal = [...cost.values()].reduce((a, b) => a + b, 0);
  const costByNode = [...tok.keys()]
    .map((k) => ({
      node: k,
      tokens: tok.get(k),
      token_share: tokTotal ? r4(tok.get(k) / tokTotal) : 0,
      cost: r4(cost.get(k), 6),
      cost_share: costTotal ? r4(cost.get(k) / costTotal) : null,
    }))
    .sort((a, b) => cmpList([-a.tokens, a.node], [-b.tokens, b.node]));

  // 9. context growth
  const perRun = new Map();
  const llmSorted = [...steps].sort((a, b) => cmpList([a.run, a.t === null, a.t], [b.run, b.t === null, b.t]));
  for (const s of llmSorted) {
    if (s.kind === "llm" && s.tin !== null) {
      if (!perRun.has(s.run)) perRun.set(s.run, []);
      perRun.get(s.run).push(s.tin);
    }
  }
  const lists = [...perRun.values()];
  const length = Math.min(Math.max(0, ...lists.map((v) => v.length)), CONTEXT_GROWTH_MAX_CALLS);
  const medianLine = [];
  for (let i = 0; i < length; i++) {
    medianLine.push(r4(quantile(lists.filter((v) => v.length > i).map((v) => v[i]).sort(num), 0.5)));
  }
  let worstId = null;
  for (const [k, v] of perRun) {
    const peak = Math.max(...v);
    if (worstId === null || cmpList([peak, k], [Math.max(...perRun.get(worstId)), worstId]) > 0) worstId = k;
  }
  const context = {
    median: medianLine,
    worst_run: worstId,
    worst: worstId ? perRun.get(worstId).slice(0, CONTEXT_GROWTH_MAX_CALLS) : [],
  };

  // 10. steps per run
  const hist = counter();
  for (const r of runs) hist.add(r.steps);
  const stepsHist = [...hist.entries()].sort((a, b) => a[0] - b[0]);

  // 11. latency by node
  const durations = new Map();
  for (const s of steps) {
    if (s.kind === "node" && s.dur !== null && s.node) {
      if (!durations.has(s.node)) durations.set(s.node, []);
      durations.get(s.node).push(s.dur);
    }
  }
  const latency = [...durations.entries()]
    .map(([k, v]) => {
      const sorted = [...v].sort(num);
      return [k, [0, 0.25, 0.5, 0.75, 1].map((q) => r4(quantile(sorted, q))), v.length];
    })
    .sort((a, b) => cmp(a[0], b[0]));

  return {
    scorecard,
    failure_frequency: frequency,
    failures_by_task: byTask,
    status_by_task: statusByTask,
    flow,
    handoffs,
    cost_by_node: costByNode,
    context_growth: context,
    steps_per_run: stepsHist,
    latency_by_node: latency,
  };
}

if (typeof module !== "undefined") {
  module.exports = { aggregate, select, quantile };
}

/* ======================================================================== */
/* Browser UI                                                               */
/* ======================================================================== */

if (typeof document !== "undefined") {
  const DATA = JSON.parse(document.getElementById("report-data").textContent);
  const ROWS = DATA.rows;
  const REPORT = DATA.report;
  const TITLES = DATA.titles;
  const state = { task: null, node: null, sev: null, search: "", replayRun: null, highlight: new Set() };
  const votes = new Map();
  const votesMeta = new Map();
  const charts = new Map();

  const SEV_ICON = { high: "▲", medium: "◆", low: "●" };
  const SEV_LABEL = { high: "High", medium: "Medium", low: "Low" };
  const STATUS_ICON = { completed: "✓", no_answer: "∅", error: "✕", recursion_limit: "↻", interrupted: "❚❚" };
  const STATUS_LABEL = { completed: "Completed", no_answer: "No answer", error: "Error", recursion_limit: "Recursion limit", interrupted: "Interrupted" };

  const $ = (sel, root = document) => root.querySelector(sel);
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const title = (id) => TITLES[id] || id;
  const shortId = (id) => (id || "").slice(-8);
  const pct = (x) => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`);
  const money = (x) => (x === null || x === undefined ? "–" : x < 0.01 && x > 0 ? `$${x.toFixed(4)}` : `$${x.toFixed(2)}`);
  const secs = (x) =>
    x === null || x === undefined ? "–" : x === 0 ? "0 s" : x < 1 ? `${+(x * 1000).toPrecision(2)} ms` : `${+x.toPrecision(3)} s`;
  const fmtInt = (x) => (x ?? 0).toLocaleString();
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const nodeLabel = (n) => (n === "__start__" ? "START" : n === "__end__" ? "END" : n);

  function sevColor(s) {
    return css({ high: "--sev-high", medium: "--sev-medium", low: "--sev-low" }[s]);
  }
  function statusColor(s) {
    return css({ completed: "--status-good", no_answer: "--status-critical", error: "--status-critical", recursion_limit: "--status-serious", interrupted: "--status-warning" }[s]);
  }
  function seqColor(t) {
    // single-hue sequential ramp (blue), light -> dark; t in [0, 1]
    const ramp = css("--seq-ramp").split(",").map((s) => s.trim());
    const i = Math.max(0, Math.min(ramp.length - 1, Math.round(t * (ramp.length - 1))));
    return ramp[i];
  }

  function onColor(hex) {
    // dark or light ink, whichever reads better on this fill (WCAG relative luminance)
    const c = hex.replace("#", "");
    const [r, g, b] = [0, 2, 4].map((i) => {
      const v = parseInt(c.slice(i, i + 2), 16) / 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 ? "#0b0b0b" : "#ffffff";
  }

  function baseOption() {
    return {
      animation: false,
      textStyle: { fontFamily: css("--font"), color: css("--text-secondary") },
      tooltip: {
        confine: true,
        backgroundColor: css("--surface-raised"),
        borderColor: css("--border"),
        textStyle: { color: css("--text-primary"), fontSize: 12 },
      },
      grid: { left: 8, right: 24, top: 16, bottom: 8, containLabel: true },
    };
  }
  function axisStyle(extra = {}) {
    return {
      axisLine: { lineStyle: { color: css("--axis") } },
      axisTick: { show: false },
      axisLabel: { color: css("--text-muted"), fontSize: 11 },
      splitLine: { lineStyle: { color: css("--grid"), type: "solid" } },
      ...extra,
    };
  }

  function chart(id, option, onClick) {
    const el = document.getElementById(id);
    if (!el) return;
    let c = charts.get(id);
    if (!c) {
      c = echarts.init(el, null, { renderer: "svg" });
      charts.set(id, c);
    }
    c.clear();
    c.setOption({ ...baseOption(), ...option });
    c.off("click");
    if (onClick) c.on("click", onClick);
  }

  function table(id, headers, rows) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML =
      `<table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>` +
      rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("") +
      `</tbody></table>`;
  }

  function empty(id, message) {
    const c = charts.get(id);
    if (c) c.clear();
    const el = document.getElementById(id);
    if (el) el.setAttribute("data-empty", message);
  }
  function filled(id) {
    const el = document.getElementById(id);
    if (el) el.removeAttribute("data-empty");
  }

  /* ---- filters --------------------------------------------------------- */

  function initFilters() {
    const tasks = [...new Set(ROWS.runs.map((r) => r.task))].sort();
    const nodes = [...new Set(ROWS.runs.flatMap((r) => r.nodes))].sort();
    const fill = (sel, values, label) => {
      sel.innerHTML = `<option value="">All ${label}</option>` + values.map((v) => `<option value="${esc(v)}">${esc(label === "severities" ? SEV_LABEL[v] : v)}</option>`).join("");
    };
    fill($("#f-task"), tasks, "task types");
    fill($("#f-node"), nodes, "nodes");
    fill($("#f-sev"), SEVERITIES, "severities");
    for (const [sel, key] of [["#f-task", "task"], ["#f-node", "node"], ["#f-sev", "sev"]]) {
      $(sel).addEventListener("change", (e) => {
        state[key] = e.target.value || null;
        render();
      });
    }
    $("#f-reset").addEventListener("click", () => {
      state.task = state.node = state.sev = null;
      for (const s of ["#f-task", "#f-node", "#f-sev"]) $(s).value = "";
      render();
    });
  }

  /* ---- sections -------------------------------------------------------- */

  function renderScorecard(sc) {
    const tiles = [
      ["Runs", fmtInt(sc.runs), ""],
      ["Completion rate", pct(sc.completion_rate), ""],
      ["Median steps", sc.median_steps ?? "–", "graph node executions per run"],
      ["Total cost", money(sc.total_cost), sc.median_cost !== null ? `median ${money(sc.median_cost)} per run` : "no prices for these models"],
      ["Median latency", secs(sc.median_latency_s), ""],
    ];
    const sevs = SEVERITIES.map(
      (s) => `<span class="sev sev-${s}"><span class="icon">${SEV_ICON[s]}</span> ${sc.findings_by_severity[s]} ${SEV_LABEL[s].toLowerCase()}</span>`
    ).join("");
    $("#scorecard").innerHTML =
      tiles.map(([label, value, sub]) => `<div class="tile"><div class="tile-label">${label}</div><div class="tile-value">${value}</div>${sub ? `<div class="tile-sub">${esc(sub)}</div>` : ""}</div>`).join("") +
      `<div class="tile"><div class="tile-label">Findings</div><div class="tile-sevs">${sevs}</div></div>`;
  }

  function evidenceLink(runId, stepIds, label) {
    return `<a href="#replay" class="evidence" data-run="${esc(runId)}" data-steps="${esc((stepIds || []).join(","))}">${esc(label)}</a>`;
  }

  function renderTopFixes() {
    const groups = REPORT.groups.slice(0, 3);
    if (!groups.length) {
      $("#top-fixes").innerHTML = `<p class="muted">No problems found.</p>`;
      return;
    }
    $("#top-fixes").innerHTML = groups
      .map((g, i) => {
        const f = g.representative;
        const where = [g.node && `node ${g.node}`, g.key && g.key !== g.node && g.key].filter(Boolean).join(" · ");
        return `<article class="card fix">
          <div class="fix-rank">${i + 1}</div>
          <div class="fix-body">
            <div class="fix-head"><span class="sev sev-${f.severity}"><span class="icon">${SEV_ICON[f.severity]}</span> ${SEV_LABEL[f.severity]}</span>
              <h3>${esc(title(g.check_id))}</h3>${f.needs_confirmation ? `<span class="badge">unconfirmed</span>` : ""}</div>
            <div class="muted small">${esc(g.check_id)}${where ? " · " + esc(where) : ""} · ${g.runs_affected} run${g.runs_affected === 1 ? "" : "s"} (${pct(g.frequency)})${g.impact.tokens ? ` · ${fmtInt(g.impact.tokens)} tokens in these runs` : ""}</div>
            <p>${esc(f.explanation)}</p>
            <p class="fix-text"><strong>Suggested fix:</strong> ${esc(f.suggested_fix)}</p>
            ${f.evidence_excerpts.length ? `<blockquote>${f.evidence_excerpts.map(esc).join("<br>")}</blockquote>` : ""}
            <div>${evidenceLink(f.run_id, f.evidence_step_ids, "Open evidence in run replay →")}</div>
          </div></article>`;
      })
      .join("");
  }

  function renderFrequency(agg) {
    const rows = agg.failure_frequency.slice(0, 20).reverse();
    if (!rows.length) return empty("c-frequency", "No findings for this filter.");
    filled("c-frequency");
    chart(
      "c-frequency",
      {
        xAxis: { type: "value", minInterval: 1, ...axisStyle() },
        yAxis: {
          type: "category",
          data: rows.map((r) => `${SEV_ICON[r.severity]} ${title(r.check)}`),
          ...axisStyle({ axisLabel: { color: css("--text-secondary"), fontSize: 12 } }),
        },
        tooltip: { ...baseOption().tooltip, formatter: (p) => `${esc(rows[p.dataIndex].check)}<br>${SEV_ICON[rows[p.dataIndex].severity]} ${SEV_LABEL[rows[p.dataIndex].severity]}<br><b>${p.value}</b> runs affected` },
        series: [
          {
            type: "bar",
            barMaxWidth: 18,
            data: rows.map((r) => ({ value: r.runs, itemStyle: { color: sevColor(r.severity), borderRadius: [0, 4, 4, 0] } })),
            label: { show: true, position: "right", color: css("--text-secondary"), fontSize: 11 },
          },
        ],
      },
      (p) => {
        state.search = rows[p.dataIndex].check;
        $("#search").value = state.search;
        renderFindings();
        document.getElementById("findings").scrollIntoView({ behavior: "smooth" });
      }
    );
    $("#c-frequency").style.height = `${Math.max(160, rows.length * 30 + 40)}px`;
    charts.get("c-frequency").resize();
    table("t-frequency", ["Check", "Severity", "Runs affected"], agg.failure_frequency.map((r) => [title(r.check) + ` (${r.check})`, SEV_LABEL[r.severity], r.runs]));
  }

  function renderByTask(agg) {
    const { tasks, checks, cells } = agg.failures_by_task;
    if (!cells.length) return empty("c-bytask", "No findings for this filter.");
    filled("c-bytask");
    const max = Math.max(...cells.map((c) => c[3]));
    chart("c-bytask", {
      grid: { left: 8, right: 24, top: 8, bottom: 8, containLabel: true },
      xAxis: { type: "category", data: checks.map(title), ...axisStyle({ axisLabel: { color: css("--text-muted"), fontSize: 11, rotate: 30, interval: 0 } }), splitArea: { show: false } },
      yAxis: { type: "category", data: tasks, ...axisStyle() },
      tooltip: { ...baseOption().tooltip, formatter: (p) => `${esc(p.data.task)} × ${esc(p.data.check)}<br><b>${p.data.runs}</b> runs (${pct(p.data.share)} of ${esc(p.data.task)} runs)` },
      series: [
        {
          type: "heatmap",
          data: cells.map(([t, c, runs, share]) => ({
            value: [checks.indexOf(c), tasks.indexOf(t), share],
            task: t, check: c, runs, share,
            itemStyle: { color: seqColor(max ? share / max : 0), borderColor: css("--surface"), borderWidth: 2, borderRadius: 4 },
            label: { color: onColor(seqColor(max ? share / max : 0)) },
          })),
          label: { show: true, formatter: (p) => pct(p.data.share), fontSize: 11 },
        },
      ],
    });
    table("t-bytask", ["Task type", "Check", "Runs affected", "Share of task's runs"], cells.map(([t, c, runs, share]) => [t, c, runs, pct(share)]));
  }

  function renderStatus(agg) {
    const tasks = Object.keys(agg.status_by_task);
    if (!tasks.length) return empty("c-status", "No runs for this filter.");
    filled("c-status");
    const present = STATUSES.filter((s) => tasks.some((t) => agg.status_by_task[t][s]));
    chart("c-status", {
      legend: { top: 0, left: 0, textStyle: { color: css("--text-secondary") }, data: present.map((s) => `${STATUS_ICON[s]} ${STATUS_LABEL[s]}`) },
      grid: { left: 8, right: 24, top: 36, bottom: 8, containLabel: true },
      xAxis: { type: "value", minInterval: 1, ...axisStyle() },
      yAxis: { type: "category", data: tasks, ...axisStyle({ axisLabel: { color: css("--text-secondary"), fontSize: 12 } }) },
      tooltip: { ...baseOption().tooltip, trigger: "axis", axisPointer: { type: "shadow" } },
      series: present.map((s) => ({
        name: `${STATUS_ICON[s]} ${STATUS_LABEL[s]}`,
        type: "bar",
        stack: "status",
        barMaxWidth: 22,
        itemStyle: { color: statusColor(s), borderColor: css("--surface"), borderWidth: 2 },
        label: { show: true, formatter: (p) => (p.value ? `${STATUS_ICON[s]} ${p.value}` : ""), color: onColor(statusColor(s)), fontSize: 11 },
        labelLayout: { hideOverlap: true },
        data: tasks.map((t) => agg.status_by_task[t][s] || 0),
      })),
    });
    table("t-status", ["Task type", ...STATUSES.map((s) => STATUS_LABEL[s])], tasks.map((t) => [t, ...STATUSES.map((s) => agg.status_by_task[t][s])]));
  }

  function renderFlow(agg) {
    const { nodes, edges } = agg.flow;
    if (!nodes.length) return empty("c-flow", "No runs for this filter.");
    filled("c-flow");
    const maxVisits = Math.max(...nodes.map((n) => n[1]));
    const maxEdge = Math.max(1, ...edges.map((e) => e[2]));
    const names = ["__start__", ...nodes.map((n) => n[0]), ...(edges.some((e) => e[1] === "__end__") ? ["__end__"] : [])];
    const info = new Map(nodes.map(([n, v, f]) => [n, { visits: v, failure: f }]));
    // Layered layout: column = shortest distance from START; END goes last. Deterministic.
    const out = new Map();
    for (const [a, b] of edges) {
      if (!out.has(a)) out.set(a, []);
      out.get(a).push(b);
    }
    const depth = new Map([["__start__", 0]]);
    const queue = ["__start__"];
    while (queue.length) {
      const cur = queue.shift();
      for (const next of (out.get(cur) || []).slice().sort(cmp)) {
        if (!depth.has(next) && next !== "__end__") {
          depth.set(next, depth.get(cur) + 1);
          queue.push(next);
        }
      }
    }
    const maxDepth = Math.max(0, ...[...depth.values()]);
    for (const name of names) if (!depth.has(name)) depth.set(name, name === "__end__" ? maxDepth + 1 : maxDepth);
    const columns = new Map();
    for (const name of names) {
      const d = depth.get(name);
      if (!columns.has(d)) columns.set(d, []);
      columns.get(d).push(name);
    }
    const pos = new Map();
    for (const [d, col] of columns) col.forEach((name, i) => pos.set(name, [d * 160, (i - (col.length - 1) / 2) * 80]));
    chart("c-flow", {
      tooltip: {
        ...baseOption().tooltip,
        formatter: (p) =>
          p.dataType === "edge"
            ? `${esc(nodeLabel(p.data.source))} → ${esc(nodeLabel(p.data.target))}<br><b>${p.data.value}</b> transitions`
            : info.has(p.data.name)
              ? `${esc(p.data.name)}<br>${info.get(p.data.name).visits} visits<br>failure rate ${pct(info.get(p.data.name).failure)}`
              : nodeLabel(p.data.name),
      },
      series: [
        {
          type: "graph",
          layout: "none",
          roam: true,
          left: 40, right: 90, top: 30, bottom: 30,
          edgeSymbol: ["none", "arrow"],
          edgeSymbolSize: 8,
          label: { show: true, position: "bottom", distance: 6, color: css("--text-primary"), fontSize: 12, formatter: (p) => nodeLabel(p.data.name) },
          lineStyle: { curveness: 0.25, opacity: 0.6, color: css("--text-muted") },
          emphasis: { focus: "adjacency" },
          data: names.map((name) => {
            const n = info.get(name);
            return {
              name,
              x: pos.get(name)[0],
              y: pos.get(name)[1],
              symbolSize: n ? 14 + 26 * Math.sqrt(n.visits / maxVisits) : 12,
              symbol: n ? "circle" : "roundRect",
              itemStyle: n
                ? { color: seqColor(n.failure), borderColor: css("--surface"), borderWidth: 2 }
                : { color: css("--axis") },
            };
          }),
          links: edges.map(([a, b, c]) => ({ source: a, target: b, value: c, lineStyle: { width: 1 + 7 * (c / maxEdge) } })),
        },
      ],
    });
    table("t-flow", ["From", "To", "Transitions"], edges.map(([a, b, c]) => [nodeLabel(a), nodeLabel(b), c]));
    table("t-flow-nodes", ["Node", "Visits", "Failure rate"], nodes.map(([n, v, f]) => [n, v, pct(f)]));
  }

  function renderHandoffs(agg) {
    const rows = agg.handoffs;
    if (!rows.length) return empty("c-handoffs", "No handoffs between agents for this filter.");
    filled("c-handoffs");
    const senders = [...new Set(rows.map((r) => r[0]))].sort();
    const receivers = [...new Set(rows.map((r) => r[1]))].sort();
    chart("c-handoffs", {
      grid: { left: 8, right: 24, top: 8, bottom: 24, containLabel: true },
      xAxis: { type: "category", name: "receiving agent", nameLocation: "middle", nameGap: 28, nameTextStyle: { color: css("--text-muted") }, data: receivers, ...axisStyle() },
      yAxis: { type: "category", name: "sending agent", nameTextStyle: { color: css("--text-muted") }, data: senders, ...axisStyle() },
      tooltip: { ...baseOption().tooltip, formatter: (p) => `${esc(p.data.from)} → ${esc(p.data.to)}<br><b>${p.data.count}</b> handoffs<br>${pct(p.data.fail)} of runs had a finding on this handoff` },
      series: [
        {
          type: "heatmap",
          data: rows.map(([a, b, c, f]) => ({
            value: [receivers.indexOf(b), senders.indexOf(a), c],
            from: a, to: b, count: c, fail: f,
            itemStyle: { color: seqColor(0.15 + 0.85 * f), borderColor: css("--surface"), borderWidth: 2, borderRadius: 4 },
            label: { color: onColor(seqColor(0.15 + 0.85 * f)) },
          })),
          label: { show: true, fontSize: 11, formatter: (p) => `${p.data.count}${p.data.fail ? ` · ${pct(p.data.fail)} failed` : ""}` },
        },
      ],
    });
    table("t-handoffs", ["From", "To", "Handoffs", "Share of runs with a finding"], rows.map(([a, b, c, f]) => [a, b, c, pct(f)]));
  }

  function renderCost(agg) {
    const rows = [...agg.cost_by_node].reverse();
    if (!rows.length) return empty("c-cost", "No model calls for this filter.");
    filled("c-cost");
    const hasCost = rows.some((r) => r.cost_share !== null);
    const series = [{ name: "Token share", key: "token_share", color: css("--series-1") }];
    if (hasCost) series.push({ name: "Cost share", key: "cost_share", color: css("--series-2") });
    chart("c-cost", {
      legend: series.length > 1 ? { top: 0, left: 0, textStyle: { color: css("--text-secondary") } } : undefined,
      grid: { left: 8, right: 40, top: series.length > 1 ? 32 : 8, bottom: 8, containLabel: true },
      xAxis: { type: "value", max: 1, axisLabel: { formatter: (v) => pct(v), color: css("--text-muted") }, ...axisStyle({ axisLabel: { formatter: (v) => pct(v), color: css("--text-muted"), fontSize: 11 } }) },
      yAxis: { type: "category", data: rows.map((r) => r.node), ...axisStyle({ axisLabel: { color: css("--text-secondary"), fontSize: 12 } }) },
      tooltip: { ...baseOption().tooltip, trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v) => pct(v) },
      series: series.map((s) => ({
        name: s.name,
        type: "bar",
        barMaxWidth: 14,
        barGap: "20%",
        itemStyle: { color: s.color, borderRadius: [0, 4, 4, 0] },
        label: { show: true, position: "right", formatter: (p) => pct(p.value), color: css("--text-secondary"), fontSize: 11 },
        data: rows.map((r) => r[s.key]),
      })),
    });
    $("#c-cost").style.height = `${Math.max(160, rows.length * (hasCost ? 44 : 30) + 60)}px`;
    charts.get("c-cost").resize();
    table("t-cost", ["Node", "Tokens", "Token share", "Cost", "Cost share"], agg.cost_by_node.map((r) => [r.node, r.tokens, pct(r.token_share), money(r.cost), pct(r.cost_share)]));
  }

  function renderContext(agg) {
    const { median, worst, worst_run } = agg.context_growth;
    if (!median.length) return empty("c-context", "No model calls for this filter.");
    filled("c-context");
    const x = median.map((_, i) => i + 1);
    const line = (name, data, color) => ({
      name, type: "line", data, showSymbol: false, symbolSize: 8, lineStyle: { width: 2, color }, itemStyle: { color },
      endLabel: { show: true, formatter: name, color: css("--text-secondary"), fontSize: 11 },
    });
    chart("c-context", {
      legend: { top: 0, left: 0, textStyle: { color: css("--text-secondary") } },
      grid: { left: 8, right: 110, top: 60, bottom: 24, containLabel: true },
      xAxis: { type: "category", data: worst.length > x.length ? worst.map((_, i) => i + 1) : x, name: "model call #", nameLocation: "middle", nameGap: 26, nameTextStyle: { color: css("--text-muted") }, ...axisStyle(), boundaryGap: false },
      yAxis: { type: "value", name: "input tokens", nameTextStyle: { color: css("--text-muted") }, ...axisStyle() },
      tooltip: { ...baseOption().tooltip, trigger: "axis" },
      series: [line("Median run", median, css("--series-1")), line(`Worst run (${shortId(worst_run)})`, worst, css("--series-2"))],
    });
    table("t-context", ["Model call #", "Median input tokens", "Worst run input tokens"], x.map((i) => [i, median[i - 1], worst[i - 1] ?? "–"]));
  }

  function renderSteps(agg) {
    const rows = agg.steps_per_run;
    if (!rows.length) return empty("c-steps", "No runs for this filter.");
    filled("c-steps");
    const lo = rows[0][0], hi = rows[rows.length - 1][0];
    const counts = new Map(rows);
    const xs = [];
    for (let v = lo; v <= hi; v++) xs.push(v);
    chart("c-steps", {
      xAxis: { type: "category", data: xs, name: "steps", nameLocation: "middle", nameGap: 26, nameTextStyle: { color: css("--text-muted") }, ...axisStyle() },
      yAxis: { type: "value", minInterval: 1, name: "runs", nameTextStyle: { color: css("--text-muted") }, ...axisStyle() },
      grid: { left: 8, right: 24, top: 28, bottom: 24, containLabel: true },
      tooltip: { ...baseOption().tooltip, formatter: (p) => `${p.name} steps<br><b>${p.value}</b> runs` },
      series: [{ type: "bar", barCategoryGap: "8%", itemStyle: { color: css("--series-1"), borderRadius: [4, 4, 0, 0] }, data: xs.map((v) => counts.get(v) || 0) }],
    });
    table("t-steps", ["Steps", "Runs"], rows);
  }

  function renderLatency(agg) {
    const rows = agg.latency_by_node;
    if (!rows.length) return empty("c-latency", "No timed steps for this filter.");
    filled("c-latency");
    chart("c-latency", {
      grid: { left: 8, right: 24, top: 28, bottom: 8, containLabel: true },
      xAxis: { type: "category", data: rows.map((r) => r[0]), ...axisStyle({ axisLabel: { color: css("--text-secondary"), fontSize: 12, interval: 0, rotate: rows.length > 5 ? 30 : 0 } }) },
      yAxis: { type: "value", ...axisStyle({ axisLabel: { color: css("--text-muted"), fontSize: 11, formatter: (v) => secs(v) } }) },
      tooltip: { ...baseOption().tooltip, formatter: (p) => { const [lo, q1, med, q3, hi] = rows[p.dataIndex][1]; return `${esc(rows[p.dataIndex][0])} (${rows[p.dataIndex][2]} steps)<br>median ${secs(med)}<br>middle half ${secs(q1)} – ${secs(q3)}<br>range ${secs(lo)} – ${secs(hi)}`; } },
      series: [{ type: "boxplot", boxWidth: [8, 28], itemStyle: { color: css("--series-1-wash"), borderColor: css("--series-1"), borderWidth: 2 }, data: rows.map((r) => r[1]) }],
    });
    table("t-latency", ["Node", "Steps", "Min", "Q1", "Median", "Q3", "Max"], rows.map(([n, q, c]) => [n, c, ...q.map(secs)]));
  }

  /* ---- run replay -------------------------------------------------------- */

  function renderReplay(runs) {
    const sel = $("#replay-run");
    const ids = runs.map((r) => r.id);
    if (!ids.includes(state.replayRun)) state.replayRun = ids[0] || null;
    sel.innerHTML = runs
      .map((r) => `<option value="${esc(r.id)}" ${r.id === state.replayRun ? "selected" : ""}>${esc(shortId(r.id))} · ${esc(r.task)} · ${STATUS_ICON[r.status]} ${esc(STATUS_LABEL[r.status])} · ${r.steps} steps</option>`)
      .join("");
    const run = ROWS.runs.find((r) => r.id === state.replayRun);
    const box = $("#replay-body");
    if (!run) {
      box.innerHTML = `<p class="muted">No runs for this filter.</p>`;
      return;
    }
    const steps = ROWS.steps.filter((s) => s.run === run.id && s.kind !== "node");
    const flagged = new Map();
    for (const f of ROWS.findings.filter((f) => f.run === run.id)) {
      for (const id of f.evidence) {
        const prev = flagged.get(id);
        if (!prev || SEVERITIES.indexOf(f.severity) < SEVERITIES.indexOf(prev.severity)) flagged.set(id, f);
      }
    }
    // evidence that points at a node step flags the steps inside it
    for (const s of steps) if (!flagged.has(s.id) && flagged.has(s.parent)) flagged.set(s.id, flagged.get(s.parent));
    const total = run.dur || Math.max(0.001, ...steps.map((s) => (s.t || 0) + (s.dur || 0)));
    const kindLabel = { llm: "model", tool: "tool", handoff: "handoff" };
    box.innerHTML =
      `<div class="replay-meta"><div><span class="muted">Task</span> ${esc(run.input)}</div><div><span class="muted">Result</span> ${STATUS_ICON[run.status]} ${esc(STATUS_LABEL[run.status])}: ${esc(run.output || run.error)}</div>
       <div class="muted small">${fmtInt(run.tin + run.tout)} tokens · ${money(run.cost)} · ${secs(run.dur)} · run ${esc(run.id)}</div></div>` +
      `<ol class="timeline">` +
      steps
        .map((s, i) => {
          const f = flagged.get(s.id);
          const left = Math.min(100, (100 * (s.t || 0)) / total);
          const width = Math.max(0.6, Math.min(100 - left, (100 * (s.dur || 0)) / total));
          const name = s.kind === "handoff" ? `${s.from} → ${s.to}` : s.name;
          const hl = state.highlight.has(s.id) || state.highlight.has(s.parent) ? " highlight" : "";
          return `<li class="step${f ? " flagged sev-border-" + f.severity : ""}${hl}" id="step-${esc(s.id)}">
            <details ${hl ? "open" : ""}><summary>
              <span class="step-n">${i + 1}</span>
              <span class="step-kind kind-${s.kind}">${kindLabel[s.kind]}</span>
              <span class="step-name">${esc(name)}</span>
              <span class="step-node muted">${esc(s.node || "")}</span>
              ${s.err ? `<span class="sev sev-high"><span class="icon">✕</span> error</span>` : ""}
              ${f ? `<span class="sev sev-${f.severity}"><span class="icon">${SEV_ICON[f.severity]}</span> ${esc(title(f.check))}</span>` : ""}
              <span class="step-bar"><span style="left:${left}%;width:${width}%"></span></span>
              <span class="step-meta muted">${s.tin !== null ? fmtInt(s.tin) + " in · " + fmtInt(s.tout) + " out · " : ""}${secs(s.dur)}</span>
            </summary>
            <div class="io"><div><div class="io-label">${s.kind === "handoff" ? "Sender produced" : s.kind === "tool" ? "Arguments" : "Latest input"}</div><pre>${esc(s.in)}</pre></div>
            <div><div class="io-label">${s.kind === "handoff" ? "Receiver got" : s.err ? "Error" : s.kind === "tool" ? "Result" : "Output"}</div><pre>${esc(s.out)}</pre></div></div>
            ${f ? `<div class="io-finding"><strong>${esc(title(f.check))}:</strong> ${esc(f.explanation)}</div>` : ""}
            </details></li>`;
        })
        .join("") +
      `</ol>`;
  }

  function openReplay(runId, stepIds) {
    state.replayRun = runId;
    state.highlight = new Set(stepIds);
    const visible = select(ROWS, state.task, state.node, state.sev)[0].some((r) => r.id === runId);
    if (!visible) {
      state.task = state.node = state.sev = null;
      for (const s of ["#f-task", "#f-node", "#f-sev"]) $(s).value = "";
      render();
    } else {
      renderReplay(select(ROWS, state.task, state.node, state.sev)[0]);
    }
    const first = document.querySelector(".step.highlight");
    (first || document.getElementById("replay")).scrollIntoView({ behavior: "smooth", block: "center" });
  }

  /* ---- signals & findings ----------------------------------------------- */

  function renderSignals(runs) {
    const ids = new Set(runs.map((r) => r.id));
    const sigs = ROWS.signals.filter((s) => ids.has(s.run) && (state.node == null || s.node === state.node));
    if (!sigs.length) {
      $("#signals-body").innerHTML = `<p class="muted">No good-behavior signals for this filter.</p>`;
      return;
    }
    const byCheck = new Map();
    for (const s of sigs) {
      if (!byCheck.has(s.check)) byCheck.set(s.check, []);
      byCheck.get(s.check).push(s);
    }
    $("#signals-body").innerHTML = [...byCheck.entries()]
      .map(([check, list]) => `<article class="card signal"><h3><span class="good-icon">✓</span> ${esc(title(check))} <span class="muted small">${new Set(list.map((s) => s.run)).size} run(s)</span></h3>
        <ul>${list.slice(0, 5).map((s) => `<li>${esc(s.explanation)} ${evidenceLink(s.run, s.evidence, "evidence")}</li>`).join("")}</ul>
        ${list.length > 5 ? `<p class="muted small">and ${list.length - 5} more</p>` : ""}</article>`)
      .join("");
  }

  function renderFindings() {
    const [, findings] = select(ROWS, state.task, state.node, state.sev);
    const q = state.search.trim().toLowerCase();
    const rows = findings
      .filter((f) => !q || [f.check, title(f.check), f.node, f.key, f.explanation, f.run].join(" ").toLowerCase().includes(q))
      .sort((a, b) => cmpList([SEVERITIES.indexOf(a.severity), a.check, a.run], [SEVERITIES.indexOf(b.severity), b.check, b.run]));
    $("#findings-count").textContent = `${rows.length} finding${rows.length === 1 ? "" : "s"}`;
    $("#findings-body").innerHTML = rows
      .map((f) => {
        const v = votes.get(f.id);
        return `<tr>
          <td><span class="sev sev-${f.severity}"><span class="icon">${SEV_ICON[f.severity]}</span> ${SEV_LABEL[f.severity]}</span></td>
          <td><strong>${esc(title(f.check))}</strong><div class="muted small">${esc(f.check)} · ${f.method}${f.unconfirmed ? " · unconfirmed" : ""}</div></td>
          <td>${esc(f.node || "–")}</td>
          <td>${esc(f.explanation)}<div class="muted small">Fix: ${esc(f.fix)}</div></td>
          <td>${evidenceLink(f.run, f.evidence, shortId(f.run))}</td>
          <td class="vote"><button class="${v === "up" ? "on" : ""}" data-vote="up" data-id="${esc(f.id)}" data-check="${esc(f.check)}" aria-label="Useful">👍</button><button class="${v === "down" ? "on" : ""}" data-vote="down" data-id="${esc(f.id)}" data-check="${esc(f.check)}" aria-label="Not useful">👎</button></td>
        </tr>`;
      })
      .join("");
  }

  function recordVote(id, check, vote) {
    votes.set(id, vote);
    votesMeta.set(id, check);
    const cmd = `tracedoc feedback ${id} ${vote}`;
    $("#vote-cmd").textContent = cmd;
    $("#vote-panel").hidden = false;
    $("#vote-count").textContent = `${votes.size} vote${votes.size === 1 ? "" : "s"} this session`;
    renderFindings();
  }

  function exportVotes() {
    const body = {
      format: "tracedoc-feedback",
      version: 1,
      report_generated_at: REPORT.metadata.generated_at,
      votes: [...votes.entries()].map(([id, vote]) => ({ finding_id: id, check_id: votesMeta.get(id), vote })),
    };
    const blob = new Blob([JSON.stringify(body, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "tracedoc-feedback.json";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  /* ---- top level --------------------------------------------------------- */

  function render() {
    const agg = aggregate(ROWS, state.task, state.node, state.sev);
    const active = [state.task, state.node, state.sev && SEV_LABEL[state.sev]].filter(Boolean);
    $("#filter-note").textContent = active.length ? `Showing ${agg.scorecard.runs} of ${ROWS.runs.length} runs: ${active.join(" · ")}` : `All ${ROWS.runs.length} runs`;
    renderScorecard(agg.scorecard);
    renderFrequency(agg);
    renderByTask(agg);
    renderStatus(agg);
    renderFlow(agg);
    renderHandoffs(agg);
    renderCost(agg);
    renderContext(agg);
    renderSteps(agg);
    renderLatency(agg);
    const runs = select(ROWS, state.task, state.node, state.sev)[0];
    renderReplay(runs);
    renderSignals(runs);
    renderFindings();
  }

  function setTheme(theme) {
    if (theme) document.documentElement.setAttribute("data-theme", theme);
    else document.documentElement.removeAttribute("data-theme");
    render();
  }

  document.addEventListener("click", (e) => {
    const link = e.target.closest("a.evidence");
    if (link) {
      e.preventDefault();
      openReplay(link.dataset.run, link.dataset.steps ? link.dataset.steps.split(",") : []);
      return;
    }
    const btn = e.target.closest("button[data-vote]");
    if (btn) recordVote(btn.dataset.id, btn.dataset.check, btn.dataset.vote);
  });
  $("#replay-run").addEventListener("change", (e) => {
    state.replayRun = e.target.value;
    state.highlight = new Set();
    renderReplay(select(ROWS, state.task, state.node, state.sev)[0]);
  });
  $("#search").addEventListener("input", (e) => {
    state.search = e.target.value;
    renderFindings();
  });
  $("#vote-export").addEventListener("click", exportVotes);
  $("#vote-copy").addEventListener("click", () => navigator.clipboard && navigator.clipboard.writeText($("#vote-cmd").textContent));
  $("#theme").addEventListener("click", () => {
    const dark = document.documentElement.getAttribute("data-theme") === "dark" ||
      (!document.documentElement.hasAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
    setTheme(dark ? "light" : "dark");
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => render());
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => charts.forEach((c) => c.resize()), 100);
  });

  initFilters();
  renderTopFixes();
  render();
}
