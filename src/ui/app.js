/* Дашборд-монитор stell-llm: данные из GET /metrics (ADR-11 §4).
 * Статика без сборочных зависимостей; обновление каждые 1 с. */
"use strict";

const TIERS = ["VRAM", "pinned RAM", "NVMe"];

function fmtBytes(b) {
  if (b === null || b === undefined) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0, v = b;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 ? 2 : 0)} ${units[i]}`;
}

function pct(v) { return v == null ? "—" : v.toFixed(1); }

function text(el, s) { el.textContent = s; }

function render() {
  fetch("/metrics").then(r => r.json()).then(snap => {
    text(document.getElementById("status"),
      `состояние: ${snap.engine?.state ?? "?"} · модель: ${snap.model ?? "?"}`);
    renderTierMap(snap.tiers?.tier_bytes || []);
    renderEngine(snap);
    renderTierStats(snap.tiers || {});
    renderSlots(snap.engine || {});
    renderFeatures(snap.features, snap.build, snap.parallel);
    renderSpec(snap);
    renderLogs(snap.serve?.logs || []);
    renderLogits(snap.logits);
    text(document.getElementById("updated"),
      `обновлено: ${new Date().toLocaleTimeString()} · запросов: ${snap.serve?.requests_total ?? 0}` +
      ` · отмен: ${snap.serve?.cancelled_total ?? 0}`);
  }).catch(() => {
    text(document.getElementById("status"), "недоступен (нет /metrics)");
  });
}

function renderTierMap(tierBytes) {
  const host = document.getElementById("tier-map");
  const legend = document.getElementById("tier-legend");
  host.innerHTML = "";
  legend.innerHTML = "";
  const max = Math.max(1, ...tierBytes.map(b => b || 0));
  tierBytes.forEach((bytes, i) => {
    const row = document.createElement("div");
    row.className = "tier-row";
    const label = document.createElement("span");
    label.className = "tier-label";
    label.textContent = TIERS[i] || `tier ${i}`;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("div");
    fill.className = "fill tier-" + i;
    const w = Math.max(2, Math.round(100 * (bytes || 0) / max));
    fill.style.width = w + "%";
    const value = document.createElement("span");
    value.className = "tier-value";
    value.textContent = fmtBytes(bytes);
    bar.appendChild(fill);
    row.appendChild(label); row.appendChild(bar); row.appendChild(value);
    host.appendChild(row);
  });
  legend.innerHTML = TIERS.map((t, i) =>
    `<span class="chip tier-${i}">${t}</span>`).join(" ");
}

function renderEngine(snap) {
  const cards = document.getElementById("engine-cards");
  const e = snap.engine || {};
  const lat = snap.latency || {};
  cards.innerHTML = "";
  [["state", e.state], ["sessions_active", e.sessions_active],
   ["slots_busy", e.slots_busy], ["slots_total", e.slots_total],
   ["cancelled_total", e.cancelled_total],
   ["decode tok/s", lat.tok_s?.decode], ["prefill tok/s", lat.tok_s?.prefill]]
    .forEach(([k, v]) => {
      const c = document.createElement("div");
      c.className = "card";
      c.innerHTML = `<div class="k">${k}</div><div class="v">${v ?? "—"}</div>`;
      cards.appendChild(c);
    });
  const table = document.getElementById("latency-table");
  table.innerHTML = "<table><tr><th>метрика</th><th>p50</th><th>p95</th><th>p99</th></tr>"
    + Object.entries({ "TTFT (мс)": lat.ttft_ms, "TPOT (мс)": lat.tpot_ms })
      .map(([name, stat]) => `<tr><td>${name}</td><td>${pct(stat?.p50)}</td>
        <td>${pct(stat?.p95)}</td><td>${pct(stat?.p99)}</td></tr>`).join("")
    + "</table>";
}

function renderTierStats(tiers) {
  const pf = document.getElementById("prefetch");
  const rate = (tiers.prefetch?.hit_rate ?? 0) * 100;
  const target = 60; // порог PLAN.md §4: hit-rate ≥ 60%
  pf.innerHTML = `<div class="gauge"><div class="gauge-fill" style="width:${Math.min(100, rate)}%"></div></div>
    <div>hit-rate prefetch: ${rate.toFixed(1)}% (порог ${target}%
      ${rate >= target ? "✅" : "⚠️"})</div>`;
  const pauses = document.getElementById("pauses");
  const lp = tiers.layer_pause_ms || {};
  const p95ok = (lp.p95 ?? 99) <= 5; // порог PLAN.md §4: p95 ≤ 5 мс
  pauses.innerHTML = `<table><tr><th>паузы слоёв</th><th>p50</th><th>p95</th><th>p99</th></tr>
    <tr><td>мс ${p95ok ? "✅" : "⚠️"}</td><td>${pct(lp.p50)}</td>
    <td>${pct(lp.p95)}</td><td>${pct(lp.p99)}</td></tr></table>`;
}

function renderSlots(engine) {
  const host = document.getElementById("slots");
  const total = engine.slots_total || 0;
  const busy = engine.slots_busy || 0;
  let html = "";
  for (let i = 0; i < total; i++) {
    html += `<span class="slot ${i < busy ? "busy" : "free"}"></span>`;
  }
  host.innerHTML = html + `<div>${busy}/${total} слотов занято</div>`;
}

function renderFeatures(features, build, parallel) {
  const host = document.getElementById("features-cards");
  host.innerHTML = "";
  [["kv_quant", features?.kv_quant], ["spec_draft", features?.spec_draft],
   ["tp", parallel?.tp], ["ep", parallel?.ep], ["nccl", parallel?.nccl],
   ["version", build?.version], ["cuda", build?.cuda], ["sm", build?.sm]]
    .forEach(([k, v]) => {
      const c = document.createElement("div");
      c.className = "card";
      c.innerHTML = `<div class="k">${k}</div><div class="v">${v ?? "—"}</div>`;
      host.appendChild(c);
    });
}

// Спекулятивный контур (round 8, ADR-11 §4): состояние фичи, телеметрия
// драфтера (accept-rate, n_draft) и tok/s спек-декода. Честность: если
// движок ещё не публикует spec-поля — служебная пометка spec_note
// (прецедент dtype_orig round 6: отсутствие поля ≠ молчаливый ноль).
function renderSpec(snap) {
  const host = document.getElementById("spec-cards");
  if (!host) return;
  const spec = snap.spec || {};
  const tokS = snap.latency?.tok_s?.spec_decode;
  const enabled = snap.features?.spec_draft;
  const rows = [
    ["features.spec_draft", enabled === undefined ? "—" : String(enabled)],
    ["spec configured/executed", (snap.features?.spec_draft_configured === undefined ||
                                  snap.features?.spec_draft_executed === undefined)
      ? "—"
      : `${snap.features.spec_draft_configured}/${snap.features.spec_draft_executed}`],
    ["spec reason", snap.features?.spec_draft_reason ?? "—"],
    ["accept-rate", spec.accept_rate === undefined ? "—" : (spec.accept_rate * 100).toFixed(1) + "%"],
    ["n_draft", spec.n_draft ?? "—"],
    ["n_draft effective", spec.n_draft_effective ?? "—"],
    ["spec-decode tok/s", tokS === undefined ? "—" : tokS.toFixed(2)],
  ];
  host.innerHTML = rows.map(([k, v]) =>
    `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("")
    + (snap.spec_note ? `<div class="note">${snap.spec_note}</div>` : "");
}

function renderLogs(logs) {
  document.getElementById("log-view").textContent =
    logs.length ? logs.join("\n") : "(пусто)";
}

/* Статистика stell_logits в /metrics — СТРОГИЙ формат (ADR-11 §4, round 3,
 * [R5], планировщик [Q5]): ровно 4 поля-счётчика, payload (массивы логитов)
 * в /metrics не передаётся никогда. Рендер — строго эти 4 KV (лишние ключи
 * движка игнорируются — формат фиксирован ADR, не «каково придёт»); до
 * merge core-4 блока нет — секция честно показывает отсутствие данных,
 * не роняя UI. */
const LOGITS_FIELDS = [
  ["requests_total", "вызовов stell_logits"],
  ["errors_total", "ошибок stell_logits"],
  ["last_n_vocab", "vocab последнего вызова"],
  ["last_positions", "позиций последнего вызова"],
];

function renderLogits(logits) {
  const host = document.getElementById("logits-stats");
  host.innerHTML = "";
  if (!logits || typeof logits !== "object" || !Object.keys(logits).length) {
    host.innerHTML = "<div>нет данных (снапшот движка без logits-статистики — " +
      "поле появится после merge core-4 [R5])</div>";
    return;
  }
  const table = document.createElement("table");
  table.innerHTML = LOGITS_FIELDS
    .map(([k, label]) => `<tr><td>${label}</td><td>${logits[k] ?? "—"}</td></tr>`)
    .join("\n");
  host.appendChild(table);
}

render();
setInterval(render, 1000);
