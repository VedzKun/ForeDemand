/**
 * ForeDemand Dashboard – app.js
 * Loads data.json and renders all Plotly charts + tables
 */

"use strict";

// ── Plotly default layout overrides ─────────────────────────────────────────
const LAYOUT_BASE = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "rgba(255,255,255,0.02)",
  font: { family: "Inter, sans-serif", color: "#8a94a6", size: 12 },
  margin: { t: 30, r: 24, b: 50, l: 70 },
  xaxis: {
    gridcolor: "rgba(255,255,255,0.05)",
    linecolor: "rgba(255,255,255,0.08)",
    tickfont: { color: "#8a94a6" },
    showgrid: true,
  },
  yaxis: {
    gridcolor: "rgba(255,255,255,0.05)",
    linecolor: "rgba(255,255,255,0.08)",
    tickfont: { color: "#8a94a6" },
    showgrid: true,
    rangemode: "tozero",
  },
  legend: {
    bgcolor: "rgba(0,0,0,0.4)",
    bordercolor: "rgba(255,255,255,0.08)",
    borderwidth: 1,
    font: { size: 12 },
  },
  hoverlabel: {
    bgcolor: "#182030",
    bordercolor: "rgba(255,255,255,0.15)",
    font: { family: "Inter, sans-serif", size: 13 },
  },
};

const PLOTLY_CONFIG = {
  displayModeBar: true,
  modeBarButtonsToRemove: ["pan2d", "lasso2d", "select2d", "toImage"],
  displaylogo: false,
  responsive: true,
};

// Model colour palette
const MODEL_COLORS = {
  "Seasonal Naive":    "#fbbf24",
  "ETS":               "#a78bfa",
  "STL-ETS (Hybrid)": "#4f9eff",
};

// ── Utils ────────────────────────────────────────────────────────────────────
function fmt(n, dec = 0) {
  if (n == null) return "—";
  return Number(n).toLocaleString("en-IN", {
    minimumFractionDigits: dec,
    maximumFractionDigits: dec,
  });
}

function fmtDate(str) {
  const d = new Date(str);
  return d.toLocaleDateString("en-IN", { month: "short", year: "numeric" });
}

// ── Data loading ─────────────────────────────────────────────────────────────
async function loadData() {
  const resp = await fetch("data.json");
  if (!resp.ok) throw new Error(`HTTP ${resp.status} – could not load data.json`);
  return resp.json();
}

// ── Hero stats ────────────────────────────────────────────────────────────────
function renderHeroStats(data) {
  const series = data.historical;
  const last = series[series.length - 1];
  const yearAgo = series[series.length - 13] || series[0];
  const yoy = ((last.transactions_mn / yearAgo.transactions_mn - 1) * 100).toFixed(1);

  const items = [
    { val: fmt(last.transactions_mn, 0) + " Mn", label: "Latest Month Volume" },
    { val: data.meta.coverage_target,              label: "Conformal Target" },
    { val: series.length + " pts",                label: "Monthly Observations" },
    { val: "+" + yoy + "%",                       label: "YoY Growth" },
  ];

  const container = document.getElementById("hero-stats");
  container.innerHTML = items
    .map(
      ({ val, label }) => `
    <div class="hero-stat">
      <div class="hero-stat-val">${val}</div>
      <div class="hero-stat-label">${label}</div>
    </div>`
    )
    .join("");
}

// ── Section 1: Historical chart ───────────────────────────────────────────────
function renderHistorical(data) {
  const hist = data.historical;
  const dates = hist.map((d) => d.date);
  const vals  = hist.map((d) => d.transactions_mn);

  // Gradient-fill area
  const trace = {
    x: dates,
    y: vals,
    type: "scatter",
    mode: "lines",
    name: "Monthly UPI Transactions",
    line: { color: "#4f9eff", width: 2.5, shape: "spline" },
    fill: "tozeroy",
    fillcolor: "rgba(79,158,255,0.08)",
    hovertemplate: "<b>%{x}</b><br>%{y:.0f} Mn transactions<extra></extra>",
  };

  const layout = {
    ...LAYOUT_BASE,
    title: { text: "India UPI Monthly Transaction Volume (Mn)", font: { color: "#e8edf5", size: 14 } },
    yaxis: { ...LAYOUT_BASE.yaxis, title: { text: "Transactions (Mn)", font: { color: "#8a94a6" } } },
    shapes: [
      // COVID dip annotation line
      {
        type: "line", x0: "2020-03-01", x1: "2020-03-01",
        y0: 0, y1: 1, yref: "paper",
        line: { color: "rgba(251,191,36,0.4)", width: 1, dash: "dot" },
      },
    ],
    annotations: [
      {
        x: "2020-03-01", y: 0.85, yref: "paper",
        text: "COVID-19", showarrow: false,
        font: { color: "rgba(251,191,36,0.7)", size: 11 },
        xshift: 32,
      },
    ],
  };

  Plotly.newPlot("chart-historical", [trace], layout, PLOTLY_CONFIG);

  // KPI row
  renderKPIs(data);
}

function renderKPIs(data) {
  const series = data.historical;
  const vals = series.map((d) => d.transactions_mn);
  const last = vals[vals.length - 1];
  const peak = Math.max(...vals);
  const firstYear = series.filter((d) => d.date.startsWith("2017"));
  const firstAvg = firstYear.reduce((s, d) => s + d.transactions_mn, 0) / firstYear.length;
  const cagr = ((last / firstAvg) ** (1 / 7) - 1) * 100;

  const kpis = [
    { label: "Current Volume", val: fmt(last, 0) + " Mn", sub: fmtDate(series[series.length - 1].date) },
    { label: "All-Time High", val: fmt(peak, 0) + " Mn", sub: "Peak month" },
    { label: "CAGR (7-yr)", val: cagr.toFixed(1) + "%", sub: "Since Jan 2017" },
    { label: "Total Observations", val: series.length, sub: "Monthly data points" },
  ];

  const container = document.getElementById("kpi-row");
  container.innerHTML = kpis
    .map(
      ({ label, val, sub }) => `
    <div class="kpi-card">
      <div class="kpi-label">${label}</div>
      <div class="kpi-value">${val}</div>
      <div class="kpi-sub">${sub}</div>
    </div>`
    )
    .join("");
}

// ── Section 2: Forecast chart ────────────────────────────────────────────────
let currentModel = "STL-ETS (Hybrid)";

function renderForecast(data, modelName) {
  currentModel = modelName;
  const hist = data.train_series;
  const test = data.test_series;
  const fc   = data.forecasts[modelName];
  const color = MODEL_COLORS[modelName] || "#4f9eff";

  const traces = [
    {
      x: hist.map((d) => d.date),
      y: hist.map((d) => d.transactions_mn),
      type: "scatter", mode: "lines", name: "Historical (train)",
      line: { color: "#4f9eff", width: 2 },
      hovertemplate: "<b>%{x}</b><br>%{y:.0f} Mn<extra>Historical</extra>",
    },
    {
      x: test.map((d) => d.date),
      y: test.map((d) => d.transactions_mn),
      type: "scatter", mode: "lines+markers", name: "Actual (test)",
      line: { color: "#2dd4bf", width: 2, dash: "dot" },
      marker: { size: 6, color: "#2dd4bf" },
      hovertemplate: "<b>%{x}</b><br>%{y:.0f} Mn<extra>Actual</extra>",
    },
    // Conformal band (upper then lower reversed for fill)
    {
      x: [...fc.map((d) => d.date), ...fc.map((d) => d.date).reverse()],
      y: [...fc.map((d) => d.upper), ...fc.map((d) => d.lower).reverse()],
      type: "scatter", mode: "lines", name: "90% Conformal PI",
      fill: "toself",
      fillcolor: color.replace(")", ", 0.12)").replace("rgb", "rgba").replace("#", "rgba(") + ")",
      line: { color: "transparent" },
      showlegend: true,
      hoverinfo: "skip",
      // Simpler rgba fill
      fillcolor: hexToRgba(color, 0.12),
    },
    {
      x: fc.map((d) => d.date),
      y: fc.map((d) => d.forecast),
      type: "scatter", mode: "lines+markers", name: modelName + " Forecast",
      line: { color, width: 2.5 },
      marker: { size: 7, color },
      hovertemplate: "<b>%{x}</b><br>Forecast: %{y:.0f} Mn<extra>" + modelName + "</extra>",
    },
  ];

  const layout = {
    ...LAYOUT_BASE,
    title: { text: modelName + " – 12-Month Forecast with 90% Conformal PI", font: { color: "#e8edf5", size: 14 } },
    yaxis: { ...LAYOUT_BASE.yaxis, title: { text: "Transactions (Mn)", font: { color: "#8a94a6" } } },
    shapes: [
      {
        type: "rect",
        x0: test[0]?.date, x1: fc[fc.length - 1]?.date,
        y0: 0, y1: 1, yref: "paper",
        fillcolor: "rgba(255,255,255,0.02)", line: { width: 0 },
        layer: "below",
      },
      {
        type: "line",
        x0: test[0]?.date, x1: test[0]?.date,
        y0: 0, y1: 1, yref: "paper",
        line: { color: "rgba(255,255,255,0.15)", width: 1, dash: "dash" },
      },
    ],
    annotations: [
      {
        x: test[0]?.date, y: 0.97, yref: "paper",
        text: "Forecast start", showarrow: false,
        font: { color: "rgba(255,255,255,0.4)", size: 10 },
        xshift: 48,
      },
    ],
  };

  Plotly.react("chart-forecast", traces, layout, PLOTLY_CONFIG);
  renderForecastTable(fc);
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function renderForecastTable(fc) {
  const tbody = document.getElementById("forecast-tbody");
  tbody.innerHTML = fc
    .map((row) => {
      const width = ((row.upper || 0) - (row.lower || 0)).toFixed(0);
      return `<tr>
        <td>${fmtDate(row.date)}</td>
        <td class="mono text-blue">${fmt(row.forecast, 0)}</td>
        <td class="mono">${fmt(row.lower, 0)}</td>
        <td class="mono">${fmt(row.upper, 0)}</td>
        <td class="mono text-teal">±${fmt(width / 2, 0)}</td>
      </tr>`;
    })
    .join("");
}

// ── Section 3: Backtest ──────────────────────────────────────────────────────
function renderBacktest(data) {
  const rows = data.backtest_summary;
  if (!rows || rows.length === 0) {
    document.getElementById("backtest-tbody").innerHTML =
      '<tr><td colspan="10" style="text-align:center;padding:40px;color:var(--text-muted)">No backtest data – re-run pipeline</td></tr>';
    return;
  }

  const METRICS = ["MAE", "RMSE", "sMAPE (%)", "MASE", "Coverage", "Target", "Δ Coverage", "Mean Width", "Winkler ↓"];

  // Find best values for highlighting
  const numericCols = {
    "MAE": (r) => parseFloat(r["MAE"]),
    "RMSE": (r) => parseFloat(r["RMSE"]),
    "sMAPE (%)": (r) => parseFloat(r["sMAPE (%)"]),
    "MASE": (r) => parseFloat(r["MASE"]),
    "Winkler ↓": (r) => parseFloat(r["Winkler ↓"]),
  };

  const bests = {};
  Object.keys(numericCols).forEach((col) => {
    bests[col] = Math.min(...rows.map((r) => numericCols[col](r)));
  });

  const tbody = document.getElementById("backtest-tbody");
  tbody.innerHTML = rows
    .map((row) => {
      const delta = row["Δ Coverage"] || "";
      const deltaNum = parseFloat(delta);
      const deltaClass = Math.abs(deltaNum) <= 5 ? "good" : deltaNum > 5 ? "text-violet" : "bad";
      const color = MODEL_COLORS[row["Model"]] || "#fff";

      return `<tr>
        <td style="font-weight:600;color:${color}">${row["Model"]}</td>
        <td class="${parseFloat(row["MAE"]) === bests["MAE"] ? "best" : ""}">${row["MAE"]}</td>
        <td class="${parseFloat(row["RMSE"]) === bests["RMSE"] ? "best" : ""}">${row["RMSE"]}</td>
        <td class="${parseFloat(row["sMAPE (%)"]) === bests["sMAPE (%)"] ? "best" : ""}">${row["sMAPE (%)"]}</td>
        <td class="${parseFloat(row["MASE"]) === bests["MASE"] ? "best" : ""}">${row["MASE"]}</td>
        <td class="mono">${row["Coverage"]}</td>
        <td class="text-muted">${row["Target"]}</td>
        <td class="mono ${deltaClass}">${row["d.Coverage"]}</td>
        <td>${row["Mean Width"]}</td>
        <td class="${parseFloat(row["Winkler ↓"]) === bests["Winkler ↓"] ? "best" : ""}">${row["Winkler ↓"]}</td>
      </tr>`;
    })
    .join("");

  renderBacktestErrorChart(data);
}

function renderBacktestErrorChart(data) {
  const rows = data.backtest_summary;
  if (!rows || rows.length === 0) return;

  const models = rows.map((r) => r["Model"]);
  const maes   = rows.map((r) => parseFloat(r["MAE"]));
  const mases  = rows.map((r) => parseFloat(r["MASE"]));
  const winks  = rows.map((r) => parseFloat(r["Winkler ↓"]));

  const traces = [
    {
      x: models, y: maes, type: "bar", name: "MAE",
      marker: { color: models.map((m) => MODEL_COLORS[m] || "#4f9eff") },
      hovertemplate: "%{x}<br>MAE: %{y:.1f} Mn<extra></extra>",
      yaxis: "y",
    },
    {
      x: models, y: mases, type: "scatter", mode: "lines+markers",
      name: "MASE", marker: { size: 10, color: "#2dd4bf" },
      line: { color: "#2dd4bf", width: 2 },
      yaxis: "y2",
      hovertemplate: "%{x}<br>MASE: %{y:.3f}<extra></extra>",
    },
  ];

  const layout = {
    ...LAYOUT_BASE,
    title: { text: "Model Accuracy Comparison", font: { color: "#e8edf5", size: 14 } },
    barmode: "group",
    yaxis: {
      ...LAYOUT_BASE.yaxis,
      title: { text: "MAE (Mn transactions)", font: { color: "#8a94a6" } },
    },
    yaxis2: {
      overlaying: "y", side: "right",
      title: { text: "MASE (< 1 beats naive)", font: { color: "#2dd4bf" } },
      gridcolor: "transparent",
      tickfont: { color: "#2dd4bf" },
      rangemode: "tozero",
      showgrid: false,
    },
    legend: { ...LAYOUT_BASE.legend },
    shapes: [
      {
        type: "line", x0: -0.5, x1: models.length - 0.5,
        y0: 1, y1: 1, yref: "y2",
        line: { color: "rgba(251,191,36,0.5)", width: 1.5, dash: "dash" },
      },
    ],
    annotations: [
      {
        x: 0.98, y: 1, xref: "paper", yref: "y2",
        text: "MASE = 1 (baseline)", showarrow: false,
        font: { color: "rgba(251,191,36,0.7)", size: 10 },
      },
    ],
  };

  Plotly.newPlot("chart-backtest-error", traces, layout, PLOTLY_CONFIG);
}

// ── Section 4: Coverage ────────────────────────────────────────────────────
function renderCoverage(data) {
  const cov = data.coverage_check;
  if (!cov) return;

  const grid = document.getElementById("coverage-grid");
  grid.innerHTML = Object.entries(cov)
    .map(([model, info]) => {
      const pct = (info.empirical_coverage * 100).toFixed(1);
      const target = 90;
      const delta = (info.empirical_coverage * 100 - target).toFixed(1);
      const deltaSign = delta >= 0 ? "+" : "";

      let statusClass, labelClass, labelText;
      if (info.calibrated) {
        statusClass = "calibrated";
        labelClass = "calibrated";
        labelText = "Calibrated";
      } else if (info.empirical_coverage * 100 > target) {
        statusClass = "over";
        labelClass = "conservative";
        labelText = "Conservative";
      } else {
        statusClass = "under";
        labelClass = "overconfident";
        labelText = "Overconfident";
      }

      const barColor =
        statusClass === "calibrated"
          ? "linear-gradient(90deg, #2dd4bf, #4f9eff)"
          : statusClass === "over"
          ? "linear-gradient(90deg, #a78bfa, #4f9eff)"
          : "linear-gradient(90deg, #fb7185, #fbbf24)";

      const fillWidth = Math.min(100, info.empirical_coverage * 100).toFixed(1);
      const targetLeft = (target).toFixed(1);
      const deltaClass2 = delta >= 0 ? "pos" : "neg";

      return `
      <div class="cov-card ${statusClass}">
        <div class="cov-model">${model}</div>
        <div class="cov-value ${statusClass}">${pct}%</div>
        <div class="cov-target">Empirical coverage</div>
        <div class="cov-delta ${deltaClass2}">${deltaSign}${delta}pp vs. 90% target</div>
        <div class="cov-bar-wrap">
          <div class="cov-bar-track">
            <div class="cov-bar-fill" style="width:${fillWidth}%;background:${barColor}"></div>
            <div class="cov-bar-target" style="left:${targetLeft}%"></div>
          </div>
        </div>
        <div>
          <span class="cov-label ${labelClass}">${labelText}</span>
        </div>
        <div style="margin-top:12px;font-size:0.78rem;color:var(--text-muted)">
          ${info.n_covered} / ${info.n_total} observations covered
        </div>
      </div>`;
    })
    .join("");

  renderCoverageChart(data);
}

function renderCoverageChart(data) {
  const cov = data.coverage_check;
  const models = Object.keys(cov);
  const empirical = models.map((m) => cov[m].empirical_coverage * 100);
  const target = 90;

  const traces = [
    {
      x: models,
      y: empirical,
      type: "bar",
      name: "Empirical Coverage",
      marker: {
        color: models.map((m) =>
          cov[m].calibrated ? "#2dd4bf" : empirical[models.indexOf(m)] > target ? "#a78bfa" : "#fb7185"
        ),
        opacity: 0.85,
      },
      text: empirical.map((v) => v.toFixed(1) + "%"),
      textposition: "outside",
      textfont: { color: "#e8edf5", size: 13 },
      hovertemplate: "%{x}<br>Coverage: %{y:.1f}%<extra></extra>",
    },
    {
      x: models,
      y: Array(models.length).fill(target),
      type: "scatter",
      mode: "lines",
      name: "90% Target",
      line: { color: "#fbbf24", width: 2, dash: "dash" },
      hoverinfo: "skip",
    },
  ];

  const layout = {
    ...LAYOUT_BASE,
    title: { text: "Empirical vs Target Coverage (90%)", font: { color: "#e8edf5", size: 14 } },
    yaxis: {
      ...LAYOUT_BASE.yaxis,
      title: { text: "Coverage (%)", font: { color: "#8a94a6" } },
      range: [0, 115],
      ticksuffix: "%",
    },
    showlegend: true,
    bargap: 0.35,
  };

  Plotly.newPlot("chart-coverage", traces, layout, PLOTLY_CONFIG);
}

// ── Navigation ────────────────────────────────────────────────────────────────
function initNav() {
  const navbar = document.getElementById("navbar");
  const sections = document.querySelectorAll(".section[id]");
  const links = document.querySelectorAll(".nav-link");

  window.addEventListener("scroll", () => {
    navbar.classList.toggle("scrolled", window.scrollY > 20);

    let current = "";
    sections.forEach((sec) => {
      if (window.scrollY >= sec.offsetTop - 120) current = sec.id;
    });
    links.forEach((l) => {
      l.classList.toggle("active", l.dataset.section === current);
    });
  });

  // Smooth scroll for hero CTA
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    a.addEventListener("click", (e) => {
      const target = document.querySelector(a.getAttribute("href"));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: "smooth" });
      }
    });
  });

  // Section fade-in
  const io = new IntersectionObserver(
    (entries) => entries.forEach((e) => e.target.classList.toggle("visible", e.isIntersecting)),
    { threshold: 0.08 }
  );
  sections.forEach((s) => io.observe(s));
}

// ── Model selector buttons ────────────────────────────────────────────────────
function initModelSelector(data) {
  document.querySelectorAll(".ctrl-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".ctrl-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderForecast(data, btn.dataset.model);
    });
  });
}

// ── Footer ────────────────────────────────────────────────────────────────────
function renderFooter(data) {
  const el = document.getElementById("footer-generated");
  if (el && data.meta?.generated_at) {
    const d = new Date(data.meta.generated_at);
    el.textContent = `Results generated: ${d.toLocaleString("en-IN")}`;
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  initNav();

  try {
    const data = await loadData();

    renderHeroStats(data);
    renderHistorical(data);
    renderForecast(data, currentModel);
    renderBacktest(data);
    renderCoverage(data);
    renderFooter(data);
    initModelSelector(data);

    // Hide loader
    const loader = document.getElementById("loader");
    loader.classList.add("hidden");

    // Make first section visible immediately
    document.querySelector("#overview")?.classList.add("visible");
  } catch (err) {
    console.error("ForeDemand error:", err);
    document.getElementById("loader").innerHTML = `
      <div class="loader-inner">
        <p style="color:#fb7185;font-size:1rem">⚠ Could not load data.json</p>
        <p style="color:#8a94a6;font-size:0.8rem;margin-top:8px">Run <code>python run_forecast.py</code> first</p>
      </div>`;
  }
}

document.addEventListener("DOMContentLoaded", main);
