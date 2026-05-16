const $ = (id) => document.getElementById(id);

let currentAnalysis = null;
let currentAiKind = "momentum";
let strategies = [];
let backtestLoaded = false;
let strategyLoaded = false;

const fmt = (value, digits = 2, suffix = "") => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${Number(value).toLocaleString("zh-TW", { minimumFractionDigits: digits, maximumFractionDigits: digits })}${suffix}`;
};

const fmtPct = (value, digits = 1) => fmt(Number(value) * 100, digits, "%");

const setStatus = (text) => {
  $("statusText").textContent = text;
};

const params = (extra = {}) => {
  const p = new URLSearchParams();
  const base = {
    ticker: $("tickerInput").value,
    mode: $("modeInput").value,
    start_date: $("startDateInput").value,
    end_date: $("endDateInput").value,
    capital: $("capitalInput").value,
    risk_pct: $("riskPctInput").value,
    ...extra,
  };
  Object.entries(base).forEach(([key, value]) => {
    if (value !== null && value !== undefined && String(value).trim() !== "") p.set(key, value);
  });
  return p.toString();
};

async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function renderTable(id, rows, columns = null) {
  const table = $(id);
  table.innerHTML = "";
  if (!rows || rows.length === 0) {
    table.innerHTML = "<tbody><tr><td>目前沒有資料</td></tr></tbody>";
    return;
  }
  const keys = columns || Object.keys(rows[0]).filter((key) => key !== "index");
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  keys.forEach((key) => {
    const th = document.createElement("th");
    th.textContent = key;
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    keys.forEach((key) => {
      const td = document.createElement("td");
      const value = row[key];
      td.textContent = typeof value === "number" ? fmt(value, Math.abs(value) >= 100 ? 1 : 2) : value ?? "--";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
}

function renderList(id, items, emptyText) {
  const list = $(id);
  list.innerHTML = "";
  (items?.length ? items : [emptyText]).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function chartScale(data, key, height, pad) {
  const values = data.map((row) => Number(row[key])).filter(Number.isFinite);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min || 1;
  return {
    min,
    max,
    y: (value) => height - pad - ((Number(value) - min) / spread) * (height - pad * 2),
  };
}

function renderLineChart(id, rows, valueKeys, colors) {
  const chart = $(id);
  const data = [...(rows || [])].filter((row) => row.date).sort((a, b) => String(a.date).localeCompare(String(b.date)));
  if (data.length < 2) {
    chart.innerHTML = "<p class='status'>資料不足，暫無法繪圖</p>";
    return;
  }
  if (window.Plotly) {
    const labels = {
      Close: "收盤價",
      BuyHold_Equity: "買進持有",
      Strategy_Equity: "策略淨值",
    };
    const traces = valueKeys.map((key, index) => ({
      x: data.map((row) => row.date),
      y: data.map((row) => row[key]),
      type: "scatter",
      mode: "lines",
      name: labels[key] || key,
      line: { color: colors[index], width: 2.5 },
      hovertemplate: `%{x}<br>${labels[key] || key}: %{y:.2f}<extra></extra>`,
    }));
    Plotly.react(chart, traces, {
      margin: { l: 48, r: 18, t: 12, b: 36 },
      paper_bgcolor: "#f8fafc",
      plot_bgcolor: "#ffffff",
      hovermode: "x unified",
      xaxis: { showgrid: false },
      yaxis: { gridcolor: "#e2e8f0", fixedrange: false },
      legend: { orientation: "h", y: 1.14 },
    }, { responsive: true, displaylogo: false });
    return;
  }
  const width = 920;
  const height = 300;
  const pad = 34;
  const allValues = data.flatMap((row) => valueKeys.map((key) => Number(row[key]))).filter(Number.isFinite);
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const spread = max - min || 1;
  const x = (i) => pad + (i / (data.length - 1)) * (width - pad * 2);
  const y = (v) => height - pad - ((Number(v) - min) / spread) * (height - pad * 2);
  const lines = valueKeys.map((key, i) => {
    const pts = data.map((row, idx) => `${x(idx).toFixed(1)},${y(row[key]).toFixed(1)}`).join(" ");
    return `<polyline points="${pts}" fill="none" stroke="${colors[i]}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />`;
  }).join("");
  chart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}">
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#cbd5e1" />
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#cbd5e1" />
      <text x="${pad}" y="24" fill="#64748b" font-size="12">${fmt(max, 2)}</text>
      <text x="${pad}" y="${height - 8}" fill="#64748b" font-size="12">${fmt(min, 2)}</text>
      ${lines}
    </svg>`;
}

function renderCandleChart(rows) {
  const chart = $("candleChart");
  const data = [...(rows || [])].filter((row) => row.date && row.Close !== null).sort((a, b) => String(a.date).localeCompare(String(b.date)));
  if (data.length < 20) {
    chart.innerHTML = "<p class='status'>資料不足，暫無法繪圖</p>";
    return;
  }
  if (window.Plotly) {
    const plot = data.slice(-180);
    const x = plot.map((row) => row.date);
    const upDown = plot.map((row) => Number(row.Close) >= Number(row.Open));
    const volumeColors = upDown.map((up) => up ? "#b91c1c" : "#15803d");
    const traces = [
      {
        x,
        open: plot.map((row) => row.Open),
        high: plot.map((row) => row.High),
        low: plot.map((row) => row.Low),
        close: plot.map((row) => row.Close),
        type: "candlestick",
        name: "K 線",
        increasing: { line: { color: "#b91c1c" }, fillcolor: "#b91c1c" },
        decreasing: { line: { color: "#15803d" }, fillcolor: "#15803d" },
        xaxis: "x",
        yaxis: "y",
      },
      { x, y: plot.map((row) => row.MA20), type: "scatter", mode: "lines", name: "MA20", line: { color: "#0f766e", width: 1.8 }, xaxis: "x", yaxis: "y" },
      { x, y: plot.map((row) => row.MA60), type: "scatter", mode: "lines", name: "MA60", line: { color: "#2563eb", width: 1.8 }, xaxis: "x", yaxis: "y" },
      { x, y: plot.map((row) => row.BB_UPPER), type: "scatter", mode: "lines", name: "布林上軌", line: { color: "#94a3b8", width: 1, dash: "dot" }, xaxis: "x", yaxis: "y" },
      { x, y: plot.map((row) => row.BB_LOWER), type: "scatter", mode: "lines", name: "布林下軌", line: { color: "#94a3b8", width: 1, dash: "dot" }, xaxis: "x", yaxis: "y" },
      { x, y: plot.map((row) => row.Volume), type: "bar", name: "成交量", marker: { color: volumeColors, opacity: 0.48 }, xaxis: "x", yaxis: "y2" },
      { x, y: plot.map((row) => row.MACD_HIST), type: "bar", name: "MACD Histogram", marker: { color: plot.map((row) => Number(row.MACD_HIST) >= 0 ? "#b91c1c" : "#15803d"), opacity: 0.72 }, xaxis: "x", yaxis: "y3" },
      { x, y: plot.map((row) => row.MACD_DIF), type: "scatter", mode: "lines", name: "DIF", line: { color: "#2563eb", width: 1.4 }, xaxis: "x", yaxis: "y3" },
      { x, y: plot.map((row) => row.MACD_SIGNAL), type: "scatter", mode: "lines", name: "MACD", line: { color: "#f59e0b", width: 1.4 }, xaxis: "x", yaxis: "y3" },
    ];
    Plotly.react(chart, traces, {
      height: 520,
      margin: { l: 54, r: 18, t: 16, b: 32 },
      paper_bgcolor: "#f8fafc",
      plot_bgcolor: "#ffffff",
      hovermode: "x",
      xaxis: { rangeslider: { visible: false }, showspikes: true, spikemode: "across", domain: [0, 1], anchor: "y3" },
      yaxis: { domain: [0.43, 1], title: "股價", gridcolor: "#e2e8f0", fixedrange: false },
      yaxis2: { domain: [0.24, 0.38], title: "量", gridcolor: "#e2e8f0", fixedrange: false },
      yaxis3: { domain: [0, 0.18], title: "MACD", gridcolor: "#e2e8f0", fixedrange: false },
      legend: { orientation: "h", y: 1.08 },
      bargap: 0.08,
    }, { responsive: true, displaylogo: false });
    return;
  }
  const plot = data.slice(-120);
  const width = 1000;
  const height = 520;
  const pad = 46;
  const priceH = 310;
  const volTop = 346;
  const macdTop = 428;
  const highs = plot.flatMap((r) => [r.High, r.MA20, r.MA60, r.BB_UPPER]).map(Number).filter(Number.isFinite);
  const lows = plot.flatMap((r) => [r.Low, r.MA20, r.MA60, r.BB_LOWER]).map(Number).filter(Number.isFinite);
  const min = Math.min(...lows);
  const max = Math.max(...highs);
  const spread = max - min || 1;
  const x = (i) => pad + (i / Math.max(plot.length - 1, 1)) * (width - pad * 2);
  const y = (v) => pad + (1 - (Number(v) - min) / spread) * (priceH - pad);
  const candleW = Math.max(3, (width - pad * 2) / plot.length * 0.52);
  const maxVol = Math.max(...plot.map((r) => Number(r.Volume) || 0), 1);
  const macdScale = chartScale(plot, "MACD_HIST", height - macdTop, 8);

  const candles = plot.map((r, i) => {
    const xi = x(i);
    const up = Number(r.Close) >= Number(r.Open);
    const color = up ? "#b91c1c" : "#15803d";
    const top = Math.min(y(r.Open), y(r.Close));
    const h = Math.max(Math.abs(y(r.Open) - y(r.Close)), 1);
    const volH = ((Number(r.Volume) || 0) / maxVol) * 54;
    return `
      <line x1="${xi}" x2="${xi}" y1="${y(r.High)}" y2="${y(r.Low)}" stroke="${color}" stroke-width="1" />
      <rect x="${xi - candleW / 2}" y="${top}" width="${candleW}" height="${h}" fill="${color}" />
      <rect x="${xi - candleW / 2}" y="${volTop + 60 - volH}" width="${candleW}" height="${volH}" fill="${color}" opacity="0.45" />
      <rect x="${xi - candleW / 2}" y="${macdTop + macdScale.y(r.MACD_HIST)}" width="${candleW}" height="${Math.abs(macdScale.y(0) - macdScale.y(r.MACD_HIST))}" fill="${Number(r.MACD_HIST) >= 0 ? "#b91c1c" : "#15803d"}" opacity="0.65" />
    `;
  }).join("");

  const line = (key, color, widthPx = 2) => {
    const pts = plot.filter((r) => r[key] !== null).map((r, i) => `${x(i).toFixed(1)},${y(r[key]).toFixed(1)}`).join(" ");
    return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="${widthPx}" stroke-linejoin="round" />`;
  };

  chart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}">
      <rect x="${pad}" y="${pad}" width="${width - pad * 2}" height="${priceH - pad}" fill="#fff" stroke="#d9e2ec" />
      <text x="${pad}" y="24" fill="#64748b" font-size="12">${fmt(max, 2)}</text>
      <text x="${pad}" y="${priceH + 4}" fill="#64748b" font-size="12">${fmt(min, 2)}</text>
      ${candles}
      ${line("MA20", "#0f766e")}
      ${line("MA60", "#2563eb")}
      ${line("BB_UPPER", "#94a3b8", 1)}
      ${line("BB_LOWER", "#94a3b8", 1)}
      <text x="${pad}" y="${volTop - 8}" fill="#64748b" font-size="12">Volume</text>
      <line x1="${pad}" x2="${width - pad}" y1="${volTop + 62}" y2="${volTop + 62}" stroke="#d9e2ec" />
      <text x="${pad}" y="${macdTop - 8}" fill="#64748b" font-size="12">MACD Histogram</text>
      <line x1="${pad}" x2="${width - pad}" y1="${macdTop + macdScale.y(0)}" y2="${macdTop + macdScale.y(0)}" stroke="#cbd5e1" />
    </svg>`;
}

function renderModeTables() {
  renderTable("modeTable", [
    { 面向: "核心目的", "短線／波段": "抓 1-8 週價差與動能延續", "長線／存股": "看半年以上趨勢、估值與現金流" },
    { 面向: "主要判斷", "短線／波段": "MA5、MA20、MACD、RSI、量比", "長線／存股": "MA60、MA120、MA240、PE、EPS、殖利率、Beta" },
    { 面向: "加分條件", "短線／波段": "短均線向上、MACD 擴大、RSI 強勢未過熱", "長線／存股": "站上季線/年線、PE 合理、EPS 為正、殖利率佳" },
    { 面向: "風控邏輯", "短線／波段": "停損較嚴格，重視 ATR 與隔日跳空", "長線／存股": "可分批布局，但仍限制單筆最大損失" },
  ]);
  renderTable("scoreRuleTable", [
    { 分數區間: "8.5-10", 評語: "條件優良／可優先觀察", 代表意義: "多數關鍵條件符合，目前結構相對完整" },
    { 分數區間: "7.0-8.4", 評語: "偏多格局／可列入候選", 代表意義: "整體偏多，但仍需觀察追價風險" },
    { 分數區間: "5.5-6.9", 評語: "中性偏多／等待確認", 代表意義: "部分條件轉佳，但尚未全面確認" },
    { 分數區間: "4.0-5.4", 評語: "中性偏弱／保守觀望", 代表意義: "條件偏弱，進場需要更保守" },
    { 分數區間: "0-3.9", 評語: "偏弱或高風險", 代表意義: "多數條件不理想，應降低積極度" },
  ]);
}

function downloadCsv(filename, rows) {
  if (!rows?.length) return;
  const keys = Object.keys(rows[0]).filter((key) => key !== "index");
  const csv = [keys.join(","), ...rows.map((row) => keys.map((key) => `"${String(row[key] ?? "").replaceAll('"', '""')}"`).join(","))].join("\n");
  const blob = new Blob(["\ufeff", csv], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

function renderAnalysis(data) {
  currentAnalysis = data;
  $("stockName").textContent = `${data.name}（${data.ticker}）`;
  $("scoreValue").textContent = data.score?.score_10 ?? "--";
  $("scoreLabel").textContent = data.score?.label || "--";
  $("closeValue").textContent = fmt(data.latest?.Close, 2);
  $("rsiValue").textContent = fmt(data.latest?.RSI14, 1);
  $("macdValue").textContent = fmt(data.latest?.MACD_HIST, 2);
  $("volumeRatioValue").textContent = fmt(data.latest?.Volume_Ratio, 2);
  $("latestDateValue").textContent = data.latest_date || "--";

  const change = Number(data.price_change || 0);
  const changePct = Number(data.price_change_pct || 0);
  $("changeValue").textContent = `${change >= 0 ? "+" : ""}${fmt(change, 2)} (${changePct >= 0 ? "+" : ""}${fmt(changePct, 2)}%)`;
  $("changeValue").className = change >= 0 ? "up" : "down";

  renderList("reasonsList", data.score?.reasons, "目前沒有明顯偏多理由。");
  renderList("warningsList", data.score?.warnings, "目前沒有明顯技術警訊，但仍需控管部位。");
  renderTable("levelsTable", data.levels, ["項目", "數值", "與現價距離%"]);
  renderLineChart("priceChart", data.price_history, ["Close"], ["#0f766e"]);
  renderCandleChart(data.price_history);
  renderTable("indicatorTable", data.recent_indicators, ["date", "Close", "Volume", "MA5", "MA20", "MA60", "MA120", "MA240", "RSI14", "MACD_HIST", "ATR14", "Volume_Ratio", "Return_5D", "Return_20D"]);

  $("peValue").textContent = fmt(data.fundamentals?.pe, 2);
  $("epsValue").textContent = fmt(data.fundamentals?.eps, 2);
  $("yieldValue").textContent = fmt(data.fundamentals?.dividend_yield_pct, 2, "%");
  $("betaValue").textContent = fmt(data.fundamentals?.beta, 2);
  $("marketCapValue").textContent = fmt(data.fundamentals?.market_cap / 100000000, 1, " 億");
  renderTable("financialTable", data.financials);

  const risk = data.risk_plan || {};
  $("stopLossValue").textContent = fmt(risk.stop_loss, 2);
  $("takeProfitValue").textContent = fmt(risk.take_profit, 2);
  $("riskBudgetValue").textContent = fmt(risk.risk_budget, 0);
  $("lotsValue").textContent = fmt(risk.lots, 2, " 張");
  $("sharesValue").textContent = fmt(risk.shares, 0, " 股");
}

async function loadAnalysis() {
  setStatus("分析中，正在取得股價與財務資料...");
  const data = await getJson(`/api/analyze?${params()}`);
  renderAnalysis(data);
  backtestLoaded = false;
  strategyLoaded = false;
  setStatus("分析完成");
}

async function loadStrategies() {
  const data = await getJson("/api/strategies");
  strategies = data.strategies || [];
  $("strategySelect").innerHTML = strategies.map((s) => `<option>${s.name}</option>`).join("");
}

async function loadBacktest() {
  const costRate = Number($("costPctInput").value || 0) / 100;
  const strategyName = $("strategySelect").value || "保守均線趨勢｜少交易";
  const data = await getJson(`/api/backtest?${params({ execution_mode: $("backtestModeInput").value, cost_rate: costRate, strategy_name: strategyName })}`);
  const d = data.detail || {};
  $("btReturnValue").textContent = fmtPct(d.total_return);
  $("btBuyHoldValue").textContent = fmtPct(d.buyhold_return);
  $("btCagrValue").textContent = fmtPct(d.cagr);
  $("btDrawdownValue").textContent = fmtPct(d.max_dd);
  $("btSharpeValue").textContent = fmt(d.sharpe, 2);
  $("btTradesValue").textContent = d.trades ?? "--";
  renderLineChart("equityChart", d.equity_curve || [], ["BuyHold_Equity", "Strategy_Equity"], ["#64748b", "#0f766e"]);
  renderTable("backtestTable", data.comparison);
  backtestLoaded = true;
}

async function loadOptimize() {
  const costRate = Number($("costPctInput").value || 0) / 100;
  const data = await getJson(`/api/optimize?${params({ strategy_family: $("optFamilyInput").value, execution_mode: $("backtestModeInput").value, cost_rate: costRate, optimize_target: $("optTargetInput").value })}`);
  renderTable("optTable", data.rows);
}

async function loadStrategyAdvice() {
  const strategyName = $("strategySelect").value || "保守均線趨勢｜少交易";
  const data = await getJson(`/api/strategy-advice?${params({ strategy_name: strategyName })}`);
  const a = data.advice || {};
  $("strategyMetaText").textContent = data.meta ? `${data.meta.style}｜${data.meta.rule}` : "--";
  $("adviceStatus").textContent = a.status || "--";
  $("adviceLevel").textContent = fmt(a.key_level, 2);
  $("adviceDistance").textContent = fmt(a.distance_pct, 2, "%");
  $("adviceAction").textContent = a.action || "--";
  renderTable("adviceRuleTable", [{ 項目: "進場", 規則: a.entry || "--" }, { 項目: "出場", 規則: a.exit_rule || "--" }]);
  renderList("adviceReasons", a.reasons, "目前沒有明確判斷依據。");
  strategyLoaded = true;
}

async function loadWatchlist() {
  const symbols = $("watchlistInput").value;
  setStatus("正在掃描觀察清單...");
  const data = await getJson(`/api/watchlist?${params({ symbols })}`);
  renderTable("watchlistTable", data.rows);
  setStatus("觀察清單掃描完成");
}

async function loadAi(kind = currentAiKind) {
  currentAiKind = kind;
  const data = await getJson(`/api/ai/${kind}?top_n=30`);
  $("aiUpdatedAt").textContent = data.updated_at ? `資料更新時間：${data.updated_at}` : "資料更新時間：未知";
  const columns = kind === "momentum"
    ? ["Rank", "ID", "Name", "Industry", "Close", "AI_Score", "F_Hist_Vol", "F_BB_Width", "F_ROC_10"]
    : ["Rank", "ID", "Name", "Industry", "Close", "AI_Potential_Score", "Tag", "RSI14", "Drawdown_52W", "Reason", "Risk"];
  renderTable("aiTable", data.rows, columns);
}

function wireEvents() {
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab").forEach((el) => el.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((el) => el.classList.remove("active"));
      btn.classList.add("active");
      $(`tab-${btn.dataset.tab}`).classList.add("active");
      if (btn.dataset.tab === "backtest" && !backtestLoaded) {
        loadBacktest().catch((error) => setStatus(`回測失敗：${error.message}`));
      }
      if (btn.dataset.tab === "strategy" && !strategyLoaded) {
        loadStrategyAdvice().catch((error) => setStatus(`策略助手失敗：${error.message}`));
      }
    });
  });

  $("analysisForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await loadAnalysis();
    } catch (error) {
      setStatus(`分析失敗：${error.message}`);
    }
  });
  $("runBacktestBtn").addEventListener("click", () => Promise.allSettled([loadBacktest(), loadStrategyAdvice()]));
  $("runOptimizeBtn").addEventListener("click", () => loadOptimize().catch((e) => setStatus(`最佳化失敗：${e.message}`)));
  $("strategySelect").addEventListener("change", () => Promise.allSettled([loadBacktest(), loadStrategyAdvice()]));
  $("runWatchlistBtn").addEventListener("click", () => loadWatchlist().catch((e) => setStatus(`掃描失敗：${e.message}`)));
  $("momentumBtn").addEventListener("click", async () => {
    $("momentumBtn").classList.add("active");
    $("potentialBtn").classList.remove("active");
    await loadAi("momentum");
  });
  $("potentialBtn").addEventListener("click", async () => {
    $("potentialBtn").classList.add("active");
    $("momentumBtn").classList.remove("active");
    await loadAi("potential");
  });
  $("downloadIndicatorsBtn").addEventListener("click", () => downloadCsv(`${currentAnalysis?.display_code || "stock"}_analysis.csv`, currentAnalysis?.recent_indicators || []));
}

async function init() {
  const today = new Date();
  const yearAgo = new Date();
  yearAgo.setFullYear(today.getFullYear() - 1);
  $("endDateInput").value = today.toISOString().slice(0, 10);
  $("startDateInput").value = yearAgo.toISOString().slice(0, 10);
  renderModeTables();
  wireEvents();
  await loadStrategies();
  await loadAi("momentum");
  await loadAnalysis();
}

init().catch((error) => setStatus(`初始化失敗：${error.message}`));
