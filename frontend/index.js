/* ═══════════════════════════════════════════════════════════
   IEEE-CIS Fraud Detection — Frontend Logic
   Talks to FastAPI at http://localhost:8000
   ═══════════════════════════════════════════════════════════ */

const API_BASE = "http://localhost:8000";

// ── helpers ─────────────────────────────────────────────────
function $(id) { return document.getElementById(id); }

function trafficClass(val) {
    if (val >= 0.7) return "tl-good";
    if (val >= 0.5) return "tl-monitor";
    return "tl-bad";
}

function levelClass(level) {
    if (!level) return "";
    return "level-" + level.toLowerCase();
}

function debounce(fn, ms) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ── 1 · Load metrics (/metrics) ─────────────────────────────
async function loadMetrics() {
    try {
        const res = await fetch(API_BASE + "/metrics");
        if (!res.ok) throw new Error(res.statusText);
        const m = await res.json();

        setMetricCard("precision", m.precision);
        setMetricCard("recall",    m.recall);
        setMetricCard("f1",        m.f1_score);
        setMetricCard("auprc",     m.auprc);

        $("val-roc-auc").textContent = m.roc_auc.toFixed(4);

        // Dynamically set slider to the optimal mathematical threshold found during training
        if (m.threshold) {
            $("threshold-slider").value = m.threshold;
            $("threshold-display").textContent = m.threshold.toFixed(2);
        }

        setStatus("online", "API Connected");
    } catch (e) {
        console.error("loadMetrics:", e);
        setStatus("offline", "API Offline");
    }
}

function setMetricCard(key, val) {
    const valEl  = $("val-" + key);
    const barEl  = $("bar-" + key);
    const cardEl = $("card-" + key);
    if (!valEl) return;

    valEl.textContent = val.toFixed(4);
    barEl.style.width = (val * 100).toFixed(1) + "%";

    // traffic-light
    cardEl.classList.remove("tl-good", "tl-monitor", "tl-bad");
    cardEl.classList.add(trafficClass(val));
}

function setStatus(state, text) {
    const dot = document.querySelector(".status-dot");
    const txt = document.querySelector(".status-text");
    dot.className = "status-dot " + state;
    txt.textContent = text;
}

// ── 1.5 Load curves (/curves) ───────────────────────────────
let rocChartInstance = null;
let prChartInstance = null;

async function loadCurves() {
    try {
        const res = await fetch(API_BASE + "/curves");
        if (!res.ok) throw new Error(res.statusText);
        const data = await res.json();
        
        // ── ROC Chart
        const rocCtx = document.getElementById('rocChart').getContext('2d');
        const rocDataPoints = data.roc.fpr.map((x, i) => ({ x: x, y: data.roc.tpr[i] }));
        if (rocChartInstance) rocChartInstance.destroy();
        
        Chart.defaults.color = '#8b95ab';
        Chart.defaults.font.family = "'Inter', sans-serif";

        rocChartInstance = new Chart(rocCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'ROC Curve',
                        data: rocDataPoints,
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        fill: true,
                        tension: 0.1,
                        pointRadius: 0,
                        pointHoverRadius: 5
                    },
                    {
                        label: 'Random Guess',
                        data: [{x: 0, y: 0}, {x: 1, y: 1}],
                        borderColor: '#555f74',
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0,
                        pointHoverRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False Positive Rate' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'True Positive Rate' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: function(ctx) { return `FPR: ${ctx.raw.x.toFixed(3)}, TPR: ${ctx.raw.y.toFixed(3)}`; } } }
                }
            }
        });

        // ── PR Chart
        const prCtx = document.getElementById('prChart').getContext('2d');
        const prDataPoints = data.pr.recall.map((x, i) => ({ x: x, y: data.pr.precision[i] }));
        if (prChartInstance) prChartInstance.destroy();

        prChartInstance = new Chart(prCtx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Precision-Recall',
                    data: prDataPoints,
                    borderColor: '#818cf8',
                    backgroundColor: 'rgba(129, 140, 248, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Recall' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Precision' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: function(ctx) { return `Recall: ${ctx.raw.x.toFixed(3)}, Prec: ${ctx.raw.y.toFixed(3)}`; } } }
                }
            }
        });

    } catch (e) {
        console.error("loadCurves:", e);
    }
}

// ── 2 · Threshold simulation (/simulate) ────────────────────
const slider  = $("threshold-slider");
const display = $("threshold-display");

slider.addEventListener("input", () => {
    display.textContent = parseFloat(slider.value).toFixed(2);
    debouncedSimulate();
});

const debouncedSimulate = debounce(simulateThreshold, 300);

async function simulateThreshold() {
    const threshold = parseFloat(slider.value);
    try {
        const res = await fetch(API_BASE + "/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ threshold })
        });
        if (!res.ok) throw new Error(res.statusText);
        const d = await res.json();

        // FP / FN cards
        $("val-fp").textContent = d.false_positives.toLocaleString();
        $("val-fn").textContent = d.false_negatives.toLocaleString();

        const levelFriction = $("level-friction");
        levelFriction.textContent = d.customer_friction;
        levelFriction.className = "tradeoff-level " + levelClass(d.customer_friction);

        const levelLoss = $("level-loss");
        levelLoss.textContent = d.financial_loss;
        levelLoss.className = "tradeoff-level " + levelClass(d.financial_loss);

        // sim metrics
        $("sim-precision").textContent = d.precision.toFixed(4);
        $("sim-recall").textContent    = d.recall.toFixed(4);
        $("sim-f1").textContent        = d.f1_score.toFixed(4);
    } catch (e) {
        console.error("simulate:", e);
    }
}

// ── 3 · Transaction analyzer (/analyze, /sample) ────────────
$("btn-load-sample").addEventListener("click", loadSample);
$("btn-analyze").addEventListener("click", analyzeTransaction);

async function loadSample() {
    try {
        const res = await fetch(API_BASE + "/sample");
        if (!res.ok) throw new Error(res.statusText);
        const d = await res.json();

        $("feature-json").value = JSON.stringify(d.features, null, 2);

        // show actual label
        const row = $("actual-label-row");
        row.style.display = "flex";
        const badge = $("actual-label-value");
        if (d.actual_label === 1) {
            badge.textContent = "Fraud";
            badge.className = "badge badge-fraud";
        } else {
            badge.textContent = "Legitimate";
            badge.className = "badge badge-legitimate";
        }
    } catch (e) {
        console.error("loadSample:", e);
        alert("Could not load sample. Is the API running?");
    }
}

async function analyzeTransaction() {
    const raw = $("feature-json").value.trim();
    if (!raw) { alert("Paste or load transaction features first."); return; }

    let features;
    try { features = JSON.parse(raw); } catch (e) { alert("Invalid JSON."); return; }

    const threshold = parseFloat(slider.value);

    try {
        const res = await fetch(API_BASE + "/analyze?explain=true", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features, threshold })
        });
        if (!res.ok) { const t = await res.text(); alert("Error: " + t); return; }
        const d = await res.json();
        renderAnalysis(d, threshold);
    } catch (e) {
        console.error("analyze:", e);
        alert("API error. Is the server running?");
    }
}

function renderAnalysis(d, threshold) {
    const panel = $("analyzer-results");
    panel.style.display = "block";

    // prediction label
    const pred = $("result-prediction");
    pred.textContent = d.prediction;
    pred.className = "result-prediction " + (d.prediction === "Fraud" ? "pred-fraud" : "pred-legitimate");

    // near-miss
    const nm = $("near-miss-badge");
    nm.style.display = d.is_near_miss ? "inline-block" : "none";

    // gauge
    const pct = (d.probability_score * 100).toFixed(1);
    $("gauge-fill").style.width = pct + "%";
    $("gauge-marker").style.left = (threshold * 100).toFixed(1) + "%";
    $("result-score").textContent = d.probability_score.toFixed(6);

    // LIME chart
    renderLime(d.lime_explanation || []);
}

function renderLime(features) {
    const container = $("lime-chart");
    container.innerHTML = "";
    if (!features.length) {
        container.innerHTML = '<p style="color:var(--text-muted);font-size:.85rem;">No LIME data available.</p>';
        return;
    }

    // find max weight for scaling
    const maxW = Math.max(...features.map(f => Math.abs(f.weight)), 0.0001);

    for (const f of features) {
        const row = document.createElement("div");
        row.className = "lime-row";

        const nameEl = document.createElement("div");
        nameEl.className = "lime-feat-name";
        nameEl.textContent = f.feature;
        nameEl.title = f.feature;

        const wrapEl = document.createElement("div");
        wrapEl.className = "lime-bar-wrap";

        // center line
        const center = document.createElement("div");
        center.className = "lime-bar-center";
        wrapEl.appendChild(center);

        // bar
        const bar = document.createElement("div");
        const frac = Math.abs(f.weight) / maxW;
        const widthPct = (frac * 48).toFixed(1); // max 48% of half
        bar.className = "lime-bar " + (f.direction === "positive" ? "lime-bar-pos" : "lime-bar-neg");

        if (f.direction === "positive") {
            bar.style.left = "50%";
            bar.style.width = widthPct + "%";
        } else {
            bar.style.right = "50%";
            bar.style.width = widthPct + "%";
            bar.style.left = (50 - parseFloat(widthPct)).toFixed(1) + "%";
        }

        // weight label
        const lbl = document.createElement("span");
        lbl.className = "lime-weight";
        lbl.textContent = f.weight > 0 ? "+" + f.weight.toFixed(4) : f.weight.toFixed(4);
        if (f.direction === "positive") {
            lbl.style.left = "calc(50% + " + widthPct + "% + 4px)";
            lbl.style.color = "var(--red)";
        } else {
            lbl.style.right = "calc(50% + " + widthPct + "% + 4px)";
            lbl.style.textAlign = "right";
            lbl.style.color = "var(--green)";
        }

        wrapEl.appendChild(bar);
        wrapEl.appendChild(lbl);

        row.appendChild(nameEl);
        row.appendChild(wrapEl);
        container.appendChild(row);
    }
}

// ── Boot ────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    loadMetrics();
    loadCurves();
    simulateThreshold();
});
