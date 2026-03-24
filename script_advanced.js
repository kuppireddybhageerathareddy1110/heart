/* =========================================================
   CardioAI — script_advanced.js
   ========================================================= */

const API = 'http://127.0.0.1:8001';

const state = { patient: null, prediction: null };

/* ── navigation ─────────────────────────────────────────── */
document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    const target = document.getElementById(`${btn.dataset.section}Section`);
    if (target) target.classList.add('active');
  });
});

/* ── tabs ───────────────────────────────────────────────── */
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    const target = document.getElementById(`${btn.dataset.tab}Tab`);
    if (target) target.classList.add('active');
  });
});

/* ── loading overlay ────────────────────────────────────── */
function loading(show, msg = 'Analysing patient data…') {
  const el = document.getElementById('loadingOverlay');
  document.getElementById('loadingMsg').textContent = msg;
  el.classList.toggle('active', show);
}

/* ── form submit ────────────────────────────────────────── */
document.getElementById('predictionForm').addEventListener('submit', async e => {
  e.preventDefault();
  loading(true);

  const patient = {
    age:             parseFloat(document.getElementById('age').value),
    sex:             document.getElementById('sex').value,
    oldpeak:         parseFloat(document.getElementById('oldpeak').value),
    chest_pain:      document.getElementById('chest_pain').value,
    restingbp_final: parseFloat(document.getElementById('restingbp_final').value),
    chol_final:      parseFloat(document.getElementById('chol_final').value),
    maxhr_final:     parseFloat(document.getElementById('maxhr_final').value),
    fasting_bs:      document.getElementById('fasting_bs').value,
    resting_ecg:     document.getElementById('resting_ecg').value,
    exercise_angina: document.getElementById('exercise_angina').value,
    st_slope:        document.getElementById('st_slope').value,
  };

  state.patient = patient;

  try {
    const res = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patient),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const result = await res.json();
    state.prediction = result;

    displayResults(result);
    // kick off explanations silently
    loadExplanations(patient);

  } catch (err) {
    alert(`❌ Could not reach the API.\n\nMake sure the backend is running:\n  uvicorn api:app --port 8001\n\n(${err.message})`);
  } finally {
    loading(false);
  }
});

/* ── display results ────────────────────────────────────── */
function displayResults(r) {
  document.getElementById('resultsPlaceholder').style.display = 'none';
  const panel = document.getElementById('predictionResults');
  panel.style.display = 'block';

  /* verdict badge */
  const lvl  = r.risk_level.toLowerCase();               // 'low' | 'medium' | 'high'
  const badge = document.getElementById('verdictBadge');
  badge.className = `verdict-badge ${lvl}`;
  const emojis = { low: '🟢', medium: '🟡', high: '🔴' };
  const labels = { low: 'Low Risk', medium: 'Medium Risk', high: 'High Risk' };
  badge.innerHTML = `
    <div class="verdict-emoji">${emojis[lvl]}</div>
    <div class="verdict-label">${labels[lvl]}</div>
    <div class="verdict-sub">${r.prediction}</div>
  `;

  /* arc gauge */
  const pct = r.probability * 100;
  // arc total path length ≈ 251.2 (half-circle, r=80)
  const arcLen = 251.2;
  const offset = arcLen - (pct / 100) * arcLen;
  const arcFill = document.getElementById('arcFill');
  arcFill.style.strokeDashoffset = offset;
  arcFill.style.stroke = pct >= 70 ? '#ef4444' : pct >= 40 ? '#f59e0b' : '#10b981';
  document.getElementById('arcPct').textContent = pct.toFixed(1) + '%';

  /* CI */
  const ci = r.confidence_interval;
  document.getElementById('ciText').textContent =
    `${(ci.lower * 100).toFixed(1)}% – ${(ci.upper * 100).toFixed(1)}%`;

  /* factors */
  renderList('riskList', r.risk_factors, 'No significant risk factors identified');
  renderList('protList', r.protective_factors, 'No significant protective factors identified');

  /* scroll */
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderList(id, items, emptyMsg) {
  const ul = document.getElementById(id);
  ul.innerHTML = '';
  if (items && items.length) {
    items.forEach(item => {
      const li = document.createElement('li');
      li.textContent = item;
      ul.appendChild(li);
    });
  } else {
    const li = document.createElement('li');
    li.textContent = emptyMsg;
    ul.appendChild(li);
  }
}

/* ── load explanations ───────────────────────────────────── */
async function loadExplanations(patient) {
  // waterfall
  try {
    const res = await fetch(`${API}/explain/waterfall`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patient),
    });
    if (res.ok) {
      const d = await res.json();
      setPlotImg('waterfallPlot', 'waterfallPh', d.plot);
    }
  } catch (_) {}

  // force plot
  try {
    const res = await fetch(`${API}/explain/force`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patient),
    });
    if (res.ok) {
      const d = await res.json();
      const frame = document.getElementById('forcePlotFrame');
      frame.srcdoc = d.plot;
      document.getElementById('forcePh').style.display = 'none';
    }
  } catch (_) {}

  // lime
  try {
    const res = await fetch(`${API}/explain/lime`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patient),
    });
    if (res.ok) {
      const d = await res.json();
      setPlotImg('limePlot', 'limePh', d.plot);
      if (d.feature_weights) renderLimeTable(d.feature_weights);
    }
  } catch (_) {}
}

function setPlotImg(imgId, phId, b64) {
  const img = document.getElementById(imgId);
  const ph  = document.getElementById(phId);
  img.src = `data:image/png;base64,${b64}`;
  img.classList.add('loaded');
  if (ph) ph.style.display = 'none';
}

function renderLimeTable(weights) {
  const tbl = document.getElementById('limeTable');
  const sec = document.getElementById('limeWeights');
  if (!weights.length) return;
  sec.style.display = 'block';

  let html = `<thead><tr>
    <th>Feature</th><th>Weight</th><th>Direction</th>
  </tr></thead><tbody>`;

  weights.forEach(({ feature, weight }) => {
    const pos = weight > 0;
    const color = pos ? '#be123c' : '#059669';
    html += `<tr>
      <td>${feature}</td>
      <td style="color:${color};font-weight:600">${weight > 0 ? '+' : ''}${weight.toFixed(4)}</td>
      <td style="color:${color}">${pos ? '↑ Increases Risk' : '↓ Decreases Risk'}</td>
    </tr>`;
  });

  html += '</tbody>';
  tbl.innerHTML = html;
}

/* ── global plots ────────────────────────────────────────── */
async function loadSummaryPlot() {
  loading(true, 'Computing global SHAP summary…');
  try {
    const res = await fetch(`${API}/explain/summary`);
    if (!res.ok) throw new Error();
    const d = await res.json();
    setPlotImg('summaryPlot', 'summaryPh', d.plot);
  } catch (_) {
    alert('Could not load summary plot. Is the API running?');
  } finally {
    loading(false);
  }
}

async function loadImportancePlot() {
  loading(true, 'Computing feature importance…');
  try {
    const res = await fetch(`${API}/explain/importance`);
    if (!res.ok) throw new Error();
    const d = await res.json();
    setPlotImg('importancePlot', 'importPh', d.plot);
  } catch (_) {
    alert('Could not load importance plot. Is the API running?');
  } finally {
    loading(false);
  }
}

/* ── navigate to explanations tab ───────────────────────── */
function showExplanations() {
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  const explBtn = document.querySelector('[data-section="explain"]');
  if (explBtn) explBtn.classList.add('active');
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.getElementById('explainSection').classList.add('active');
}

/* ── sample patient ──────────────────────────────────────── */
function loadSamplePatient() {
  const vals = {
    age: 65, sex: 'Male', chest_pain: 'Asymptomatic',
    restingbp_final: 160, chol_final: 300, fasting_bs: 'Yes',
    resting_ecg: 'ST-T Abnormality', maxhr_final: 130,
    exercise_angina: 'Yes', oldpeak: 2.5, st_slope: 'Flat',
  };
  Object.entries(vals).forEach(([k, v]) => {
    const el = document.getElementById(k);
    if (el) el.value = v;
  });

  // toast
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = '<i class="fas fa-check-circle"></i> Sample patient loaded';
  document.body.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 400);
  }, 2200);
}

/* ── download report ─────────────────────────────────────── */
function downloadReport() {
  if (!state.prediction || !state.patient) { alert('Run a prediction first.'); return; }
  const data = {
    ...state.patient,
    prediction:  state.prediction.prediction,
    probability: (state.prediction.probability * 100).toFixed(2) + '%',
    risk_level:  state.prediction.risk_level,
    generated:   new Date().toISOString(),
  };
  const csv   = Object.keys(data).join(',') + '\n' + Object.values(data).join(',');
  const blob  = new Blob([csv], { type: 'text/csv' });
  const url   = URL.createObjectURL(blob);
  const a     = document.createElement('a');
  a.href = url; a.download = `cardioai_report_${Date.now()}.csv`;
  a.click(); URL.revokeObjectURL(url);
}

/* ── toast styles (injected) ─────────────────────────────── */
const toastCSS = document.createElement('style');
toastCSS.textContent = `
.toast {
  position: fixed; bottom: 2rem; right: 2rem;
  background: #0f172a; color: #e2e8f0;
  padding: .75rem 1.25rem;
  border-radius: 8px;
  display: flex; align-items: center; gap: .5rem;
  font-size: .875rem; font-weight: 500;
  box-shadow: 0 8px 24px rgba(0,0,0,.25);
  transform: translateY(20px); opacity: 0;
  transition: transform .3s, opacity .3s;
  z-index: 9998;
}
.toast i { color: #10b981; }
.toast.show { transform: translateY(0); opacity: 1; }
`;
document.head.appendChild(toastCSS);
