import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Sampling lab", layout="wide")
st.markdown(
    """
    <style>
      .stApp, [data-testid="stAppViewContainer"] {background: white;}
      header[data-testid="stHeader"], #MainMenu, footer {display: none;}
      .block-container {padding-top: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

components.html(
    r"""
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; }
  html, body { background: white; color: #17212a; color-scheme: light; }
  body { margin: 0; font: 14px system-ui, sans-serif; }
  main {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: clamp(3px, 1vw, 12px);
    max-width: 1120px;
  }
  section { min-width: 0; }
  h2 { font-size: clamp(12px, 1.5vw, 17px); margin: 0; }
  .controls {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 5px;
    padding-top: 8px;
  }
  select, input { max-width: 100%; font: inherit; }
  input[type=number] { width: 5em; }
  input[type=checkbox] { accent-color: #007c78; }
  canvas { display: block; width: 100%; background: white; }
  .pick { cursor: crosshair; touch-action: manipulation; }
  [hidden] { display: none !important; }
</style>

<main id="lab">
  <section>
    <h2>1. Input signal</h2>
    <div class="controls">
      <select id="signal" aria-label="Input signal">
        <option>Cosine</option><option>Sine</option><option>Exponential</option>
      </select>
      <label id="freqLabel">f
        <input id="f0" type="number" min="0" max="32" step="any" value="2"> Hz
      </label>
      <label id="decayLabel" hidden>a
        <input id="decay" type="number" min="0" step="any" value="2"> s⁻¹
      </label>
    </div>
  </section>

  <section>
    <h2>2. Samples</h2>
    <div class="controls">
      <select id="mode" aria-label="Sampling mode">
        <option>Manual</option><option>Sampling frequency</option>
      </select>
      <label id="rateLabel" hidden>fₛ
        <input id="fs" type="number" min="0.1" max="64" step="any" value="8"> Hz
      </label>
    </div>
  </section>

  <section>
    <h2>3. Piecewise linear</h2>
    <div class="controls"></div>
  </section>

  <section>
    <h2>4. Inverse Fourier transform</h2>
    <div class="controls">
      <label><input id="low" type="checkbox" role="switch"> Low pass</label>
      <label id="cutoffLabel" hidden>f꜀
        <input id="fc" type="number" min="0" max="32" step="any" value="4"> Hz
      </label>
    </div>
  </section>
</main>

<script>
const get = id => document.getElementById(id);
const T = 2;                           // Duration in seconds
const displayRate = 256;               // Dense grid for drawing continuous curves
const N = T * displayRate;
const tau = 2 * Math.PI;
const time = Array.from({length: N}, (_, i) => i / displayRate);
const frequency = Array.from({length: 129}, (_, i) => (i - 64) / T);
const manualRate = 16;
const manualIndices = new Set();
const sections = [...document.querySelectorAll("section")];

const plots = sections.map(section => {
  return ["Time", "Frequency"].map(domain => {
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-label", section.querySelector("h2").textContent + " — " + domain);
    section.appendChild(canvas);
    return canvas;
  });
});

function readNumber(id) {
  const input = get(id);
  return input.validity.valid ? input.valueAsNumber : NaN;
}

// Fourier-series coefficients over the two-second observation window.
function spectrum(t, y, weight) {
  return frequency.map(f => {
    let re = 0, im = 0;
    for (let i = 0; i < t.length; i++) {
      const angle = tau * f * t[i];
      re += y[i] * Math.cos(angle);
      im -= y[i] * Math.sin(angle);
    }
    return [weight * re, weight * im];
  });
}

function inverse(coefficients) {
  return time.map(t => {
    let sum = 0;
    coefficients.forEach(([re, im], k) => {
      const angle = tau * frequency[k] * t;
      sum += re * Math.cos(angle) - im * Math.sin(angle);
    });
    return sum;
  });
}

function linearValue(t, points, values) {
  const i = points.findIndex(u => u >= t);
  if (i < 0) return 0;
  if (points[i] === t) return values[i];
  if (i === 0) return 0;
  const fraction = (t - points[i - 1]) / (points[i] - points[i - 1]);
  return values[i - 1] + fraction * (values[i] - values[i - 1]);
}

const W = 280, H = 165, L = 36, R = 271, U = 10, B = 139;

function draw(canvas, x, y, options = {}) {
  const {spectral = false, stems = false, dots = false, candidates = []} = options;
  const width = Math.max(1, canvas.clientWidth);
  const height = width * H / W;
  const dpr = window.devicePixelRatio || 1;
  canvas.style.height = height + "px";
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.max(1, Math.round(height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(canvas.width / W, 0, 0, canvas.height / H, 0, 0);

  const xmin = spectral ? -32 : 0;
  const xmax = spectral ? 32 : T;
  const peak = Math.max(spectral ? 0.01 : 1, ...y.map(Math.abs));
  const ymax = 1.08 * peak;
  const ymin = spectral ? 0 : -ymax;
  const X = v => L + (v - xmin) / (xmax - xmin) * (R - L);
  const Y = v => B - (v - ymin) / (ymax - ymin) * (B - U);

  function line(x1, y1, x2, y2, color) {
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
  function dot(t, v, radius, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(X(t), Y(v), radius, 0, tau);
    ctx.fill();
  }

  line(L, U, L, B, "#a0aab2");
  line(L, Y(0), R, Y(0), "#a0aab2");
  ctx.fillStyle = "#465864";
  ctx.font = "11px system-ui";
  ctx.textAlign = "center";
  for (const v of [xmin, (xmin + xmax) / 2, xmax]) {
    ctx.fillText(String(v), X(v), B + 17);
  }
  ctx.textAlign = "right";
  for (const v of [ymin, ymax]) ctx.fillText(String(+v.toFixed(2)), L - 4, Y(v) + 3);

  ctx.lineWidth = 1.6;
  if (stems) {
    y.forEach((v, i) => {
      if (Math.abs(v) > 1e-12) line(X(x[i]), Y(0), X(x[i]), Y(v), "#007c78");
    });
  } else {
    ctx.strokeStyle = "#007c78";
    ctx.beginPath();
    y.forEach((v, i) => {
      if (i === 0) ctx.moveTo(X(x[i]), Y(v));
      else ctx.lineTo(X(x[i]), Y(v));
    });
    ctx.stroke();
  }
  for (const [t, v] of candidates) dot(t, v, 2, "#a0aab2");
  if (dots) y.forEach((v, i) => dot(x[i], v, 3, "#bd5027"));
}

function render() {
  const exponential = get("signal").value === "Exponential";
  const uniform = get("mode").value === "Sampling frequency";
  const filtered = get("low").checked;
  get("freqLabel").hidden = exponential;
  get("decayLabel").hidden = !exponential;
  get("rateLabel").hidden = !uniform;
  get("cutoffLabel").hidden = !filtered;
  plots[1][0].classList.toggle("pick", !uniform);

  // Align all plots even when a heading or control wraps.
  const controls = sections.map(section => section.querySelector(".controls"));
  for (const c of controls) c.style.minHeight = "";
  const plotTop = Math.max(...controls.map(c => c.offsetTop + c.offsetHeight));
  for (const c of controls) c.style.minHeight = (plotTop - c.offsetTop) + "px";

  const f0 = readNumber("f0"), a = readNumber("decay");
  const fs = readNumber("fs"), fc = readNumber("fc");
  const validSignal = Number.isFinite(exponential ? a : f0);

  function wave(t) {
    if (exponential) return Math.exp(-a * t);
    return get("signal").value === "Cosine"
      ? Math.cos(tau * f0 * t)
      : Math.sin(tau * f0 * t);
  }

  let samples;
  if (uniform) {
    samples = Number.isFinite(fs)
      ? Array.from({length: Math.ceil(T * fs)}, (_, k) => k / fs)
      : [];
  } else {
    samples = [...manualIndices].sort((a, b) => a - b).map(k => k / manualRate);
  }
  if (!validSignal) samples = [];

  const original = validSignal ? time.map(wave) : [];
  const values = samples.map(wave);
  const joined = time.map(t => linearValue(t, samples, values));
  const candidates = [];
  if (!uniform && validSignal) {
    for (let k = 0; k < T * manualRate; k++) {
      if (!manualIndices.has(k)) candidates.push([k / manualRate, wave(k / manualRate)]);
    }
  }

  // A sampled impulse train has coefficients (1/T) Σ x[n] exp(-j2πft_n).
  const sampledSpectrum = spectrum(samples, values, 1 / T);

  // Uniform sampling: the ideal reconstruction filter has passband gain Ts.
  // Manual sampling has no single Ts; the switch applies a unit-gain LPF.
  const gain = uniform && Number.isFinite(fs) ? 1 / fs : 1;
  const outputSpectrum = filtered
    ? sampledSpectrum.map(([re, im], k) =>
        Math.abs(frequency[k]) <= fc ? [gain * re, gain * im] : [0, 0])
    : sampledSpectrum;

  const spectra = [
    spectrum(time.slice(0, original.length), original, 1 / N),
    sampledSpectrum,
    spectrum(time, joined, 1 / N),
    outputSpectrum
  ];

  draw(plots[0][0], time, original);
  draw(plots[1][0], samples, values, {stems: true, dots: true, candidates});
  draw(plots[2][0], samples, values, {dots: true});
  if (filtered) draw(plots[3][0], time, inverse(outputSpectrum));
  else draw(plots[3][0], samples, values, {stems: true, dots: true});

  spectra.forEach((coefficients, i) => {
    const magnitude = coefficients.map(([re, im]) => Math.hypot(re, im));
    draw(plots[i][1], frequency, magnitude, {spectral: true, stems: true});
  });
}

plots[1][0].onclick = event => {
  if (get("mode").value !== "Manual") return;
  const rect = plots[1][0].getBoundingClientRect();
  const x = (event.clientX - rect.left) * W / rect.width;
  const y = (event.clientY - rect.top) * H / rect.height;
  if (x < L || x > R || y < U || y > B) return;
  const k = Math.min(T * manualRate - 1, Math.round((x - L) / (R - L) * T * manualRate));
  if (manualIndices.has(k)) manualIndices.delete(k);
  else manualIndices.add(k);
  render();
};

get("lab").addEventListener("input", render);
window.addEventListener("resize", render);
render();
</script>
</html>
""",
    height=530,
    scrolling=True,
)