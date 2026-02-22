import { useState, useEffect, useRef, useMemo } from "react";

// ─── FFT (Cooley-Tukey) ───────────────────────────────────────────────────────
function fft(re, im, invert = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (2 * Math.PI / len) * (invert ? 1 : -1);
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const uRe = re[i + j], uIm = im[i + j];
        const vRe = re[i + j + len / 2] * curRe - im[i + j + len / 2] * curIm;
        const vIm = re[i + j + len / 2] * curIm + im[i + j + len / 2] * curRe;
        re[i + j] = uRe + vRe; im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe; im[i + j + len / 2] = uIm - vIm;
        const nCr = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe; curRe = nCr;
      }
    }
  }
  if (invert) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

// ─── Hilbert analytic signal ──────────────────────────────────────────────────
// z[n] = x[n] + j·H{x}[n]   →   envelope = |z|,  phase = arg(z)
function analyticSignal(signal) {
  const n = signal.length;
  let N = 1; while (N < n) N <<= 1;
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  for (let i = 0; i < n; i++) re[i] = signal[i];
  fft(re, im, false);
  // one-sided spectrum: double k=1..N/2-1, zero k=N/2+1..N-1
  for (let k = 1; k < N / 2; k++) { re[k] *= 2; im[k] *= 2; }
  for (let k = N / 2 + 1; k < N; k++) { re[k] = 0; im[k] = 0; }
  fft(re, im, true);
  const envelope = new Float64Array(n);
  const phase = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    envelope[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    phase[i] = Math.atan2(im[i], re[i]);
  }
  return { envelope, phase };
}

// ─── Generation-indexed color ─────────────────────────────────────────────────
// t ∈ [0,1]:  0 = born early (deep indigo)  →  1 = born late (hot amber)
function genColor(t, alpha = 0.85) {
  let r, g, b;
  if (t < 0.33) {
    const s = t / 0.33;
    r = Math.round(40 + s * 10);   g = Math.round(60 + s * 130); b = Math.round(240 - s * 40);
  } else if (t < 0.66) {
    const s = (t - 0.33) / 0.33;
    r = Math.round(50 + s * 170);  g = Math.round(190 - s * 40); b = Math.round(200 - s * 130);
  } else {
    const s = (t - 0.66) / 0.34;
    r = Math.round(220 + s * 35);  g = Math.round(150 - s * 100); b = Math.round(70 - s * 40);
  }
  return `rgba(${r},${g},${b},${alpha})`;
}

// ─── Prime sieve + Goldbach computation ──────────────────────────────────────
function computeData(limit) {
  const sieve = new Uint8Array(limit + 1).fill(1);
  sieve[0] = sieve[1] = 0;
  for (let i = 2; i * i <= limit; i++)
    if (sieve[i]) for (let j = i * i; j <= limit; j += i) sieve[j] = 0;

  const oddP = [];
  for (let i = 3; i <= limit; i += 2) if (sieve[i]) oddP.push(i);

  // Appearance index τ(2m): the stage n at which 2m is first produced
  const appearedAt = new Map();
  const seen = new Set();
  for (let n = 0; n < oddP.length; n++) {
    const pn = oddP[n];
    for (let i = 0; i <= n; i++) {
      const s = pn + oddP[i];
      if (s > limit) break;
      if (!seen.has(s)) { appearedAt.set(s, n); seen.add(s); }
    }
  }

  const data = [];
  for (let m = 3; m * 2 <= limit; m++) {
    const twoM = 2 * m;
    let r2 = 0;
    for (const p of oddP) {
      if (p > m) break;
      if (sieve[twoM - p]) r2++;
    }
    const stage = appearedAt.get(twoM) ?? oddP.length - 1;
    data.push({
      x: twoM,
      y: r2,
      t: stage / Math.max(1, oddP.length - 1),   // normalised appearance index
    });
  }

  const { envelope, phase } = analyticSignal(data.map(d => d.y));
  return { data, envelope: Array.from(envelope), phase: Array.from(phase) };
}

// ─── Canvas draw helpers ──────────────────────────────────────────────────────
function drawComet(canvas, data, envelope, phase, opts) {
  if (!canvas || !data.length) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const pad = { l: 52, r: 18, t: 22, b: 36 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

  // Background
  ctx.fillStyle = "#030309";
  ctx.fillRect(0, 0, W, H);

  // Subtle grid
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = H - pad.b - (i / 5) * ph;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    const x = pad.l + (i / 5) * pw;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
  }

  const maxX = data[data.length - 1].x;
  const maxY = Math.max(...data.map(d => d.y), 1);
  const maxE = Math.max(...envelope.slice(0, data.length), 1);

  const cx = x => pad.l + (x / maxX) * pw;
  const cy = y => H - pad.b - (y / maxY) * ph;
  const cyE = y => H - pad.b - (y / maxE) * ph;

  // Points — coloured by appearance index
  for (let i = 0; i < data.length; i++) {
    const d = data[i];
    ctx.beginPath();
    ctx.arc(cx(d.x), cy(d.y), 2.8, 0, Math.PI * 2);
    ctx.fillStyle = genColor(d.t, 0.82);
    ctx.fill();
  }

  // ── Hilbert envelope ─────────────────────────────────────────────────────
  if (opts.showEnv) {
    ctx.save();
    ctx.shadowBlur = 14; ctx.shadowColor = "rgba(255,210,50,0.7)";
    ctx.strokeStyle = "rgba(255,205,40,0.92)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = cx(data[i].x), y = cyE(envelope[i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke(); ctx.restore();
  }

  // ── Axes & labels ─────────────────────────────────────────────────────────
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b);
  ctx.lineTo(W - pad.r, H - pad.b);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,0.38)";
  ctx.font = "11px 'Courier New', monospace";
  for (let i = 0; i <= 4; i++) {
    ctx.fillText(Math.round((i / 4) * maxY), 4, cy((i / 4) * maxY) + 4);
    ctx.fillText(Math.round((i / 4) * maxX), cx((i / 4) * maxX) - 12, H - pad.b + 18);
  }
  ctx.fillStyle = "rgba(255,255,255,0.28)";
  ctx.fillText("r₂(2m)", 4, pad.t - 4);
  ctx.fillText("2m →", W - pad.r - 28, H - 4);
}

function drawPhase(canvas, data, phase) {
  if (!canvas || !data.length) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const pad = { l: 52, r: 18, t: 12, b: 28 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

  ctx.fillStyle = "#020208";
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 1;
  [-Math.PI, 0, Math.PI].forEach(v => {
    const y = H - pad.b - ((v + Math.PI) / (2 * Math.PI)) * ph;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
  });

  const maxX = data[data.length - 1].x;
  const cx = x => pad.l + (x / maxX) * pw;
  const cy = v => H - pad.b - ((v + Math.PI) / (2 * Math.PI)) * ph;

  // Phase line — coloured by generation index
  ctx.lineWidth = 1.4;
  for (let i = 1; i < data.length; i++) {
    // skip discontinuities (phase wrap-around)
    if (Math.abs(phase[i] - phase[i - 1]) > Math.PI * 1.5) continue;
    ctx.strokeStyle = genColor(data[i].t, 0.65);
    ctx.beginPath();
    ctx.moveTo(cx(data[i - 1].x), cy(phase[i - 1]));
    ctx.lineTo(cx(data[i].x), cy(phase[i]));
    ctx.stroke();
  }

  // Zero reference
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.setLineDash([4, 5]);
  ctx.beginPath(); ctx.moveTo(pad.l, cy(0)); ctx.lineTo(W - pad.r, cy(0)); ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b);
  ctx.lineTo(W - pad.r, H - pad.b);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.font = "11px 'Courier New', monospace";
  ctx.fillText("φ(m)", 4, pad.t + 10);
  ctx.fillText("+π", 4, cy(Math.PI) + 4);
  ctx.fillText(" 0", 4, cy(0) + 4);
  ctx.fillText("−π", 4, cy(-Math.PI) + 4);
}

function drawSpectrum(canvas, data, envelope) {
  // Power spectral density of r₂ (log scale) — shows fractal frequency structure
  if (!canvas || !data.length) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const pad = { l: 52, r: 18, t: 12, b: 28 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

  ctx.fillStyle = "#020208";
  ctx.fillRect(0, 0, W, H);

  const n = data.length;
  let N = 1; while (N < n) N <<= 1;
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  for (let i = 0; i < n; i++) re[i] = data[i].y;
  fft(re, im, false);

  const half = Math.floor(N / 2);
  const psd = new Float64Array(half);
  let maxPsd = 1e-10;
  for (let k = 0; k < half; k++) {
    psd[k] = re[k] * re[k] + im[k] * im[k];
    if (psd[k] > maxPsd) maxPsd = psd[k];
  }

  const cx = k => pad.l + (k / (half - 1)) * pw;
  const cy = v => H - pad.b - (Math.log10(v / maxPsd + 1e-6) + 6) / 6 * ph;

  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = H - pad.b - (i / 4) * ph;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
  }

  // PSD bars
  for (let k = 1; k < half; k++) {
    const t = k / half;
    ctx.fillStyle = genColor(t, 0.55);
    const x = cx(k), y = cy(psd[k]);
    ctx.fillRect(x - 0.8, y, 1.6, H - pad.b - y);
  }

  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b);
  ctx.lineTo(W - pad.r, H - pad.b);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.font = "11px 'Courier New', monospace";
  ctx.fillText("PSD (log)", 4, pad.t + 10);
  ctx.fillText("freq →", W - pad.r - 40, H - 4);
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function GoldbachFractal() {
  const cometRef   = useRef(null);
  const phaseRef   = useRef(null);
  const specRef    = useRef(null);

  const [limit,       setLimit]       = useState(500);
  const [showEnv,     setShowEnv]     = useState(true);
  const [showPhase,   setShowPhase]   = useState(true);
  const [showSpec,    setShowSpec]    = useState(true);

  const computed = useMemo(() => computeData(limit), [limit]);

  useEffect(() => {
    const { data, envelope, phase } = computed;
    drawComet(cometRef.current, data, envelope, phase, { showEnv });
    if (showPhase) drawPhase(phaseRef.current, data, phase);
    if (showSpec)  drawSpectrum(specRef.current, data, envelope);
  }, [computed, showEnv, showPhase, showSpec]);

  const sectionLabel = txt => (
    <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(255,255,255,0.25)", marginBottom: 4, marginTop: 14 }}>
      {txt}
    </div>
  );

  return (
    <div style={{ background: "#020209", minHeight: "100vh", color: "#ccc", fontFamily: "'Courier New', monospace", padding: "28px 32px", boxSizing: "border-box" }}>

      {/* Header */}
      <div style={{ marginBottom: 18 }}>
        <div style={{ fontSize: 20, fontWeight: 700, letterSpacing: 3, color: "#e8c84a" }}>
          GOLDBACH FRACTAL
        </div>
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", letterSpacing: 2, marginTop: 4 }}>
          Bhagwanani generational framework · Hilbert analytic signal · power spectrum
        </div>
      </div>

      {/* Legend */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
        <span style={{ fontSize: 10, color: "rgba(255,255,255,0.35)", letterSpacing: 1 }}>APPEARANCE INDEX τ(2m)</span>
        <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
          {Array.from({ length: 30 }, (_, i) => (
            <div key={i} style={{ width: 7, height: 10, background: genColor(i / 29, 1) }} />
          ))}
        </div>
        <span style={{ fontSize: 10, color: "rgba(255,255,255,0.3)" }}>early genesis → late genesis</span>
        {showEnv && <>
          <div style={{ width: 28, height: 2, background: "#ffd030", boxShadow: "0 0 6px #ffd030", marginLeft: 12 }} />
          <span style={{ fontSize: 10, color: "rgba(255,255,255,0.3)" }}>Hilbert envelope</span>
        </>}
      </div>

      {/* Main comet */}
      {sectionLabel("GOLDBACH'S COMET  —  r₂(2m) vs 2m")}
      <canvas ref={cometRef} width={920} height={320}
        style={{ display: "block", borderRadius: 3, border: "1px solid rgba(255,255,255,0.06)" }} />

      {/* Phase plot */}
      {showPhase && <>
        {sectionLabel("INSTANTANEOUS PHASE  —  φ(m) = arctan( H{r₂}(m) / r₂(m) )")}
        <canvas ref={phaseRef} width={920} height={130}
          style={{ display: "block", borderRadius: 3, border: "1px solid rgba(255,255,255,0.06)" }} />
      </>}

      {/* Power spectrum */}
      {showSpec && <>
        {sectionLabel("POWER SPECTRAL DENSITY  —  |FFT(r₂)|²  (log scale)")}
        <canvas ref={specRef} width={920} height={130}
          style={{ display: "block", borderRadius: 3, border: "1px solid rgba(255,255,255,0.06)" }} />
      </>}

      {/* Controls */}
      <div style={{ display: "flex", gap: 28, alignItems: "center", marginTop: 20, flexWrap: "wrap" }}>
        <label style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 11 }}>
          <span style={{ color: "rgba(255,255,255,0.4)" }}>RANGE</span>
          <input type="range" min={100} max={1200} step={50} value={limit}
            onChange={e => setLimit(+e.target.value)}
            style={{ width: 130, accentColor: "#e8c84a" }} />
          <span style={{ color: "#e8c84a", minWidth: 42 }}>2m ≤ {limit}</span>
        </label>
        {[["showEnv", showEnv, setShowEnv, "ENVELOPE"],
          ["showPhase", showPhase, setShowPhase, "PHASE"],
          ["showSpec", showSpec, setShowSpec, "SPECTRUM"]].map(([, val, set, label]) => (
          <label key={label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, cursor: "pointer" }}>
            <input type="checkbox" checked={val} onChange={e => set(e.target.checked)}
              style={{ accentColor: "#e8c84a" }} />
            <span style={{ color: "rgba(255,255,255,0.45)", letterSpacing: 1 }}>{label}</span>
          </label>
        ))}
      </div>

      {/* Mathematical annotations */}
      <div style={{ marginTop: 22, fontSize: 10, color: "rgba(255,255,255,0.2)", lineHeight: 1.8, maxWidth: 760, letterSpacing: 0.5 }}>
        <div style={{ color: "rgba(255,255,255,0.35)", marginBottom: 6, letterSpacing: 2 }}>READING THE PLOTS</div>
        <div><span style={{ color: "rgba(255,255,255,0.35)" }}>Comet:</span> Each point = one even number 2m. Height = r₂(2m), the count of unordered odd-prime pairs (p,q) with p+q=2m. Colour encodes τ(2m), the generational appearance index from the Bhagwanani framework — the stage n at which 2m first enters S_n = P_n + P_n. The golden curve is the Hilbert analytic envelope, tracing the smooth attractor ≈ C·m/ln²m predicted by the Hardy–Littlewood circle method.</div>
        <div style={{ marginTop: 6 }}><span style={{ color: "rgba(255,255,255,0.35)" }}>Phase:</span> φ(m) = arg(r₂(m) + j·H{{"{r₂}(m)"}}) where H denotes the Hilbert transform. The quasi-periodic oscillation reflects the explicit formula for prime sums: r₂(2m) = S(2m)·2m/ln²m + Σ_ρ (error over zeta zeros ρ = ½+iγ). Each zero γ contributes a frequency to the phase plot — connecting the fractal oscillation directly to the Riemann ζ function.</div>
        <div style={{ marginTop: 6 }}><span style={{ color: "rgba(255,255,255,0.35)" }}>Spectrum:</span> |FFT(r₂)|² on log scale. A fractal signal would show power-law decay (straight line in log-log). The spectral structure encodes the distribution of prime gaps and the self-similar banding visible in the comet.</div>
      </div>
    </div>
  );
}
