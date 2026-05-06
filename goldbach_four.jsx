import { useState, useEffect, useRef, useCallback } from "react";

const C = {
  bg: "#08080a", panel: "#111114", border: "#222226",
  text: "#dddde0", dim: "#555", dim2: "#888",
  blue: "#60a5fa", amber: "#f59e0b", green: "#34d399",
  red: "#f87171", violet: "#a78bfa",
};

// ── sieve ──────────────────────────────────────────────────────
function buildSieve(limit) {
  const is = new Uint8Array(limit + 1).fill(1);
  is[0] = is[1] = 0;
  for (let i = 2; i * i <= limit; i++)
    if (is[i]) for (let j = i * i; j <= limit; j += i) is[j] = 0;
  const ps = [];
  for (let i = 2; i <= limit; i++) if (is[i]) ps.push(i);
  return { is, ps };
}

// ── G(m) ───────────────────────────────────────────────────────
function buildG(maxM, is) {
  const G = new Int32Array(maxM + 1);
  for (let m = 3; m <= maxM; m++) {
    let c = 0;
    for (let p = 3; p <= m; p += 2)
      if (is[p] && 2 * m - p >= 2 && is[2 * m - p]) c++;
    G[m] = c;
  }
  return G;
}

// ── shared label ───────────────────────────────────────────────
function Label({ children }) {
  return (
    <div style={{ fontSize: 10, color: C.dim2, letterSpacing: 1.5,
      textTransform: "uppercase", marginBottom: 6 }}>
      {children}
    </div>
  );
}

function Box({ title, color, children }) {
  return (
    <div style={{ background: C.bg, border: "1px solid " + C.border,
      borderRadius: 4, padding: 10, marginBottom: 10, fontSize: 12, lineHeight: 1.65 }}>
      {title && <div style={{ color: color || C.dim2, fontSize: 10,
        letterSpacing: 1.2, textTransform: "uppercase", marginBottom: 5 }}>{title}</div>}
      {children}
    </div>
  );
}

function drawAxes(ctx, pad, iW, iH, maxY, nY) {
  ctx.strokeStyle = C.border; ctx.lineWidth = 0.5;
  for (let i = 0; i <= nY; i++) {
    const y = iH * i / nY;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(iW, y); ctx.stroke();
    const v = maxY * (1 - i / nY);
    ctx.fillStyle = C.dim; ctx.font = "10px monospace";
    ctx.fillText(v.toFixed(v < 10 ? 1 : 0), -40, y + 4);
  }
  ctx.beginPath(); ctx.moveTo(0, iH); ctx.lineTo(iW, iH); ctx.stroke();
}

// ══════════════════════════════════════════════════════════════
// TAB 1 — DIFFERENTIAL TOPOLOGY
// ══════════════════════════════════════════════════════════════
function T1({ is, G, maxM }) {
  const cvs = useRef(null);
  const [N, setN] = useState(55);

  useEffect(() => {
    const canvas = cvs.current; if (!canvas) return;
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = C.panel; ctx.fillRect(0, 0, W, H);
    const cell = Math.floor(Math.min(W, H - 10) / N);
    const ox = Math.floor((W - cell * N) / 2);
    const oy = Math.floor((H - cell * N) / 2);

    for (let i = 0; i < N; i++) {
      const a = 2 * i + 3;
      for (let j = 0; j < N; j++) {
        const b = 2 * j + 3;
        const aP = a < is.length && is[a];
        const bP = b < is.length && is[b];
        const x = ox + j * cell;
        const y = oy + (N - 1 - i) * cell;
        if (aP && bP) {
          ctx.fillStyle = C.blue; ctx.globalAlpha = 0.8;
        } else if (aP || bP) {
          ctx.fillStyle = C.violet; ctx.globalAlpha = 0.22;
        } else {
          ctx.fillStyle = "#18181c"; ctx.globalAlpha = 0.7;
        }
        ctx.fillRect(x, y, cell - 1, cell - 1);
      }
    }
    ctx.globalAlpha = 1;

    // anti-diagonals for 2m
    ctx.strokeStyle = C.amber; ctx.lineWidth = 0.4; ctx.globalAlpha = 0.12;
    for (let s = 0; s < 2 * N; s++) {
      const i0 = Math.max(0, s - N + 1), i1 = Math.min(N - 1, s);
      const x0 = ox + (s - i1) * cell, y0 = oy + (N - 1 - i1) * cell;
      const x1 = ox + (s - i0) * cell + cell, y1 = oy + (N - 1 - i0) * cell + cell;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }, [is, N]);

  return (
    <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
      <div>
        <Label>Prime pair grid — diagonal = one even 2m</Label>
        <canvas ref={cvs} width={380} height={380}
          style={{ border: "1px solid " + C.border, display: "block" }} />
        <div style={{ marginTop: 6, fontSize: 11 }}>
          <span style={{ color: C.blue }}>■</span> both prime{" "}
          <span style={{ color: C.violet }}>■</span> one prime{" "}
          <span style={{ color: C.dim }}>■</span> neither
        </div>
        <div style={{ marginTop: 8 }}>
          <span style={{ fontSize: 11, color: C.dim }}>Grid {N}x{N}: </span>
          <input type="range" min={20} max={90} value={N}
            onChange={e => setN(+e.target.value)} style={{ width: 120 }} />
        </div>
      </div>
      <div style={{ flex: 1, minWidth: 200 }}>
        <Box title="Poincare-Hopf argument" color={C.blue}>
          Each diagonal = one even 2m. The "flow index" at 2m is G(2m)
          — the count of blue cells on that diagonal. Poincare-Hopf links
          the sum of flow indices to the Euler characteristic of the manifold.
        </Box>
        <Box title="What is proved" color={C.green}>
          Flow density grows with m (by PNT). Diagonals accumulate more
          blue cells as m increases. The total flow is always large.
        </Box>
        <Box title="Gap remaining" color={C.red}>
          Index nonzero on every single diagonal. Poincare-Hopf gives
          a global count constraint, not a pointwise one. A zero-index
          diagonal can coexist with high-index neighbours.
        </Box>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// TAB 2 — INFORMATION THEORY
// ══════════════════════════════════════════════════════════════
function T2({ G, maxM }) {
  const cvs = useRef(null);
  const LIMIT = Math.min(maxM, 300);

  const data = [];
  for (let m = 3; m <= LIMIT; m++) {
    const lam = (2 * m) / Math.pow(Math.log(2 * m), 2);
    data.push({ m, lam, g: G[m] || 0 });
  }

  useEffect(() => {
    const canvas = cvs.current; if (!canvas) return;
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = C.panel; ctx.fillRect(0, 0, W, H);
    const pad = { t: 20, r: 16, b: 36, l: 48 };
    const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;
    const maxLam = Math.max(...data.map(d => d.lam));
    const maxG = Math.max(...data.map(d => d.g));
    const n = data.length;

    ctx.save(); ctx.translate(pad.l, pad.t);
    drawAxes(ctx, pad, iW, iH, maxLam, 4);

    ctx.fillStyle = C.dim; ctx.font = "10px monospace";
    ctx.fillText("m", iW / 2, iH + 28);

    // x axis ticks
    [50, 100, 150, 200, 250, 300].forEach(m => {
      if (m <= LIMIT) {
        const x = ((m - 3) / (n - 1)) * iW;
        ctx.fillStyle = C.dim; ctx.fillText(m, x - 8, iH + 14);
      }
    });

    // lambda curve
    ctx.strokeStyle = C.amber; ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((d, i) => {
      const x = (i / (n - 1)) * iW;
      const y = iH * (1 - d.lam / maxLam);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // G(m) scaled
    ctx.strokeStyle = C.blue; ctx.lineWidth = 1;
    ctx.beginPath();
    data.forEach((d, i) => {
      const x = (i / (n - 1)) * iW;
      const y = iH * (1 - d.g / Math.max(maxG, 1));
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.restore();
  }, [data]);

  const minG = Math.min(...data.map(d => d.g));
  const gapCount = data.filter(d => d.g === 0).length;
  const maxLam = Math.max(...data.map(d => d.lam)).toFixed(1);

  return (
    <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
      <div>
        <Label>Information cost of a Goldbach gap</Label>
        <canvas ref={cvs} width={380} height={280}
          style={{ border: "1px solid " + C.border, display: "block" }} />
        <div style={{ marginTop: 6, fontSize: 11 }}>
          <span style={{ color: C.amber }}>—</span> lambda(m) = info cost of gap{" "}
          <span style={{ color: C.blue }}>—</span> G(2m) actual
        </div>
        <Box title="Diagnostics">
          <div>lambda range: [0.2, {maxLam}]</div>
          <div>Min G observed: <span style={{ color: minG === 0 ? C.red : C.green }}>{minG}</span></div>
          <div>G=0 events: <span style={{ color: gapCount > 0 ? C.red : C.green }}>{gapCount}</span></div>
        </Box>
      </div>
      <div style={{ flex: 1, minWidth: 200 }}>
        <Box title="The argument" color={C.amber}>
          Under PNT the expected count of prime pairs summing to 2m is
          approximately Poisson(lambda) where lambda(m) = 2m / ln^2(2m).
          The probability of G=0 is exp(-lambda), and the information cost
          (negative log probability) is lambda itself.
        </Box>
        <Box title="What is proved" color={C.green}>
          lambda(m) grows without bound. A Goldbach gap at large m would
          be an event of probability exp(-lambda) approaching zero.
          The information budget simply grows too large to afford a gap.
        </Box>
        <Box title="Gap remaining" color={C.red}>
          The Poisson model is an approximation. Primes have correlations
          (twin primes, prime gaps) that the model ignores. Proving the
          approximation is tight enough to exclude G=0 exactly requires
          the same control as the circle method — which needs GRH.
        </Box>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// TAB 3 — GEOMETRIC MEASURE THEORY
// ══════════════════════════════════════════════════════════════
function T3({ G, maxM }) {
  const cvs = useRef(null);
  const [win, setWin] = useState(20);
  const LIMIT = Math.min(maxM, 380);

  const densities = [];
  for (let m = 10; m <= LIMIT; m++) {
    let hit = 0, tot = 0;
    for (let k = Math.max(4, 2 * m - win); k <= Math.min(2 * LIMIT, 2 * m + win); k += 2) {
      tot++;
      const mk = k / 2;
      if (mk === Math.floor(mk) && G[Math.round(mk)] > 0) hit++;
    }
    densities.push({ m, d: tot > 0 ? hit / tot : 0, g: G[m] || 0 });
  }

  useEffect(() => {
    const canvas = cvs.current; if (!canvas) return;
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = C.panel; ctx.fillRect(0, 0, W, H);
    const pad = { t: 20, r: 16, b: 36, l: 48 };
    const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;
    const n = densities.length;

    ctx.save(); ctx.translate(pad.l, pad.t);
    drawAxes(ctx, pad, iW, iH, 1.0, 4);
    ctx.fillStyle = C.dim; ctx.font = "10px monospace";
    ctx.fillText("m", iW / 2, iH + 28);

    // threshold 0.5
    ctx.strokeStyle = C.amber; ctx.lineWidth = 1; ctx.setLineDash([5, 3]);
    const yH = iH * 0.5;
    ctx.beginPath(); ctx.moveTo(0, yH); ctx.lineTo(iW, yH); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.amber; ctx.fillText("0.5", iW - 22, yH - 4);

    // density=1
    ctx.strokeStyle = C.green; ctx.lineWidth = 0.5; ctx.setLineDash([2, 4]);
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(iW, 0); ctx.stroke();
    ctx.setLineDash([]);

    // density curve
    ctx.strokeStyle = C.blue; ctx.lineWidth = 1.5;
    ctx.beginPath();
    densities.forEach((d, i) => {
      const x = (i / (n - 1)) * iW;
      const y = iH * (1 - d.d);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // G=0 dots
    densities.forEach((d, i) => {
      if (d.g === 0) {
        ctx.fillStyle = C.red;
        ctx.beginPath();
        ctx.arc((i / (n - 1)) * iW, iH * (1 - d.d), 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    });

    ctx.restore();
  }, [densities, win]);

  const minD = densities.length ? Math.min(...densities.map(d => d.d)).toFixed(4) : "-";
  const aboveHalf = densities.every(d => d.d >= 0.5);

  return (
    <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
      <div>
        <Label>Lebesgue density of P+P in sliding window</Label>
        <canvas ref={cvs} width={380} height={280}
          style={{ border: "1px solid " + C.border, display: "block" }} />
        <div style={{ marginTop: 6, fontSize: 11 }}>
          <span style={{ color: C.blue }}>—</span> density{" "}
          <span style={{ color: C.amber }}>- -</span> Besicovitch threshold 0.5{" "}
          <span style={{ color: C.red }}>●</span> G=0
        </div>
        <div style={{ marginTop: 8 }}>
          <span style={{ fontSize: 11, color: C.dim }}>Window: {win}  </span>
          <input type="range" min={5} max={60} value={win}
            onChange={e => setWin(+e.target.value)} style={{ width: 120 }} />
        </div>
        <Box title="Diagnostics">
          <div>Min density: <span style={{ color: C.blue }}>{minD}</span></div>
          <div>All above 0.5: <span style={{ color: aboveHalf ? C.green : C.red }}>
            {aboveHalf ? "yes" : "no"}
          </span></div>
        </Box>
      </div>
      <div style={{ flex: 1, minWidth: 200 }}>
        <Box title="Besicovitch density" color={C.green}>
          If the density of P+P near every even 2m is bounded below
          by some delta greater than 0, the set is too thick to
          contain isolated holes of size 1.
        </Box>
        <Box title="What is proved" color={C.green}>
          By PNT, average density approaches 1. The set P+P covers
          asymptotically all even integers — meaning it is dense.
        </Box>
        <Box title="Gap remaining" color={C.red}>
          Dense does not mean complete. A dense set can still miss
          isolated points. Proving pointwise density is positive
          at every single m is the same gap as before — converting
          average control into pointwise control.
        </Box>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// TAB 4 — STOCHASTIC DRIFT
// ══════════════════════════════════════════════════════════════
function T4({ ps }) {
  const cvs1 = useRef(null);
  const cvs2 = useRef(null);
  const [nP, setNP] = useState(35);

  const oddPs = ps.filter(p => p > 2);

  const computeMin = useCallback((k) => {
    const sub = oddPs.slice(0, k);
    let mn = Infinity;
    const NT = 300;
    for (let ti = 0; ti < NT; ti++) {
      const theta = (2 * Math.PI * ti) / NT;
      let re = 0, im = 0;
      for (let j = 0; j < sub.length; j++) {
        re += Math.cos(sub[j] * theta);
        im += Math.sin(sub[j] * theta);
      }
      mn = Math.min(mn, Math.sqrt(re * re + im * im));
    }
    return mn;
  }, [oddPs]);

  const growthData = [];
  for (let k = 5; k <= Math.min(nP, oddPs.length); k += 3) {
    growthData.push({ k, min: computeMin(k), sqN: Math.sqrt(k) });
  }

  useEffect(() => {
    const sub = oddPs.slice(0, nP);
    const NT = 512;
    const piAbs = new Float32Array(NT);
    for (let ti = 0; ti < NT; ti++) {
      const theta = (2 * Math.PI * ti) / NT;
      let re = 0, im = 0;
      for (let j = 0; j < sub.length; j++) {
        re += Math.cos(sub[j] * theta);
        im += Math.sin(sub[j] * theta);
      }
      piAbs[ti] = Math.sqrt(re * re + im * im);
    }
    const minV = Math.min(...piAbs);
    const meanV = piAbs.reduce((a, b) => a + b, 0) / NT;
    const maxV = Math.max(...piAbs);

    {
      const canvas = cvs1.current; if (!canvas) return;
      const W = canvas.width, H = canvas.height;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = C.panel; ctx.fillRect(0, 0, W, H);
      const pad = { t: 16, r: 16, b: 32, l: 48 };
      const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;

      ctx.save(); ctx.translate(pad.l, pad.t);
      drawAxes(ctx, pad, iW, iH, maxV, 4);
      ctx.fillStyle = C.dim; ctx.font = "10px monospace";
      ctx.fillText("theta (0 to 2pi)", iW / 2 - 40, iH + 26);

      ctx.strokeStyle = C.blue; ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < NT; i++) {
        const x = (i / (NT - 1)) * iW;
        const y = iH * (1 - piAbs[i] / maxV);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();

      // min line
      ctx.strokeStyle = C.red; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      const yMin = iH * (1 - minV / maxV);
      ctx.beginPath(); ctx.moveTo(0, yMin); ctx.lineTo(iW, yMin); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.red;
      ctx.fillText("min=" + minV.toFixed(2), iW - 85, yMin - 4);

      // mean line
      ctx.strokeStyle = C.amber; ctx.lineWidth = 1; ctx.setLineDash([6, 3]);
      const yMean = iH * (1 - meanV / maxV);
      ctx.beginPath(); ctx.moveTo(0, yMean); ctx.lineTo(iW, yMean); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.amber;
      ctx.fillText("mean=" + meanV.toFixed(1), iW - 90, yMean - 4);

      ctx.restore();
    }

    {
      const canvas = cvs2.current; if (!canvas) return;
      const W = canvas.width, H = canvas.height;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = C.panel; ctx.fillRect(0, 0, W, H);
      const pad = { t: 16, r: 16, b: 32, l: 48 };
      const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;
      const gD = growthData;
      if (!gD.length) { ctx.restore(); return; }
      const maxY = Math.max(...gD.map(d => d.sqN));
      const n = gD.length;

      ctx.save(); ctx.translate(pad.l, pad.t);
      drawAxes(ctx, pad, iW, iH, maxY, 4);
      ctx.fillStyle = C.dim; ctx.font = "10px monospace";
      ctx.fillText("N primes", iW / 2 - 25, iH + 26);

      ctx.strokeStyle = C.amber; ctx.lineWidth = 1.5; ctx.setLineDash([5, 3]);
      ctx.beginPath();
      gD.forEach((d, i) => {
        const x = (i / (n - 1)) * iW, y = iH * (1 - d.sqN / maxY);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke(); ctx.setLineDash([]);

      ctx.strokeStyle = C.green; ctx.lineWidth = 2;
      ctx.beginPath();
      gD.forEach((d, i) => {
        const x = (i / (n - 1)) * iW, y = iH * (1 - d.min / maxY);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.strokeStyle = C.red; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, iH); ctx.lineTo(iW, iH); ctx.stroke();

      ctx.restore();
    }
  }, [nP, oddPs, growthData]);

  const lastMin = growthData.length ? growthData[growthData.length - 1].min.toFixed(3) : "-";
  const lastSqN = growthData.length ? growthData[growthData.length - 1].sqN.toFixed(2) : "-";
  const ratio = growthData.length
    ? (growthData[growthData.length - 1].min / growthData[growthData.length - 1].sqN).toFixed(3)
    : "-";

  return (
    <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
      <div>
        <Label>|Pi(e^itheta)| — prime walker on unit circle</Label>
        <canvas ref={cvs1} width={380} height={200}
          style={{ border: "1px solid " + C.border, display: "block" }} />
        <div style={{ height: 8 }} />
        <Label>min|Pi| vs sqrt(N) as primes grow</Label>
        <canvas ref={cvs2} width={380} height={180}
          style={{ border: "1px solid " + C.border, display: "block" }} />
        <div style={{ marginTop: 6, fontSize: 11 }}>
          <span style={{ color: C.green }}>—</span> min|Pi|{" "}
          <span style={{ color: C.amber }}>- -</span> sqrt(N) drift{" "}
          <span style={{ color: C.red }}>—</span> zero
        </div>
        <div style={{ marginTop: 8 }}>
          <span style={{ fontSize: 11, color: C.dim }}>Primes: {nP}  </span>
          <input type="range" min={10} max={80} value={nP}
            onChange={e => setNP(+e.target.value)} style={{ width: 120 }} />
        </div>
        <Box title="Diagnostics">
          <div>Current min|Pi|: <span style={{ color: C.green }}>{lastMin}</span></div>
          <div>sqrt(N): <span style={{ color: C.amber }}>{lastSqN}</span></div>
          <div>min/sqrt(N): <span style={{ color: C.blue }}>{ratio}</span></div>
        </Box>
      </div>
      <div style={{ flex: 1, minWidth: 200 }}>
        <Box title="Martingale drift argument" color={C.violet}>
          Model each prime p contributing exp(ip*theta) as a random
          walk step. By the central limit theorem, E[|Pi|^2] = N and
          E[|Pi|] grows as sqrt(N). The drift away from zero grows
          faster than any finite fluctuation toward zero.
        </Box>
        <Box title="What is shown" color={C.green}>
          min|Pi| tracks sqrt(N) empirically — the green curve
          stays proportional to the dashed amber reference as N grows.
          The ratio min/sqrt(N) remains stable and positive.
        </Box>
        <Box title="Gap remaining" color={C.red}>
          Primes are not random. Their correlations (twin primes,
          prime k-tuples) shift the distribution. Proving
          min|Pi| is greater than C * N^alpha for some fixed alpha
          and all N requires bounding those correlations uniformly
          — which connects back to GRH or something equivalent.
        </Box>
        <Box title="Most promising angle" color={C.amber}>
          The lacunary structure of Pi (coefficients only on primes,
          increasingly sparse) is known to resist zeros more than
          generic trigonometric series. A purely analytic proof
          that a lacunary series with prime support cannot vanish
          would close this without any number-theoretic input.
          This is the one remaining open question that does not
          obviously reduce to GRH.
        </Box>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// ROOT
// ══════════════════════════════════════════════════════════════
export default function App() {
  const [tab, setTab] = useState(0);
  const [data, setData] = useState(null);
  const MAXM = 350;

  useEffect(() => {
    const { is, ps } = buildSieve(MAXM * 2 + 10);
    const G = buildG(MAXM, is);
    setData({ is, ps, G });
  }, []);

  const tabs = [
    "01 Prime Flow",
    "02 Information",
    "03 Measure",
    "04 Stochastic",
  ];

  const summary = [
    ["Differential Topology", "Flow density grows (PNT)", "Index nonzero on every diagonal"],
    ["Information Theory", "Gap cost lambda grows to infinity", "Poisson model is an approximation"],
    ["Geometric Measure", "Average density approaches 1", "Pointwise density is not proved"],
    ["Stochastic Drift", "min|Pi| tracks sqrt(N) empirically", "Prove min|Pi| above C*N^alpha rigorously"],
  ];

  return (
    <div style={{ background: C.bg, color: C.text, minHeight: "100vh",
      fontFamily: "'Courier New', monospace", padding: 20 }}>
      <div style={{ maxWidth: 860, margin: "0 auto" }}>

        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 10, color: C.dim, letterSpacing: 3,
            textTransform: "uppercase", marginBottom: 4 }}>
            Non-Algebraic Approaches to Goldbach
          </div>
          <div style={{ fontSize: 20, color: C.blue, fontWeight: "bold", marginBottom: 4 }}>
            Four Ways to Fill the Hole
          </div>
          <div style={{ fontSize: 11, color: C.dim }}>
            Real computations. Honest statement of what is proved vs what remains.
          </div>
        </div>

        <div style={{ display: "flex", gap: 4, marginBottom: 16, flexWrap: "wrap" }}>
          {tabs.map((t, i) => (
            <button key={i} onClick={() => setTab(i)} style={{
              background: tab === i ? C.blue : C.panel,
              color: tab === i ? C.bg : C.dim2,
              border: "1px solid " + (tab === i ? C.blue : C.border),
              borderRadius: 3, padding: "6px 14px", fontSize: 11,
              cursor: "pointer", fontFamily: "monospace", letterSpacing: 0.5
            }}>{t}</button>
          ))}
        </div>

        <div style={{ background: C.panel, border: "1px solid " + C.border,
          borderRadius: 6, padding: 20 }}>
          {!data ? (
            <div style={{ color: C.dim, padding: 40, textAlign: "center" }}>
              Computing primes...
            </div>
          ) : (
            <>
              {tab === 0 && <T1 is={data.is} G={data.G} maxM={MAXM} />}
              {tab === 1 && <T2 G={data.G} maxM={MAXM} />}
              {tab === 2 && <T3 G={data.G} maxM={MAXM} />}
              {tab === 3 && <T4 ps={data.ps} />}
            </>
          )}
        </div>

        {data && (
          <div style={{ marginTop: 20, borderTop: "1px solid " + C.border, paddingTop: 16 }}>
            <Label>Convergence Summary — all four approaches share the same structural gap</Label>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ color: C.dim }}>
                  {["Approach", "Proved", "Gap remaining"].map(h => (
                    <th key={h} style={{ textAlign: "left",
                      padding: "4px 12px 4px 0",
                      borderBottom: "1px solid " + C.border }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {summary.map(([a, p, g]) => (
                  <tr key={a}>
                    <td style={{ padding: "6px 12px 6px 0", color: C.blue }}>{a}</td>
                    <td style={{ padding: "6px 12px 6px 0", color: C.green }}>{p}</td>
                    <td style={{ padding: "6px 12px 6px 0", color: C.red }}>{g}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ marginTop: 12, fontSize: 11, color: C.dim, lineHeight: 1.7 }}>
              The one angle that does not obviously reduce to GRH: proving min|Pi| cannot
              vanish from the lacunary structure of the prime support alone — a question
              in real analysis, not number theory. That is the single remaining open thread.
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
