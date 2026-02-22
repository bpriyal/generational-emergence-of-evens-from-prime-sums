import { useRef, useEffect, useState } from "react";

function fft(re, im, invert) {
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
      let cRe = 1, cIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const uRe = re[i+j], uIm = im[i+j];
        const vRe = re[i+j+len/2]*cRe - im[i+j+len/2]*cIm;
        const vIm = re[i+j+len/2]*cIm + im[i+j+len/2]*cRe;
        re[i+j] = uRe+vRe; im[i+j] = uIm+vIm;
        re[i+j+len/2] = uRe-vRe; im[i+j+len/2] = uIm-vIm;
        const nr = cRe*wRe - cIm*wIm;
        cIm = cRe*wIm + cIm*wRe; cRe = nr;
      }
    }
  }
  if (invert) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

function compute(LIMIT) {
  const sieve = new Uint8Array(LIMIT + 1).fill(1);
  sieve[0] = sieve[1] = 0;
  for (let i = 2; i * i <= LIMIT; i++)
    if (sieve[i]) for (let j = i*i; j <= LIMIT; j += i) sieve[j] = 0;

  const oddP = [];
  for (let i = 3; i <= LIMIT; i += 2) if (sieve[i]) oddP.push(i);

  const signal = [];
  let minG = Infinity;
  for (let m = 3; m * 2 <= LIMIT; m++) {
    let g = 0;
    for (const pn of oddP) {
      if (pn > m) break;
      const q = 2 * m - pn;
      if (q >= 2 && q <= LIMIT && sieve[q]) g++;
    }
    signal.push(g);
    if (g < minG) minG = g;
  }

  const n = signal.length;
  let N = 1; while (N < n) N <<= 1;
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  for (let i = 0; i < n; i++) re[i] = signal[i];
  fft(re, im, false);
  for (let k = 1; k < N/2; k++) { re[k] *= 2; im[k] *= 2; }
  for (let k = N/2+1; k < N; k++) { re[k] = 0; im[k] = 0; }
  fft(re, im, true);

  let maxA = 0;
  for (let i = 0; i < n; i++) {
    const a = Math.sqrt(re[i]*re[i] + im[i]*im[i]);
    if (a > maxA) maxA = a;
  }

  const pts = [];
  let minA = 1;
  for (let i = 0; i < n; i++) {
    const A = Math.sqrt(re[i]*re[i] + im[i]*im[i]) / maxA;
    const phi = Math.atan2(im[i], re[i]);
    const lat = (Math.PI / 2) * A;
    const lon = phi;
    const sx = Math.cos(lat) * Math.cos(lon);
    const sy = Math.cos(lat) * Math.sin(lon);
    const sz = Math.sin(lat);
    const tilt = Math.PI / 6;
    const px = sx;
    const py = sy * Math.cos(tilt) - sz * Math.sin(tilt);
    if (A < minA) minA = A;
    pts.push({ px, py, A, phi, g: signal[i], m: i + 3 });
  }

  return { pts, minG, minA: minA.toFixed(4), n };
}

function draw(canvas, pts, LIMIT) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#0a0a0a";
  ctx.fillRect(0, 0, W, H);

  const cx = W / 2, cy = H / 2;
  const R = Math.min(W, H) / 2 - 40;

  // wireframe
  ctx.strokeStyle = "#1e1e1e";
  ctx.lineWidth = 1;
  for (let lat = -Math.PI/3; lat <= Math.PI/2; lat += Math.PI/6) {
    const r2d = R * Math.cos(lat);
    const yOff = -R * Math.sin(lat) * Math.cos(Math.PI/6);
    ctx.beginPath();
    ctx.ellipse(cx, cy + yOff, r2d, r2d * Math.sin(Math.PI/6), 0, 0, Math.PI*2);
    ctx.stroke();
  }
  for (let lon = 0; lon < Math.PI; lon += Math.PI/6) {
    ctx.beginPath();
    ctx.ellipse(cx, cy, R * Math.abs(Math.cos(lon)), R * Math.cos(Math.PI/6), lon, 0, Math.PI*2);
    ctx.stroke();
  }

  // equator — the danger zone
  ctx.strokeStyle = "#f44";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.ellipse(cx, cy, R, R * Math.sin(Math.PI/6), 0, 0, Math.PI*2);
  ctx.stroke();
  ctx.setLineDash([]);

  // trajectory
  const total = pts.length;
  for (let i = 1; i < total; i++) {
    const p = pts[i-1], q = pts[i];
    const t = i / total;
    ctx.strokeStyle = `hsla(${t * 260}, 80%, 60%, 0.7)`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx + p.px * R, cy + p.py * R);
    ctx.lineTo(cx + q.px * R, cy + q.py * R);
    ctx.stroke();
  }

  // mark point with lowest envelope (closest to equator)
  let minIdx = 0, minA = 1;
  for (let i = 0; i < pts.length; i++) if (pts[i].A < minA) { minA = pts[i].A; minIdx = i; }
  const mp = pts[minIdx];
  ctx.fillStyle = "#ff0";
  ctx.beginPath();
  ctx.arc(cx + mp.px * R, cy + mp.py * R, 5, 0, Math.PI*2);
  ctx.fill();

  // labels
  ctx.fillStyle = "#888";
  ctx.font = "11px monospace";
  ctx.fillText(`2m = ${(minIdx + 3)*2}  closest to equator (min A=${minA.toFixed(3)})`, cx + mp.px * R + 8, cy + mp.py * R);
  ctx.fillStyle = "#f44";
  ctx.fillText("← equator = Goldbach gap (must never be touched)", cx - R + 4, cy + R * Math.sin(Math.PI/6) + 14);
  ctx.fillStyle = "#888";
  ctx.fillText(`G(2m) on Hilbert Sphere  |  ${total} even numbers  |  2m up to ${LIMIT}`, 8, 18);
}

export default function HilbertSphere() {
  const canvasRef = useRef(null);
  const [limit, setLimit] = useState(1000);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    const { pts, minG, minA, n } = compute(limit);
    setStats({ minG, minA, n });
    draw(canvasRef.current, pts, limit);
  }, [limit]);

  return (
    <div style={{ background: "#111", padding: 20, fontFamily: "monospace", color: "#ccc" }}>
      <canvas ref={canvasRef} width={560} height={560} style={{ display: "block" }} />

      <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 16 }}>
        <span style={{ fontSize: 11, color: "#666" }}>2m up to:</span>
        <input type="range" min={200} max={5000} step={200} value={limit}
          onChange={e => setLimit(+e.target.value)}
          style={{ width: 200, accentColor: "#4af" }} />
        <span style={{ fontSize: 12, color: "#4af" }}>{limit}</span>
      </div>

      {stats && (
        <div style={{ marginTop: 12, fontSize: 11, color: "#666", lineHeight: 2 }}>
          <div>samples computed: <span style={{ color: "#aaa" }}>{stats.n} even numbers</span></div>
          <div>min G(2m) in range: <span style={{ color: "#ff0" }}>{stats.minG}</span>
            &nbsp;← if this ever hit 0, Goldbach is false</div>
          <div>min envelope A(m): <span style={{ color: "#ff0" }}>{stats.minA}</span>
            &nbsp;← yellow dot on sphere, closest to equator</div>
          <div style={{ color: "#555", marginTop: 6 }}>
            The trajectory staying above the equator for these {stats.n} samples does NOT prove
            Goldbach. It replicates what direct verification already knows. A proof requires showing
            A(m) &gt; 0 analytically for all m — that is still open.
          </div>
        </div>
      )}
    </div>
  );
}
