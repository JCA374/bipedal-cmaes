// In-browser replay of an exported BipedalWalker episode (experiments/export_episode.py).
// Draws terrain, hull, both legs, the lidar rays and foot contacts from recorded geometry.

function setupCanvas(cv, w, h) {
  const dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr;
  cv.style.aspectRatio = `${w} / ${h}`;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

class WalkerReplay {
  constructor(canvas, opts = {}) {
    this.cv = canvas;
    this.W = 720; this.H = 320;
    this.ctx = setupCanvas(canvas, this.W, this.H);
    this.scale = opts.scale || 40;        // px per world unit
    this.lidar = opts.lidar !== false;
    this.onFrame = opts.onFrame || (() => {});
    this.speed = 1;
    this.i = 0; this.playing = false; this.ep = null;
    window.addEventListener("themechange", () => this.draw());
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.draw());
  }

  async load(url) {
    this.ep = await (await fetch(url)).json();
    this.i = 0;
    this.draw();
    return this.ep;
  }

  get n() { return this.ep ? this.ep.frames.length : 0; }

  play() {
    if (!this.ep || this.playing) return;
    if (this.i >= this.n - 1) this.i = 0;
    this.playing = true;
    let last = performance.now(), acc = 0;
    const fps = 50 / this.ep.every;  // real time = 50 env steps per second
    const tick = now => {
      if (!this.playing) return;
      acc += (now - last) / 1000 * fps * this.speed; last = now;
      const step = Math.floor(acc);
      if (step > 0) { acc -= step; this.seek(Math.min(this.n - 1, this.i + step)); }
      if (this.i >= this.n - 1) { this.playing = false; this.onEnd && this.onEnd(); return; }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  pause() { this.playing = false; }

  seek(i) { this.i = Math.max(0, Math.min(this.n - 1, i)); this.draw(); }

  draw() {
    const { ctx, W, H, ep } = this;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = css("--sky"); ctx.fillRect(0, 0, W, H);
    if (!ep) return;
    const f = ep.frames[this.i];
    const hull = f.p[0];
    let hx = 0, hy = 0;
    for (let k = 0; k < hull.length; k += 2) { hx += hull[k]; hy += hull[k + 1]; }
    hx /= hull.length / 2; hy /= hull.length / 2;
    const s = this.scale, camX = hx - W / s / 3, camY = 2.6;
    const X = x => (x - camX) * s, Y = y => H - (y - camY) * s;

    // finish line (x ≈ 88.7)
    const fin = (200 - 10) * (14 / 30);
    if (X(fin) > 0 && X(fin) < W) {
      ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2; ctx.setLineDash([6, 5]);
      ctx.beginPath(); ctx.moveTo(X(fin), 0); ctx.lineTo(X(fin), H); ctx.stroke(); ctx.setLineDash([]);
    }
    // terrain
    const [tx, ty] = ep.terrain;
    ctx.fillStyle = css("--ground"); ctx.strokeStyle = css("--ground-line"); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(X(tx[0]), H);
    for (let k = 0; k < tx.length; k++) ctx.lineTo(X(tx[k]), Y(ty[k]));
    ctx.lineTo(X(tx[tx.length - 1]), H); ctx.closePath(); ctx.fill();
    ctx.beginPath();
    for (let k = 0; k < tx.length; k++) (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(tx[k]), Y(ty[k]));
    ctx.stroke();
    // distance ticks every 10 units
    ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui";
    for (let m = 0; m <= 90; m += 10) if (X(m) > 0 && X(m) < W) ctx.fillText(m, X(m) - 6, H - 6);

    // lidar
    if (this.lidar) {
      ctx.strokeStyle = "rgba(235,104,52,.45)"; ctx.fillStyle = "rgba(235,104,52,.9)"; ctx.lineWidth = 1;
      for (let k = 0; k < 10; k++) {
        const a = 1.5 * k / 10, frac = f.o[14 + k], L = frac * 160 / 30;
        const ex = hx + Math.sin(a) * L, ey = hy - Math.cos(a) * L;
        ctx.beginPath(); ctx.moveTo(X(hx), Y(hy)); ctx.lineTo(X(ex), Y(ey)); ctx.stroke();
        if (frac < 1) { ctx.beginPath(); ctx.arc(X(ex), Y(ey), 2.5, 0, 7); ctx.fill(); }
      }
    }
    // legs: p[1..2] = leg 1 (upper, lower), p[3..4] = leg 2. Leg 2 drawn behind.
    const poly = (pts, fill, stroke) => {
      ctx.beginPath();
      for (let k = 0; k < pts.length; k += 2) (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(pts[k]), Y(pts[k + 1]));
      ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = stroke; ctx.lineWidth = 1.2; ctx.stroke();
    };
    const c1 = css("--s1"), c2 = css("--s2"), txt = css("--text");
    poly(f.p[3], c2, txt); poly(f.p[4], c2, txt);
    poly(hull, css("--accent"), txt);
    poly(f.p[1], c1, txt); poly(f.p[2], c1, txt);
    // contact markers on the feet
    const foot = (pts, on, col) => {
      if (!on) return;
      let bx = 0, by = 1e9;
      for (let k = 0; k < pts.length; k += 2) if (pts[k + 1] < by) { by = pts[k + 1]; bx = pts[k]; }
      ctx.beginPath(); ctx.arc(X(bx), Y(by), 5, 0, 7); ctx.fillStyle = col; ctx.fill();
      ctx.strokeStyle = txt; ctx.stroke();
    };
    foot(f.p[2], f.o[8] > 0.5, c1); foot(f.p[4], f.o[13] > 0.5, c2);

    // HUD
    ctx.fillStyle = txt; ctx.font = "600 13px system-ui";
    const step = this.i * ep.every;
    ctx.fillText(`step ${step} / 1600   x ${f.x.toFixed(1)}   reward ${f.r.toFixed(1)}`, 10, 18);
    if (this.i === this.n - 1 && ep.fell) {
      ctx.fillStyle = css("--red"); ctx.font = "700 22px system-ui";
      ctx.fillText("FELL  (−100)", W / 2 - 60, 60);
    }
    this.onFrame(f, this.i);
  }
}

// Horizontal-ish vertical bars for a vector, values in [-1, 1].
function drawBars(cv, ctx, values, labels, colors, opts = {}) {
  const W = cv.width / (window.devicePixelRatio || 1), H = cv.height / (window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, W, H);
  const n = values.length, pad = opts.pad || 4, lab = opts.labelW || 0;
  const rowH = (H - 4) / n;
  const x0 = lab + (W - lab) / 2, half = (W - lab) / 2 - 6;
  ctx.font = `${Math.min(12, rowH - 2)}px system-ui`;
  ctx.strokeStyle = css("--line"); ctx.beginPath(); ctx.moveTo(x0, 0); ctx.lineTo(x0, H); ctx.stroke();
  for (let i = 0; i < n; i++) {
    const v = Math.max(-1, Math.min(1, values[i])), y = 2 + i * rowH;
    ctx.fillStyle = typeof colors === "function" ? colors(i, v) : colors[i];
    const w = v * half;
    ctx.fillRect(Math.min(x0, x0 + w), y + pad / 2, Math.abs(w), rowH - pad);
    if (lab) { ctx.fillStyle = css("--muted"); ctx.textAlign = "right"; ctx.fillText(labels[i], lab - 6, y + rowH / 2 + 4); }
  }
  ctx.textAlign = "left";
}

// A line series with a moving cursor; ys can contain several series [{ys, color}].
function drawSeries(cv, ctx, series, cursor, opts = {}) {
  const W = cv.width / (window.devicePixelRatio || 1), H = cv.height / (window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, W, H);
  const all = series.flatMap(s => s.ys);
  let lo = opts.lo ?? Math.min(...all), hi = opts.hi ?? Math.max(...all);
  if (hi - lo < 1e-9) hi = lo + 1;
  const n = Math.max(...series.map(s => s.ys.length)), L = 34, R = 8, T = 8, B = 18;
  const X = i => L + (W - L - R) * i / Math.max(1, (opts.n || n) - 1), Y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  ctx.strokeStyle = css("--grid"); ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui"; ctx.lineWidth = 1;
  for (const v of opts.ticks || [lo, (lo + hi) / 2, hi]) {
    ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - R, Y(v)); ctx.stroke();
    ctx.fillText(Math.round(v), 2, Y(v) + 4);
  }
  if (opts.xlabel) ctx.fillText(opts.xlabel, W - R - ctx.measureText(opts.xlabel).width, H - 3);
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2; ctx.beginPath();
    s.ys.forEach((v, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(i), Y(v)));
    ctx.stroke();
  }
  if (cursor != null) {
    ctx.strokeStyle = css("--accent"); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X(cursor), T); ctx.lineTo(X(cursor), H - B); ctx.stroke();
  }
}
