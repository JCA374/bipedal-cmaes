// Lesson 3: CMA-ES (Hansen's standard algorithm) in 2-D, next to a plain ES with a fixed
// isotropic step size. Both minimise f; the page talks about "fitness" = -f.

const FUNCS = {
  valley: { name: "tilted narrow valley", f: (x, y) => { const a = .6, c = Math.cos(a), s = Math.sin(a), u = c * x + s * y, v = -s * x + c * y; return u * u + 25 * v * v; }, start: [-3.5, 3], lo: 0, hi: 60 },
  rosen: { name: "banana (Rosenbrock)", f: (x, y) => { const X = x / 1.5, Y = y / 1.5 + .5; return (1 - X) ** 2 + 20 * (Y - X * X) ** 2; }, start: [-3.5, -2.5], lo: 0, hi: 30 },
  bumpy: { name: "bumpy bowl (many small traps)", f: (x, y) => .5 * (x * x + y * y) + 3 * (2 - Math.cos(2.2 * x) - Math.cos(2.2 * y)), start: [-3.8, 3.4], lo: 0, hi: 25 },
};

class CMA2 {
  constructor(m, sigma, lam, rng) {
    this.n = 2; this.m = m.slice(); this.sigma = sigma; this.rng = rng; this.setLambda(lam);
    this.C = [[1, 0], [0, 1]]; this.pc = [0, 0]; this.ps = [0, 0]; this.gen = 0;
  }
  setLambda(lam) {
    const n = this.n; this.lam = lam; this.mu = Math.floor(lam / 2);
    let w = [...Array(this.mu)].map((_, i) => Math.log(this.mu + .5) - Math.log(i + 1));
    const s = w.reduce((a, b) => a + b); this.w = w.map(v => v / s);
    this.mueff = 1 / this.w.reduce((a, b) => a + b * b, 0);
    const me = this.mueff;
    this.cs = (me + 2) / (n + me + 5);
    this.ds = 1 + 2 * Math.max(0, Math.sqrt((me - 1) / (n + 1)) - 1) + this.cs;
    this.cc = (4 + me / n) / (n + 4 + 2 * me / n);
    this.c1 = 2 / ((n + 1.3) ** 2 + me);
    this.cmu = Math.min(1 - this.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me));
    this.chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));
  }
  eig() {  // C = B diag(d^2) B^T for a symmetric 2x2
    const [[a, b], [, c]] = this.C, t = (a + c) / 2, q = Math.sqrt(((a - c) / 2) ** 2 + b * b);
    const l1 = t + q, l2 = Math.max(1e-20, t - q);
    let v1 = Math.abs(b) > 1e-14 ? [l1 - c, b] : (a >= c ? [1, 0] : [0, 1]);
    const nv = Math.hypot(...v1); v1 = [v1[0] / nv, v1[1] / nv];
    return { B: [[v1[0], -v1[1]], [v1[1], v1[0]]], d: [Math.sqrt(l1), Math.sqrt(l2)] };
  }
  ask() {
    const { B, d } = this.eig(); this.Bd = { B, d };
    this.pop = [...Array(this.lam)].map(() => {
      const z = [randn(this.rng), randn(this.rng)];
      const y = [B[0][0] * d[0] * z[0] + B[0][1] * d[1] * z[1], B[1][0] * d[0] * z[0] + B[1][1] * d[1] * z[1]];
      return { y, x: [this.m[0] + this.sigma * y[0], this.m[1] + this.sigma * y[1]] };
    });
    return this.pop;
  }
  tell(fs) {
    const n = 2, idx = [...fs.keys()].sort((i, j) => fs[i] - fs[j]), sel = idx.slice(0, this.mu);
    this.ranked = idx;
    const yw = [0, 0];
    sel.forEach((i, k) => { yw[0] += this.w[k] * this.pop[i].y[0]; yw[1] += this.w[k] * this.pop[i].y[1]; });
    this.prevMean = this.m.slice();
    this.m = [this.m[0] + this.sigma * yw[0], this.m[1] + this.sigma * yw[1]];
    // C^{-1/2} yw
    const { B, d } = this.Bd, bt = [B[0][0] * yw[0] + B[1][0] * yw[1], B[0][1] * yw[0] + B[1][1] * yw[1]];
    const s = [bt[0] / d[0], bt[1] / d[1]], cinv = [B[0][0] * s[0] + B[0][1] * s[1], B[1][0] * s[0] + B[1][1] * s[1]];
    const k1 = Math.sqrt(this.cs * (2 - this.cs) * this.mueff);
    this.ps = this.ps.map((p, i) => (1 - this.cs) * p + k1 * cinv[i]);
    const psn = Math.hypot(...this.ps);
    this.gen++;
    const hs = psn / Math.sqrt(1 - (1 - this.cs) ** (2 * this.gen)) / this.chiN < 1.4 + 2 / (n + 1) ? 1 : 0;
    const k2 = Math.sqrt(this.cc * (2 - this.cc) * this.mueff);
    this.pc = this.pc.map((p, i) => (1 - this.cc) * p + hs * k2 * yw[i]);
    const C = this.C, c1 = this.c1, cmu = this.cmu, dh = (1 - hs) * this.cc * (2 - this.cc);
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
      let rmu = 0; sel.forEach((p, k) => rmu += this.w[k] * this.pop[p].y[i] * this.pop[p].y[j]);
      C[i][j] = (1 - c1 - cmu) * C[i][j] + c1 * (this.pc[i] * this.pc[j] + dh * C[i][j]) + cmu * rmu;
    }
    C[1][0] = C[0][1];
    this.sigma *= Math.exp((this.cs / this.ds) * (psn / this.chiN - 1));
    this.sigma = Math.min(5, Math.max(1e-6, this.sigma));
  }
}

class SimpleES {  // same recombination, fixed sigma, no covariance learning
  constructor(m, sigma, lam, rng) { this.m = m.slice(); this.sigma = sigma; this.rng = rng; this.lam = lam; this.mu = Math.floor(lam / 2); this.gen = 0; this.C = [[1, 0], [0, 1]]; }
  setLambda(l) { this.lam = l; this.mu = Math.floor(l / 2); }
  ask() { this.pop = [...Array(this.lam)].map(() => { const y = [randn(this.rng), randn(this.rng)]; return { y, x: [this.m[0] + this.sigma * y[0], this.m[1] + this.sigma * y[1]] }; }); return this.pop; }
  tell(fs) {
    const idx = [...fs.keys()].sort((i, j) => fs[i] - fs[j]), sel = idx.slice(0, this.mu);
    this.ranked = idx; this.prevMean = this.m.slice();
    this.m = [0, 1].map(d => sel.reduce((a, i) => a + this.pop[i].x[d], 0) / sel.length);
    this.gen++;
  }
}

class ESView {
  constructor(cv, title) {
    this.cv = cv; this.ctx = setupCanvas(cv, 360, 360); this.title = title; this.R = 5; this.bg = null;
  }
  X(x) { return (x + this.R) / (2 * this.R) * 360; }
  Y(y) { return 360 - (y + this.R) / (2 * this.R) * 360; }
  background(F) {
    const off = document.createElement("canvas"); off.width = 180; off.height = 180;
    const c = off.getContext("2d"), img = c.createImageData(180, 180);
    const dark = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim().startsWith("#0");
    for (let j = 0; j < 180; j++) for (let i = 0; i < 180; i++) {
      const x = -this.R + 2 * this.R * i / 179, y = this.R - 2 * this.R * j / 179;
      const v = Math.min(1, Math.max(0, (Math.log1p(F.f(x, y)) - Math.log1p(F.lo)) / (Math.log1p(F.hi) - Math.log1p(F.lo))));
      const band = (Math.floor(v * 14) % 2) * .05, k = 4 * (j * 180 + i);
      const g = dark ? [20 + 30 * (1 - v), 40 + 90 * (1 - v), 60 + 60 * (1 - v)] : [235 - 60 * (1 - v), 240 - 20 * (1 - v), 245 - 80 * (1 - v)];
      img.data[k] = g[0] * (1 - band); img.data[k + 1] = g[1] * (1 - band); img.data[k + 2] = g[2] * (1 - band); img.data[k + 3] = 255;
    }
    c.putImageData(img, 0, 0); this.bg = off;
  }
  draw(es, trail, best) {
    const ctx = this.ctx;
    ctx.imageSmoothingEnabled = true; ctx.drawImage(this.bg, 0, 0, 360, 360);
    // optimum marker
    ctx.strokeStyle = css("--text"); ctx.lineWidth = 1.5;
    const o = best; ctx.beginPath(); ctx.moveTo(this.X(o[0]) - 6, this.Y(o[1])); ctx.lineTo(this.X(o[0]) + 6, this.Y(o[1]));
    ctx.moveTo(this.X(o[0]), this.Y(o[1]) - 6); ctx.lineTo(this.X(o[0]), this.Y(o[1]) + 6); ctx.stroke();
    // trail of the mean
    ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2; ctx.beginPath();
    trail.forEach((p, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, this.X(p[0]), this.Y(p[1]))); ctx.stroke();
    // sampling ellipse: 1 and 2 std of sigma^2 C
    const [[a, b], [, c]] = es.C, t = (a + c) / 2, q = Math.sqrt(((a - c) / 2) ** 2 + b * b);
    const l1 = t + q, l2 = Math.max(0, t - q), ang = Math.abs(b) > 1e-14 ? Math.atan2(l1 - a, b) : (a >= c ? 0 : Math.PI / 2);
    const px = 360 / (2 * this.R);
    for (const k of [1, 2]) {
      ctx.beginPath();
      ctx.ellipse(this.X(es.m[0]), this.Y(es.m[1]), Math.max(.5, k * es.sigma * Math.sqrt(l1) * px), Math.max(.5, k * es.sigma * Math.sqrt(l2) * px), -ang, 0, 7);
      ctx.strokeStyle = css("--blue"); ctx.globalAlpha = k === 1 ? .9 : .4; ctx.lineWidth = 2; ctx.stroke(); ctx.globalAlpha = 1;
    }
    if (es.pop) {
      const good = new Set((es.ranked || []).slice(0, es.mu));
      es.pop.forEach((p, i) => {
        ctx.beginPath(); ctx.arc(this.X(p.x[0]), this.Y(p.x[1]), good.has(i) ? 4.5 : 3, 0, 7);
        ctx.fillStyle = es.ranked ? (good.has(i) ? css("--green") : css("--muted")) : css("--text"); ctx.fill();
      });
    }
    ctx.beginPath(); ctx.arc(this.X(es.m[0]), this.Y(es.m[1]), 6, 0, 7); ctx.fillStyle = css("--accent"); ctx.fill();
    ctx.strokeStyle = css("--text"); ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = css("--text"); ctx.font = "700 13px system-ui"; ctx.fillText(this.title, 8, 18);
  }
}
