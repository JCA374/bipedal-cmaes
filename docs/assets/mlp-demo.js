// Lesson 2: a live 24 -> 32 -> 4 tanh network, same flat layout as common.py:
// theta = [W1 (32x24 row-major), b1 (32), W2 (4x32), b2 (4)].

function randn(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

class MLP {
  constructor(sizes) { this.sizes = sizes; this.n = 0; for (let i = 0; i + 1 < sizes.length; i++) this.n += sizes[i] * sizes[i + 1] + sizes[i + 1]; }
  forward(theta, x) {
    const acts = [x.slice()], pre = [];
    let k = 0, h = x;
    for (let l = 0; l + 1 < this.sizes.length; l++) {
      const I = this.sizes[l], O = this.sizes[l + 1], z = new Array(O).fill(0), y = new Array(O);
      for (let o = 0; o < O; o++) { let s = 0; for (let i = 0; i < I; i++) s += theta[k + o * I + i] * h[i]; z[o] = s; }
      k += O * I;
      for (let o = 0; o < O; o++) { z[o] += theta[k + o]; y[o] = Math.tanh(z[o]); }
      k += O;
      pre.push(z); acts.push(y); h = y;
    }
    return { acts, pre };
  }
  // index of weight from unit i (layer l) to unit o (layer l+1), and of bias o
  w(l, o, i) { let k = 0; for (let j = 0; j < l; j++) k += this.sizes[j] * this.sizes[j + 1] + this.sizes[j + 1]; return k + o * this.sizes[l] + i; }
  b(l, o) { let k = 0; for (let j = 0; j < l; j++) k += this.sizes[j] * this.sizes[j + 1] + this.sizes[j + 1]; return k + this.sizes[l] * this.sizes[l + 1] + o; }
}

function drawNet(cv, ctx, net, theta, res, sel, names, outNames) {
  const W = cv.width / (window.devicePixelRatio || 1), H = cv.height / (window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, W, H);
  const xs = [110, W / 2 - 10, W - 150];
  const pos = net.sizes.map((n, l) => Array.from({ length: n }, (_, i) => [xs[l], 12 + (H - 40) * (n === 1 ? .5 : i / (n - 1))]));
  const pos2 = pos[2].map(([x], i) => [x, H / 2 - 90 + i * 60]);
  pos[2] = pos2;
  const P = css("--pos"), N = css("--neg");
  for (let l = 0; l < 2; l++) {
    for (let o = 0; o < net.sizes[l + 1]; o++) for (let i = 0; i < net.sizes[l]; i++) {
      const w = theta[net.w(l, o, i)];
      const focus = sel != null && ((l === 0 && o === sel) || (l === 1 && i === sel));
      const signal = Math.abs(w * res.acts[l][i]);
      ctx.globalAlpha = sel == null ? Math.min(.5, signal * .35 + .03) : focus ? Math.min(.95, signal * .6 + .12) : .02;
      ctx.strokeStyle = w > 0 ? P : N; ctx.lineWidth = Math.min(3, .4 + Math.abs(w) * 1.2);
      ctx.beginPath(); ctx.moveTo(...pos[l][i]); ctx.lineTo(...pos[l + 1][o]); ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  const txt = css("--text"), mut = css("--muted");
  pos.forEach((layer, l) => layer.forEach(([x, y], i) => {
    const a = res.acts[l][i], r = l === 1 ? 6 : l === 0 ? 6 : 13;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7);
    ctx.fillStyle = a >= 0 ? P : N; ctx.globalAlpha = .15 + .85 * Math.min(1, Math.abs(a)); ctx.fill();
    ctx.globalAlpha = 1; ctx.strokeStyle = l === 1 && i === sel ? css("--accent") : txt; ctx.lineWidth = l === 1 && i === sel ? 3 : 1; ctx.stroke();
  }));
  ctx.fillStyle = mut; ctx.font = "10.5px system-ui"; ctx.textAlign = "right";
  pos[0].forEach(([x, y], i) => ctx.fillText(names[i], x - 10, y + 3.5));
  ctx.textAlign = "left"; ctx.font = "600 12.5px system-ui"; ctx.fillStyle = txt;
  pos[2].forEach(([x, y], i) => ctx.fillText(`${outNames[i]}  ${res.acts[2][i].toFixed(2)}`, x + 18, y + 4));
  ctx.fillStyle = mut; ctx.font = "12px system-ui"; ctx.textAlign = "center";
  ctx.fillText("24 inputs", xs[0], H - 1); ctx.fillText("32 hidden (tanh)", xs[1], H - 1); ctx.fillText("4 outputs", xs[2] + 10, H - 1);
  ctx.textAlign = "left";
  return pos;
}

function drawTanh(cv, ctx, z, label) {
  const W = cv.width / (window.devicePixelRatio || 1), H = cv.height / (window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, W, H);
  const X = v => W / 2 + v * (W - 20) / 10, Y = v => H / 2 - v * (H / 2 - 12);
  ctx.strokeStyle = css("--grid"); ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(W, Y(0)); ctx.moveTo(X(0), 0); ctx.lineTo(X(0), H); ctx.stroke();
  ctx.strokeStyle = css("--muted"); ctx.lineWidth = 2; ctx.beginPath();
  for (let v = -5; v <= 5; v += .05) (v === -5 ? ctx.moveTo : ctx.lineTo).call(ctx, X(v), Y(Math.tanh(v)));
  ctx.stroke();
  const zc = Math.max(-5, Math.min(5, z));
  ctx.fillStyle = css("--accent"); ctx.beginPath(); ctx.arc(X(zc), Y(Math.tanh(z)), 6, 0, 7); ctx.fill();
  ctx.fillStyle = css("--text"); ctx.font = "12px system-ui";
  ctx.fillText(label, 6, 14);
  ctx.fillStyle = css("--muted"); ctx.fillText("tanh(z)", W - 50, 14); ctx.fillText("z →", W - 26, Y(0) - 4);
}

// θ as a barcode: 932 thin stripes, colour = sign, intensity = size. Layer boundaries marked.
function drawTheta(cv, ctx, net, theta, prev) {
  const W = cv.width / (window.devicePixelRatio || 1), H = cv.height / (window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, W, H);
  const n = theta.length, w = W / n, P = css("--pos"), N = css("--neg");
  for (let k = 0; k < n; k++) {
    const v = theta[k];
    ctx.fillStyle = v >= 0 ? P : N; ctx.globalAlpha = Math.min(1, Math.abs(v) / 1.2);
    ctx.fillRect(k * w, 16, Math.max(1, w), H - 30);
    if (prev && Math.abs(prev[k] - v) > 1e-9) {
      ctx.globalAlpha = Math.min(1, Math.abs(prev[k] - v) * 3);
      ctx.fillStyle = css("--accent"); ctx.fillRect(k * w, H - 12, Math.max(1, w), 10);
    }
  }
  ctx.globalAlpha = 1; ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui";
  const segs = [["W1  24×32 = 768", 0, 768], ["b1", 768, 800], ["W2  32×4", 800, 928], ["", 928, 932]];
  ctx.strokeStyle = css("--text");
  for (const [lab, a, b] of segs) {
    ctx.fillText(lab, a * w + 3, 11);
    ctx.beginPath(); ctx.moveTo(a * w, 14); ctx.lineTo(a * w, H - 12); ctx.stroke();
  }
}
