// Shared bits: lesson registry, header/pager, theme toggle, quizzes, colour helper.
const LESSONS = [
  { slug: "01-the-task", title: "The task", sub: "What the walker senses, does and is scored on" },
  { slug: "02-a-policy-is-numbers", title: "A policy is just numbers", sub: "The neural network, and how 932 weights become one point" },
  { slug: "03-cma-es", title: "Evolution strategies and CMA-ES", sub: "Sample, rank, move, stretch: learning without gradients" },
  { slug: "04-noisy-fitness", title: "Noisy fitness, honest selection", sub: "Why ten terrains lie, and how to pick a champion" },
  { slug: "05-the-training-run", title: "The training run", sub: "Three false starts, and the fitness that finally produced a walker", draft: true },
  { slug: "06-inside-the-champion", title: "Inside the champion", sub: "What the hidden neurons do while it walks", soon: true },
  { slug: "07-results", title: "Results and comparison", sub: "100 unseen terrains; CMA-ES against NEAT, PPO and TD3", soon: true },
];

function siteRoot() {
  return document.body.dataset.root || ".";
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function renderHeader() {
  const el = document.getElementById("top");
  if (!el) return;
  const r = siteRoot();
  el.className = "top wrap";
  el.innerHTML = `<a class="brand" href="${r}/index.html">Bipedal <span>CMA-ES</span></a>
    <nav><a href="${r}/index.html#video">Video</a><a href="${r}/index.html#lessons">Lessons</a>
    <a href="${r}/index.html#training">Training</a>
    <a href="https://github.com/JCA374/bipedal-cmaes">Code</a>
    <button class="theme-btn" id="theme-btn" aria-label="Toggle colour theme">◐</button></nav>`;
  document.getElementById("theme-btn").onclick = () => {
    const cur = document.documentElement.dataset.theme ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) {}
    window.dispatchEvent(new Event("themechange"));
  };
}

function renderPager() {
  const el = document.getElementById("pager");
  if (!el) return;
  const i = LESSONS.findIndex(l => l.slug === document.body.dataset.lesson);
  const r = siteRoot();
  const link = (l, dir) => l && !l.soon
    ? `<a href="${r}/lessons/${l.slug}/index.html">${dir === "prev" ? "← " : ""}Lesson ${LESSONS.indexOf(l) + 1}: ${l.title}${dir === "next" ? " →" : ""}</a>`
    : "<span></span>";
  el.className = "pager";
  el.innerHTML = link(LESSONS[i - 1], "prev") + link(LESSONS[i + 1], "next");
}

function renderLessonList(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = LESSONS.map((l, i) => `
    <a class="lesson-card${l.soon ? " soon" : ""}" href="${siteRoot()}/lessons/${l.slug}/index.html">
      <div class="n">LESSON ${i + 1}${l.soon ? '<span class="tag">after training</span>' : l.draft ? '<span class="tag">being completed</span>' : ""}</div>
      <h3>${l.title}</h3><p>${l.sub}</p></a>`).join("");
}

// Quiz markup: <div class="q" data-answer="1"><p>Q</p><button>A</button>...<div class="why">…</div></div>
function initQuizzes() {
  document.querySelectorAll(".q").forEach(q => {
    const right = +q.dataset.answer;
    q.querySelectorAll("button").forEach((b, i) => {
      b.onclick = () => {
        if (q.classList.contains("done")) return;
        b.classList.add(i === right ? "right" : "wrong");
        if (i !== right) q.querySelectorAll("button")[right].classList.add("right");
        q.classList.add("done");
      };
    });
  });
}

(function () {
  try {
    const t = localStorage.getItem("theme");
    if (t) document.documentElement.dataset.theme = t;
  } catch (e) {}
  document.addEventListener("DOMContentLoaded", () => {
    renderHeader();
    renderPager();
    renderLessonList("lesson-list");
    initQuizzes();
  });
})();
