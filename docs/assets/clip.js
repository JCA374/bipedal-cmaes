// Shows the lesson's narrated clip (docs/video/<lesson slug>.mp4) once it has been rendered.
(async function () {
  const slot = document.getElementById("clip-slot");
  if (!slot) return;
  const slug = document.body.dataset.lesson, r = siteRoot();
  const src = `${r}/video/${slug}.mp4`;
  let ok = false;
  try { ok = (await fetch(src, { method: "HEAD" })).ok; } catch (e) {}
  slot.innerHTML = ok
    ? `<div class="video-card"><video controls preload="metadata" poster="${r}/video/${slug}.jpg">
         <source src="${src}" type="video/mp4"><track kind="captions" src="${r}/video/${slug}.vtt" srclang="en" label="English" default></video></div>
       <p class="note">Narrated version of this lesson. The page below goes deeper and has the interactive parts.</p>`
    : `<div class="placeholder">The narrated clip for this lesson is still being rendered. The page below is complete.</div>`;
})();
