"""Render the narrated lesson clips into docs/video/<lesson slug>.mp4 (+ .vtt captions, .jpg poster).

  python video_tools/make_lessons.py                       # all lessons with narration
  python video_tools/make_lessons.py 03-cma-es --stills    # quick preview frames, no video

Narration comes from voice/lesson-NN/ (voice-text pipeline). Without it a clip renders silent
with default beat lengths.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_scenes as F  # noqa: E402
import lesson_scenes as L  # noqa: E402
import timeline as T  # noqa: E402
import vcommon as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lessons", nargs="*", default=list(L.LESSONS) + list(F.VIDEOS))
    ap.add_argument("--stills", action="store_true", help="write preview stills only")
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--out", default=os.path.join(C.REPO_ROOT, "docs", "video"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for slug in args.lessons:
        vslug, build = {**L.LESSONS, **F.VIDEOS}[slug]
        voice = os.path.join(C.REPO_ROOT, "voice", vslug)
        voice = voice if os.path.exists(os.path.join(voice, "audio", "voiceover", "timings.json")) else None
        scenes = build()
        if args.stills:
            sd = os.path.join(C.REPO_ROOT, "video_tools", "cache", "stills", slug)
            os.makedirs(sd, exist_ok=True)
            for s in scenes:
                for k in range(len(s.default)):
                    T.still(s, k, 0.85, voice).save(os.path.join(sd, f"{s.name}_{k}.jpg"), quality=85)
            print(f"stills -> {sd}")
            continue
        out = os.path.join(args.out, f"{slug}.mp4")
        T.render(scenes, out, voice_dir=voice, crf=args.crf)
        T.still(scenes[0], 0, 0.9, voice).save(os.path.join(args.out, f"{slug}.jpg"), quality=85)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
