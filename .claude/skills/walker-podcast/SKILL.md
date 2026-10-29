---
name: walker-podcast
description: Write, voice and deploy an episode of the Bipedal CMA-ES project podcast (the "Podcast" tab of https://jca374.github.io/bipedal-cmaes/). Use it whenever something significant happens in this project (a breakthrough, a stage passing, a new kind of experiment, a clear failure, a new best walker) or whenever the user asks for a podcast, an audio update, an episode, a spoken summary of progress, or to publish/deploy an episode to the site, even if they don't say "podcast". It covers gathering the facts, writing for the ear, the voice-text pipeline (Kokoro + Whisper check), the feed, commit/push and verifying the GitHub Pages deployment.
---

# Walker podcast: write → voice → deploy

> A general version for any project lives at `~/.claude/skills/project-podcast/` (same steps, the settings in
> `podcast.conf`). This repo has a `podcast.conf`, so either skill works here.

Each episode is a short spoken update (2-6 minutes) for the user, who follows the project by ear. The value
is that it's **true, current and easy to follow**: real numbers from the logs, in plain spoken language, with
what happened, what worked, what failed and what's next. Episodes are published on the site's Podcast tab.

Paths below are relative to the project root (`/home/jonas/code/games/bi-pedal/bipedal-CMA-ES`).

## When to make one

Make an episode as soon as a major change or breakthrough happens, not just on a schedule: a stage passing,
the goal met or a new best score, a new kind of experiment (new model, new reward, new curriculum), or an
experiment that clearly failed (and why). If several things happened since the last episode, one episode can
cover them all. Skip routine "still training" check-ins.

## 1. Gather the facts (only real numbers)

Find what's new since the last episode:
- the newest `podcast/episode-NN.md` (number and what it already covered);
- `docs/data/curriculum_journal.json` entries after that episode's date, and the tail of `TRAINING_LOG.md`;
- `docs/data/curriculum_attempts.json` (attempt titles/notes), the `curriculum*.log` stage outcomes;
- `docs/data/goal_*.json` (goal checks on 20 independent tracks: score, falls, front-foot changes,
  lopsidedness, effort, steps) and, for big claims, a 100-track evaluation (`evaluate.py --episodes 100`).

Every number in the episode must come from these files or from a command you just ran. If a number is
uncertain (e.g. selected on few tracks), say so; the winner's curse is a recurring theme here.

## 2. Write the episode

Create `podcast/episode-NN.md` (next number, two digits) and `podcast/episode-NN.notes.json`.

- First line: `# Episode NN: <short title>`. Then 300-700 words of prose, first person ("I"), addressed to
  the listener ("you asked for…").
- Write for the ear: short sentences; one idea per sentence; numbers spelled the way they're said
  ("two hundred and eighty four", "about ninety", "one and a half times"); no tables, lists, symbols,
  parentheses, URLs, or abbreviations a listener can't decode (say "the gait penalty", not "GAIT"). Name
  runs by what they are ("the wider network"), not by run ids.
- Structure: where we were → what happened → what worked → what failed and why → what's next. Be
  honest about failures and uncertainty; that's what makes the updates trustworthy.
- `episode-NN.notes.json`: a JSON list of 3-5 one-line show notes (these can use digits).

## 3. Prepare the voice (speech-flow scan)

```bash
.claude/skills/walker-podcast/scripts/prepare.sh NN
```
It builds `voice/podcast-NN/` from the episode with the voice-text pipeline (kind prose, project glossary)
and runs the speech-flow scan. If it reports candidates (exit code 2), fix the phrasing **in
`podcast/episode-NN.md`** (usually: split long sentences, remove a semicolon) and run it again; it
rebuilds the work dir from the source each time.

## 4. Narrate, check and deploy

```bash
setsid nohup .claude/skills/walker-podcast/scripts/publish.sh NN > podcast/publish_NN.log 2>&1 &
```
This runs in the background because narration takes several minutes per minute of audio when training
is using the CPU (~6 min of audio took ~28 min). It: preps the script, narrates with Kokoro (af_heart),
runs the Whisper listen-check, exports, copies the MP3 to `docs/podcast/`, rebuilds the feed
(`podcast/make_feed.py`), runs the tests, commits only the episode files, pushes, waits for the GitHub
Pages build and checks that the page and the MP3 return HTTP 200. Follow it with
`tail podcast/publish_NN.log`; it ends with `DEPLOYED <url>` or `FAILED: <reason>`.

## 5. Report

Tell the user the title, the length, the listen-check result and the link. Listen-check flags like
"a" heard as "one", or "sixteen" heard as "one thousand six", are Whisper writing numbers differently,
not narration errors; say so. Anything else (a skipped or garbled word) means: fix the text and publish
again (same episode number; the scripts overwrite it).

## Notes

- Don't edit a `curriculum*.sh` or other script while it runs; the podcast scripts don't touch training.
- The voice-text skill (`~/.claude/skills/voice-text/`) does the audio work; these scripts only call it.
- Glossary: `voice/lesson-01/glossary.md` (tanh, NEAT, CMA-ES, PPO, TD3, lidar) is copied into each episode.
