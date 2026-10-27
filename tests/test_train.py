import json

import numpy as np

import common
import train


def test_smoke_train_and_resume_uses_current_overrides(tmp_path):
    common_args = ["--name", "t", "--runs-dir", str(tmp_path), "--workers", "2",
                   "--hidden", "4", "--popsize", "6", "--k", "1",
                   "--hof-every", "2", "--hof-seeds", "2", "--hof-top", "1", "--checkpoint-every", "1"]
    es, state = train.main(common_args + ["--generations", "2"])
    run = tmp_path / "t"
    for f in ["stats.csv", "run_config.json", "hof.json", "best/params.npy", "best/meta.json",
              "checkpoints/latest.pkl.gz", "milestones/gen0/params.npy", "snapshots/mean_00000.npy"]:
        assert (run / f).exists(), f
    assert state["generation"] == 2
    pol, meta = common.load_policy(run)
    assert pol.n_hidden_layers >= 1 and meta["sizes"] == [24, 4, 4]
    mean_before = es.mean.copy()

    (run / "overrides.json").write_text(json.dumps({"k": 2, "popsize": 8, "reason": "test"}))
    es2, state2 = train.main(common_args + ["--resume", "--generations", "3"])
    assert state2["generation"] == 3                      # counter continues
    assert es2.popsize == 8                               # override applied
    cfg = json.loads((run / "run_config.json").read_text())
    assert cfg["effective"]["k"] == 2 and cfg["history"][-1]["reason"] == "test"
    assert state2["restart_log"][-1]["reason"] == "popsize override"
    assert not np.allclose(es2.mean, mean_before)         # optimiser kept going
    rows = (run / "stats.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 3

    # A second resume with the same override must NOT restart again.
    es3, state3 = train.main(common_args + ["--resume", "--generations", "4"])
    assert es3.popsize == 8 and len(state3["restart_log"]) == 1


def test_resume_keeps_cma_state(tmp_path):
    a = ["--name", "r", "--runs-dir", str(tmp_path), "--workers", "1", "--hidden", "3",
         "--popsize", "4", "--k", "1", "--hof-every", "0", "--checkpoint-every", "1"]
    es, _ = train.main(a + ["--generations", "2"])
    es2, _ = train.main(a + ["--resume", "--generations", "2"])  # no new generations
    assert np.allclose(es.mean, es2.mean) and es.sigma == es2.sigma
    assert es2.countiter == es.countiter


def test_sigterm_saves_checkpoint(tmp_path):
    """`kill <pid>` (SIGTERM) must stop training cleanly with a checkpoint, also when the
    job was started in the background with SIGINT ignored."""
    import signal
    import subprocess
    import sys
    import time
    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    p = subprocess.Popen([sys.executable, "-u", str(root / "train.py"), "--name", "s", "--runs-dir", str(tmp_path),
                          "--workers", "1", "--hidden", "3", "--popsize", "4", "--k", "1", "--hof-every", "0",
                          "--checkpoint-every", "1000"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
    stats = tmp_path / "s" / "stats.csv"
    for _ in range(200):
        if stats.exists() and len(stats.read_text().splitlines()) >= 3:
            break
        time.sleep(0.1)
    p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=60)
    assert "Interrupted, saving checkpoint" in out
    assert (tmp_path / "s" / "checkpoints" / "latest.pkl.gz").exists()
