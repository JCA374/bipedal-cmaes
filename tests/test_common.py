import numpy as np

import common


def test_flatten_roundtrip():
    sizes = (24, 32, 4)
    theta = np.random.default_rng(0).normal(size=common.n_params(sizes))
    assert common.n_params(sizes) == 932
    assert np.array_equal(common.flatten(common.unflatten(theta, sizes)), theta)


def test_forward_matches_hand_reference():
    sizes = (24, 5, 3, 4)
    rng = np.random.default_rng(1)
    theta = rng.normal(size=common.n_params(sizes))
    x = rng.uniform(-1, 1, size=24)
    # hand-written: walk the flat vector in the documented layout
    k, h = 0, x
    for i, o in zip(sizes[:-1], sizes[1:]):
        W = np.array([[theta[k + r * i + c] for c in range(i)] for r in range(o)])
        k += o * i
        b = theta[k:k + o]
        k += o
        h = np.tanh(W @ h + b)
    assert np.allclose(common.MLP(sizes, theta).activate(x), h)


def test_policy_has_hidden_layer():
    # Hard requirement: the default architecture (and every saved champion) has >= 1 hidden layer.
    import train
    ap_sizes = (common.N_OBS, 32, common.N_ACT)
    assert common.MLP(ap_sizes).n_hidden_layers >= 1
    import inspect
    assert "must have >= 1 hidden layer" in inspect.getsource(train.main)


def test_seed_ranges_do_not_overlap():
    rs = sorted(common.SEED_RANGES.values())
    for (a0, a1), (b0, b1) in zip(rs, rs[1:]):
        assert a1 <= b0


def _manual(pol, env, seed):
    obs, _ = common.fresh_reset(env, seed)
    total, n = 0.0, 0
    for n in range(1, 1601):
        a = pol.activate(common.scale_obs(obs))
        obs, r, term, trunc, _ = env.step(np.asarray(a, dtype=np.float32))
        total += r
        if term or trunc:
            break
    return total, n


def test_run_episode_raw_reward_without_early_stop():
    env = common.make_env()
    sizes = (24, 8, 4)
    for s in range(6):
        pol = common.MLP(sizes, np.random.default_rng(s).normal(0, 0.5, common.n_params(sizes)))
        total, n = _manual(pol, env, 100 + s)
        res = common.run_episode(pol, env, 100 + s, early_stop=False)
        assert res["steps"] == n and res["saved"] == 0 and not res["stalled"]
        assert abs(res["reward"] - total) < 1e-6


def test_early_stop_fires_on_stall_only_when_enabled():
    env = common.make_env()
    # A policy that brakes every joint (tanh saturates at -1/+1 per output via bias only).
    sizes = (24, 4, 4)
    found = False
    for s in range(30):
        pol = common.MLP(sizes, np.random.default_rng(s).normal(0, 1.0, common.n_params(sizes)))
        full = common.run_episode(pol, env, 7, early_stop=False)
        if full["steps"] == 1600:  # survived the whole episode: likely stood still
            es = common.run_episode(pol, env, 7, early_stop=True)
            assert es["stalled"] and es["saved"] == 1600 - es["steps"]
            found = True
            break
    assert found, "no standing policy found among 30 random ones"


def test_episodes_are_reproducible_on_a_reused_env():
    env = common.make_env()
    sizes = (24, 8, 4)
    pol = common.MLP(sizes, np.random.default_rng(0).normal(0, 0.5, common.n_params(sizes)))
    rs = [common.run_episode(pol, env, 100, early_stop=False)["reward"] for _ in range(5)]
    assert len(set(rs)) == 1


def test_symclock_shape_mirroring_and_clock():
    sizes = common.policy_sizes("symclock", [32])
    pol = common.make_policy("symclock", sizes, np.random.default_rng(0).normal(0, .5, 931))
    assert sizes == (26, 32, 2) and pol.n_params == 931 and pol.n_hidden_layers >= 1
    x = np.random.default_rng(1).uniform(-1, 1, 24)
    pol.reset()
    X = pol.inputs(x)
    ph = 0.0
    assert np.allclose(X[0], np.concatenate([x[0:4], x[4:9], x[9:14], x[14:24], [np.sin(ph), np.cos(ph)]]))
    assert np.allclose(X[1], np.concatenate([x[0:4], x[9:14], x[4:9], x[14:24], [np.sin(ph + np.pi), np.cos(ph + np.pi)]]))
    # leg 2's command = the shared controller applied to leg 2's view
    (W1, b1), (W2, b2) = pol.layers
    a = pol.activate(x)
    assert np.allclose(a[2:], np.tanh(W2 @ np.tanh(W1 @ X[1] + b1) + b2))
    # identical legs still get different commands: the clock breaks the symmetry
    x_sym = x.copy()
    x_sym[9:14] = x_sym[4:9]
    pol.reset()
    a = pol.activate(x_sym)
    assert not np.allclose(a[:2], a[2:])
    assert 14 < pol.period < 110


def test_run_episode_reports_gait():
    env = common.make_env()
    pol = common.make_policy("symclock", (26, 8, 2), np.random.default_rng(2).normal(0, .5, 26 * 8 + 8 + 8 * 2 + 2 + 1))
    r = common.run_episode(pol, env, 5, early_stop=False)
    assert {"crossings", "lead_changes", "front2_share"} <= set(r)
    assert 0 <= r["crossings"] <= r["lead_changes"] and 0 <= r["front2_share"] <= 1
    r2 = common.run_episode(pol, env, 5, early_stop=False)     # reset() makes it reproducible
    assert r2["reward"] == r["reward"]
