import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import canyon_jumperl  # noqa: F401  (registers CanyonRacer-v0)
from canyon_jumperl.env import CanyonRacerEnv, resolve_rom_path
from canyon_jumperl.env import CanyonRacerEnv as _Env
from canyon_jumperl.symbols import Symbols


def mk(**kw):
    """Legacy dict/discrete env used by these wiring tests."""
    return _Env(**{"observation_mode": "state", "action_mode": "discrete", **kw})



def test_symbols_resolve():
    rom = resolve_rom_path()
    sym = Symbols(rom.with_suffix(".noi"), rom.with_suffix(".cdb"))
    for name in ("state", "lives_count", "game_state", "__rand_seed", "rand_seed_copy"):
        assert 0xC000 <= sym.addr(name) < 0xE000, name  # WRAM
    for f in ("ship_x", "ship_y", "ship_z", "ship_state", "obstacles", "score", "game_level",
              "p_scx_table_frame_base", "obstacles_active_first"):
        sym.field("game_state_data", f)
    assert sym.field("game_state_data", "ship_x").offset == 15
    assert sym.field("level_entry", "obst_speed").size == 2
    assert sym.struct_size("game_state_data") == 81  # matches sizeof(state) in the .cdb


def test_registered_and_initial_state():
    env = gym.make("CanyonRacer-v0", observation_mode="state", action_mode="discrete")
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["game_state"] == 4 and info["lives"] == 10 and info["score"] == 0
    assert obs["ship_state"] == 0 and obs["lives"] == 10
    env.close()


def test_check_env():
    env = mk(max_steps=50)
    check_env(env, skip_render_check=True)
    env.close()


def test_check_env_pixels_multibinary():
    env = mk(observation_mode="both", action_mode="multibinary", max_steps=20)
    check_env(env, skip_render_check=True)
    env.close()


def _rollout(env, seed, actions):
    obs, _ = env.reset(seed=seed)
    trace = [obs]
    rewards = []
    for a in actions:
        obs, r, term, trunc, _ = env.step(a)
        trace.append(obs)
        rewards.append(r)
        if term or trunc:
            break
    return trace, rewards


def _same(t1, t2):
    assert len(t1) == len(t2)
    for a, b in zip(t1, t2):
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])


def test_seeded_determinism():
    rng = np.random.default_rng(123)
    actions = rng.integers(0, 11, 600).tolist()  # long enough for obstacles + crashes
    env1 = mk(max_steps=1000)
    t1, r1 = _rollout(env1, 7, actions)
    t1b, r1b = _rollout(env1, 7, actions)  # same env instance, reset again
    env2 = mk(max_steps=1000)  # fresh instance
    t2, r2 = _rollout(env2, 7, actions)
    _same(t1, t1b); _same(t1, t2)
    assert r1 == r1b == r2
    t3, r3 = _rollout(env1, 8, actions)  # different seed -> different game
    assert r3 != r1 or any(not np.array_equal(a["obstacles"], b["obstacles"]) for a, b in zip(t1, t3))
    env1.close(); env2.close()


def test_stock_game_seed_option():
    env = mk()
    actions = [0] * 80
    a, ra = _rollout_opt(env, actions)
    b, rb = _rollout_opt(env, actions)
    _same(a, b)
    env.close()


def _rollout_opt(env, actions):
    obs, _ = env.reset(seed=1, options={"game_seed": None})
    tr = [obs]
    for a in actions:
        obs, r, *_ = env.step(a)
        tr.append(obs)
    return tr, None


def test_random_policy_smoke_1000_steps():
    env = mk(max_steps=100000)
    obs, _ = env.reset(seed=0)
    total, resets, crashes = 0.0, 0, 0
    for _ in range(1000):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)
        total += r
        crashes += info["crashed"]
        if term or trunc:
            resets += 1
            env.reset()
    print(f"random policy: return={total:.1f} resets={resets} crashes={crashes}")
    env.close()


def test_game_over_terminates():
    env = mk(max_steps=100000, frame_skip=8)
    env.reset(seed=3)
    term = False
    for _ in range(6000):  # standing still eventually crashes through all lives
        _, _, term, _, info = env.step(0)
        if term:
            break
    assert term and info["lives"] == 0
    env.close()


def test_truncation():
    env = mk(max_steps=5)
    env.reset(seed=0)
    for i in range(5):
        _, _, term, trunc, _ = env.step(0)
    assert trunc and not term
    env.close()


def test_jump_changes_z():
    env = mk()
    env.reset(seed=0)
    zs = []
    for _ in range(6):
        obs, *_ = env.step(5)  # hold A
        zs.append(float(obs["ship"][2]))
    assert max(zs) > 0
    env.close()
