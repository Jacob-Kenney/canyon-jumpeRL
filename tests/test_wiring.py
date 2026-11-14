"""Verify actions/observations are wired to the real ROM (joypad -> game RAM)."""
import numpy as np
import pytest

from canyon_jumperl.env import BUTTON_ORDER, DISCRETE_ACTIONS_WITH_B, CanyonRacerEnv
from canyon_jumperl.env import CanyonRacerEnv as _Env


def mk(**kw):
    """Legacy dict/discrete env used by these wiring tests."""
    return _Env(**{"observation_mode": "state", "action_mode": "discrete", **kw})


NOSEED = {"game_seed": None}  # stock ROM RNG (0x1234)


def mb(*names):
    return np.array([1 if b in names else 0 for b in BUTTON_ORDER], dtype=np.int8)


@pytest.fixture()
def env():
    e = mk(action_mode="multibinary", frame_skip=1, max_steps=10**6)
    e.reset(seed=0, options=NOSEED)
    yield e
    e.close()


def hold(env, buttons, n):
    out = None
    for _ in range(n):
        out = env.step(mb(*buttons))
    return out[0], out[4]


def test_dpad_moves_ship(env):
    def after(buttons, n=20, pre=()):
        env.reset(seed=0, options=NOSEED)
        if pre:
            hold(env, pre, 20)
        o0 = env.step(mb())[0]["ship"]
        o1, _ = hold(env, buttons, n)
        return o0, o1["ship"]

    o0, o1 = after(["right"])
    assert o1[0] > o0[0] + 5 and o1[1] == o0[1], "RIGHT must increase ship x only"
    o0, o1 = after(["left"])
    assert o1[0] < o0[0] - 5 and o1[1] == o0[1], "LEFT must decrease ship x only"
    o0, o1 = after(["up"])
    assert o1[1] < o0[1] - 3 and o1[0] == o0[0], "UP must decrease ship y (toward top of screen)"
    o0, o1 = after(["down"], pre=["up"])
    assert o1[1] > o0[1] + 3 and o1[0] == o0[0], "DOWN must increase ship y"


def _air_time(env, buttons, frames=100):
    env.reset(seed=0, options=NOSEED)
    zs, states = [], []
    for _ in range(frames):
        o = env.step(mb(*buttons))[0]
        zs.append(float(o["ship"][2]))
        states.append(int(o["ship_state"]))
    return max(zs), sum(z > 0 for z in zs), states


def test_a_jumps_hold_longer_than_tap_and_a_up_is_not_special(env):
    p0, air0, _ = _air_time(env, [])
    pa, aira, sa = _air_time(env, ["a"])
    pu, airu, _ = _air_time(env, ["a", "up"])
    assert p0 == 0 and pa > 0, "A must lift the ship (z>0)"
    assert 1 in sa, "ship_state must report JUMP while airborne"
    # NOTE: the ROM source has NO separate A+UP long jump: UP only moves the ship up
    # while the normal jump happens, so the jump profile is identical to A alone.
    assert (pu, airu) == (pa, aira)
    # a brief tap of A gives a shorter jump than holding it
    env.reset(seed=0, options=NOSEED)
    env.step(mb("a"))
    zs = [float(env.step(mb())[0]["ship"][2]) for _ in range(100)]
    assert sum(z > 0 for z in zs) < aira, "tap should be shorter than hold"


def test_b_adds_save_state_costing_a_life(env):
    l0 = env.step(mb())[4]["lives"]
    env.step(mb("b"))
    info = env.step(mb())[4]
    assert info["lives"] == l0 - 1, "B must cost one life (save state)"
    info = hold(env, ["b"], 10)[1]  # held B triggers only once (edge-triggered)
    assert info["lives"] == l0 - 1


def test_discrete_table_matches_buttons():
    e = mk(allow_save_button=True, frame_skip=2)
    e.reset(seed=0, options=NOSEED)
    assert e.action_space.n == len(DISCRETE_ACTIONS_WITH_B)
    x0 = e.step(0)[0]["ship"][0]
    for _ in range(10):
        o = e.step(2)[0]  # RIGHT
    assert o["ship"][0] > x0
    for _ in range(30):
        o = e.step(1)[0]  # LEFT
    assert o["ship"][0] < x0
    assert e._buttons_for(10) == {"a", "up", "right"}
    assert e._buttons_for(11) == {"b"}
    e.close()


def test_crash_drops_lives_and_is_penalised():
    e = mk(frame_skip=2, max_steps=10**6)
    e.reset(seed=0, options=NOSEED)
    crashed, lives = False, []
    for _ in range(400):
        o, r, te, tr, info = e.step(2)  # keep steering right into the wall
        lives.append(info["lives"])
        if info["crashed"]:
            crashed = True
            assert r <= -1.0 and o["ship_state"] == 2
    assert crashed
    assert lives[-1] < 10, "crash (auto-rewind) must cost lives"
    e.close()


def test_obstacles_score_and_lives_come_from_ram():
    e = mk(frame_skip=4, max_steps=10**6)
    e.reset(seed=1)
    rng = np.random.default_rng(0)
    seen_obstacle, max_score = False, 0
    for _ in range(1500):
        o, r, te, tr, info = e.step(int(rng.integers(0, 11)))
        raw = e.get_raw_state()
        assert o["lives"] == raw["lives"] and info["score"] == raw["score"]
        assert abs(o["ship"][0] - raw["ship_x"]) < 1e-3
        if o["obstacles"][0, 0] == 1:
            seen_obstacle = True
            assert 0 <= o["obstacles"][0, 4] <= 144 and o["obstacles"][0, 1] in (0, 1, 2, 3, 4)
        max_score = max(max_score, info["score"])
        if te:
            e.reset()
    assert seen_obstacle, "obstacles array never populated"
    assert max_score >= 10 and max_score % 10 == 0, "score is +10 per cleared obstacle (BCD)"
    e.close()


def test_obstacle_y_increases_each_frame():
    e = mk(frame_skip=1, max_steps=10**6)
    e.reset(seed=1)
    ys = []
    for _ in range(600):
        o = e.step(0)[0]
        if o["obstacles"][0, 0] == 1:
            ys.append(o["obstacles"][0, 4])
            if len(ys) > 20:
                break
    assert len(ys) > 20 and ys[-1] > ys[0], "obstacle must scroll down the screen"
    e.close()
