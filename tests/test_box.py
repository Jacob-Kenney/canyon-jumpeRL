"""Default (Box) observation/action spaces and the action_offset latency option."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

import canyon_jumperl  # noqa: F401  (registers CanyonRacer-v0)
from canyon_jumperl.env import BUTTON_ORDER, OBS_DIM, OBS_LAYOUT, CanyonRacerEnv


def _act(*buttons):
    a = np.zeros(8, np.float32)
    for b in buttons:
        a[BUTTON_ORDER.index(b)] = 1.0
    return a


def test_default_spaces_are_boxes():
    env = gym.make("CanyonRacer-v0")
    assert isinstance(env.observation_space, spaces.Box) and env.observation_space.shape == (OBS_DIM,)
    assert isinstance(env.action_space, spaces.Box) and env.action_space.shape == (8,)
    assert env.observation_space.dtype == np.float32 and env.action_space.dtype == np.float32
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs[OBS_LAYOUT["lives"]] == 10 and obs[OBS_LAYOUT["ship_state"]] == 0
    obs, r, te, tr, info = env.step(_act("a", "right"))
    assert env.observation_space.contains(obs) and isinstance(r, float)
    env.close()


def test_box_env_passes_check_env():
    env = CanyonRacerEnv(max_steps=30)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_box_action_threshold():
    env = CanyonRacerEnv(frame_skip=1, max_steps=10**6)
    env.reset(seed=0)
    x0 = env.step(np.zeros(8, np.float32))[0][OBS_LAYOUT["ship"]][0]
    weak = env.step(np.full(8, 0.0, np.float32) + 0.3 * np.eye(8, dtype=np.float32)[0])[0][OBS_LAYOUT["ship"]][0]
    assert weak == x0                      # 0.3 < 0.5: not pressed
    for _ in range(5):
        obs = env.step(_act("right"))[0]
    assert obs[OBS_LAYOUT["ship"]][0] > x0  # 1.0 > 0.5: pressed
    env.close()


def test_action_offset_delays_input_by_k_frames():
    """The buttons reach the emulator exactly k frames later; the ship reacts no earlier than that."""
    k, n = 3, 16
    first_move, fed = {}, {}
    for off in (0, k):
        env = CanyonRacerEnv(frame_skip=1, action_offset=off, max_steps=10**6)
        obs, _ = env.reset(seed=0)
        log, orig = [], env._set_buttons
        env._set_buttons = lambda want, log=log, orig=orig: (log.append(set(want)), orig(want))[1]
        x_reset = obs[OBS_LAYOUT["ship"]][0]
        xs = [env.step(_act("right"))[0][OBS_LAYOUT["ship"]][0] for _ in range(n)]
        env.close()
        fed[off] = log
        assert xs[-1] > x_reset                       # holding right moves the ship
        first_move[off] = next(t for t, x in enumerate(xs) if x != x_reset)
    assert fed[0] == [{"right"}] * n
    assert fed[k] == [set()] * k + [{"right"}] * (n - k)   # exactly k frames of delay at the emulator input
    # the game samples input on its own frame phase, so the visible reaction can lag by one more frame
    assert first_move[k] >= first_move[0] + k, first_move
