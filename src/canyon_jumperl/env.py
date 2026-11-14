"""Gymnasium environment for the Canyon Racer Game Boy ROM, driven by PyBoy.

State is read straight from emulator RAM (addresses come from the ROM's
.noi/.cdb symbol files, see :mod:`canyon_jumperl.symbols`).

Determinism / RNG
-----------------
The ROM calls ``initrand(0x1234)`` at the start of every round, so the stock
game is fully deterministic given the inputs.  To make ``reset(seed=...)``
meaningful we snapshot the emulator once gameplay has started and, on every
reset, restore the snapshot and then overwrite the game RNG state
(``__rand_seed`` and its save-point copy ``rand_seed_copy``) plus the already
queued first obstacle type with values derived from ``seed``.  Same seed +
same actions => identical trajectory.  ``reset(options={"game_seed": None})``
keeps the stock ROM seed (0x1234).  Limitation: the first obstacle spawn
countdown (fixed 180 frames) and the initial map scroll are not randomised.
"""
from __future__ import annotations

import collections
import io
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .symbols import Symbols

ROM_NAME = "canyon_racer_0.7.7_mbc5_rumble"
DEFAULT_ROM_DIR = Path(__file__).resolve().parents[2] / "roms"

# GBDK joypad bit values
BUTTON_BITS = {"right": 0x01, "left": 0x02, "up": 0x04, "down": 0x08,
               "a": 0x10, "b": 0x20, "select": 0x40, "start": 0x80}
BUTTON_ORDER = ["right", "left", "up", "down", "a", "b", "select", "start"]

# Discrete action table (buttons held for the whole frame_skip window)
DISCRETE_ACTIONS: list[tuple[str, ...]] = [
    (), ("left",), ("right",), ("up",), ("down",),
    ("a",), ("a", "left"), ("a", "right"),                   # normal jump (hold = longer)
    ("a", "up"), ("a", "up", "left"), ("a", "up", "right"),  # A+Up = extra-long jump
]
DISCRETE_ACTIONS_WITH_B = DISCRETE_ACTIONS + [("b",)]  # B: add save state (costs a life)

# ROM constants (src/common.h, entity_ship.h, entity_obstacles.h)
GAME_STATE_RUNNING = 4
SHIP_STATE_NAMES = {0x10: 0, 0x11: 1, 0x22: 2, 0x23: 3}  # ground, jump, crashing, crash ended
SHIP_CRASHING = (2, 3)
OBSTACLES_COUNT = 5
OBJECTS_FLAG_HIDDEN_BIT = 0x80
OBJECT_TYPE_ITEM_PLUS_1 = 0x04
CANYON_LEFT_X_BASE = 64
# (left, right) hitbox x offsets per type: left,right,middle,full,item (sprite width 32)
OBST_HITBOX_X = {0: (0, 8), 1: (24, 32), 2: (8, 24), 3: (0, 32), 4: (8, 24)}
CANYON_SAMPLE_Y = np.arange(0, 160, 16)  # y rows at which the canyon left edge is sampled
SCREEN_H, SCREEN_W = 144, 160
BOX_PRESS_THRESHOLD = 0.5  # action_mode="box": button pressed when value > this

# observation_mode="box": flat float32 feature vector (engineered, see docs/ENV.md). Fixed scales, no clipping, so values
# outside the usual range stay valid. Dropped vs the raw state: level, speed, score, lives, obstacle rows 3-5, ship_state.
POS_SCALE = 100.0      # pixel positions / distances are divided by this
VZ_SCALE = 3.0         # jump velocity (px/frame, about -2..3) is divided by this
OBSTACLES_IN_OBS = 2   # nearest obstacles in the feature vector (the game never showed more than 2 at once)
OBSTACLE_FEATURES = 1 + 5 + 3  # present, one-hot type (L, R, mid, full, +1 item), dx_left, dx_right, dy
OBS_DIM = 4 + 1 + len(CANYON_SAMPLE_Y) + OBSTACLES_IN_OBS * OBSTACLE_FEATURES  # = 33
OBS_LAYOUT = {
    "ship": slice(0, 4),            # x/100, y/100, z/100, vz/3
    "crashing": 4,                  # 1 while the ship is crashing / waiting for the rewind (buttons do nothing), else 0
    "canyon_left_rel": slice(5, 15),  # (left wall x at y = 0,16,..,144 - ship x) / 100
    # 2 rows x 9, nearest first: present, one-hot type [L, R, mid, full, +1 item], (hit_x_left - ship_x)/100,
    # (hit_x_right - ship_x)/100, (ship_y - obstacle_y)/100 (positive while the obstacle is still ahead of the ship)
    "obstacles": slice(15, 33),
}

_SNAPSHOT_CACHE: dict[tuple, bytes] = {}


def _bcd(bs) -> int:
    v = 0
    for b in reversed(list(bs)):
        v = v * 100 + (b >> 4) * 10 + (b & 0xF)
    return v


def resolve_rom_path(rom_path: str | os.PathLike | None = None) -> Path:
    return Path(rom_path or os.environ.get("CANYON_RACER_ROM") or (DEFAULT_ROM_DIR / f"{ROM_NAME}.gb"))


class CanyonRacerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str | os.PathLike | None = None,
        observation_mode: str = "box",        # "box" (engineered feature vector, default) | "state" (Dict) | "pixels" | "both"
        action_mode: str = "box",             # "box" (8 floats in [0,1], pressed if > 0.5) | "discrete" | "multibinary"
        allow_save_button: bool = False,      # include B in the discrete action table
        action_offset: int = 0,               # latency: a step's buttons take effect this many frames later
        frame_skip: int = 4,
        max_steps: int = 2000,
        render_mode: str | None = None,
        crash_penalty: float = 1.0,           # applied once when the ship starts crashing
        life_loss_penalty: float = 0.5,       # per life lost (crash rewind or B save)
        life_gain_bonus: float = 1.0,         # per +1 item collected
        score_reward_scale: float = 0.1,      # reward per game-score point (0.1 = +1 per obstacle, 1.0 = reward is the game score)
        game_over_penalty: float = 1.0,
        settle_frames: int = 60,              # frames run after gameplay start before the snapshot
    ):
        super().__init__()
        assert observation_mode in ("box", "state", "pixels", "both")
        assert action_mode in ("box", "discrete", "multibinary")
        assert frame_skip >= 1
        assert action_offset >= 0
        self.rom_path = resolve_rom_path(rom_path)
        if not self.rom_path.exists():
            raise FileNotFoundError(self.rom_path)
        self.symbols = Symbols(self.rom_path.with_suffix(".noi"), self.rom_path.with_suffix(".cdb"))
        self._init_addresses()

        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.crash_penalty = crash_penalty
        self.life_loss_penalty = life_loss_penalty
        self.life_gain_bonus = life_gain_bonus
        self.score_reward_scale = score_reward_scale
        self.game_over_penalty = game_over_penalty
        self.settle_frames = settle_frames
        self.action_offset = action_offset
        self._queue: collections.deque = collections.deque()
        self._need_pixels = observation_mode in ("pixels", "both") or render_mode == "rgb_array"

        if action_mode == "discrete":
            self._action_table = DISCRETE_ACTIONS_WITH_B if allow_save_button else DISCRETE_ACTIONS
            self.action_space = spaces.Discrete(len(self._action_table))
        elif action_mode == "box":
            # one float per button in BUTTON_ORDER; a button is pressed when its value is > BOX_PRESS_THRESHOLD
            self.action_space = spaces.Box(0.0, 1.0, (8,), np.float32)
        else:
            self.action_space = spaces.MultiBinary(8)  # order: BUTTON_ORDER

        state_space = spaces.Dict({
            "ship": spaces.Box(-256.0, 256.0, (4,), np.float32),  # x, y, z (px), jump velocity (px/frame)
            "ship_state": spaces.Discrete(4),                      # 0 ground, 1 jump, 2 crashing, 3 crash ended
            "canyon_left": spaces.Box(0.0, 255.0, (len(CANYON_SAMPLE_Y),), np.float32),  # left wall x at y=0,16,..
            "lives": spaces.Discrete(100),
            "score": spaces.Box(0.0, 99999.0, (1,), np.float32),
            "level": spaces.Discrete(256),
            "speed": spaces.Box(0.0, 256.0, (1,), np.float32),     # obstacle speed px/frame
            # nearest-first: [present, type(0 L,1 R,2 mid,3 full,4 +1 item), hit_x_left, hit_x_right, y]
            "obstacles": spaces.Box(0.0, 255.0, (OBSTACLES_COUNT, 5), np.float32),
        })
        pix = spaces.Box(0, 255, (SCREEN_H, SCREEN_W, 3), np.uint8)
        if observation_mode == "box":
            self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)  # engineered, unbounded
        elif observation_mode == "state":
            self.observation_space = state_space
        elif observation_mode == "pixels":
            self.observation_space = pix
        else:
            self.observation_space = spaces.Dict({**state_space.spaces, "pixels": pix})

        self._pb = None
        self._held: set[str] = set()
        self._steps = 0
        self._frames = 0
        self._game_seed: int | None = None
        self._prev: dict[str, Any] = {}

    # ---------------------------------------------------------------- setup
    def _init_addresses(self):
        s = self.symbols
        self._a_state = s.addr("state")
        self._a_lives = s.addr("lives_count")
        self._a_game_state = s.addr("game_state")
        self._a_rand_seed = s.addr("__rand_seed")
        self._a_rand_copy = s.addr("rand_seed_copy")
        self._state_size = s.struct_size("game_state_data")
        self._f = {n: s.field("game_state_data", n) for n in (
            "game_level", "ship_x", "ship_y", "ship_z", "ship_state", "ship_counter", "ship_jump_velocity",
            "obstacles", "obstacles_active_first", "obstacles_active_count", "obstacles_next_type",
            "p_scx_table_frame_base", "score", "paused", "cur_level")}
        self._f_speed = self._f["cur_level"].offset + s.field("level_entry", "obst_speed").offset

    def _open(self):
        from pyboy import PyBoy
        pb = PyBoy(str(self.rom_path), window="null", sound_emulated=False)
        pb.set_emulation_speed(0)
        return pb

    @staticmethod
    def _tap(pb, button: str, hold=8, gap=20):
        pb.button_press(button)
        pb.tick(hold, False)
        pb.button_release(button)
        pb.tick(gap, False)

    def _boot_to_gameplay(self, pb) -> None:
        pb.tick(1400, False)  # splash / credits / intro
        for _ in range(60):
            if pb.memory[self._a_game_state] == GAME_STATE_RUNNING:
                break
            # START is only pressed while NOT in gameplay (START pauses in-game)
            self._tap(pb, "start")
        else:
            raise RuntimeError("could not reach gameplay from the title screen")
        pb.tick(self.settle_frames, False)

    def _ensure_pyboy(self):
        if self._pb is not None:
            return
        self._pb = self._open()
        key = (str(self.rom_path.resolve()), self.settle_frames)
        if key not in _SNAPSHOT_CACHE:
            self._boot_to_gameplay(self._pb)
            buf = io.BytesIO()
            self._pb.save_state(buf)
            _SNAPSHOT_CACHE[key] = buf.getvalue()
        self._snapshot = _SNAPSHOT_CACHE[key]

    # ---------------------------------------------------------------- RAM
    def _read_raw(self) -> dict[str, Any]:
        m = self._pb.memory
        st = m[self._a_state:self._a_state + self._state_size]
        f = self._f

        def u16(o):
            return st[o] | (st[o + 1] << 8)

        vz = u16(f["ship_jump_velocity"].offset)
        raw = {
            "game_state": m[self._a_game_state],
            "lives": _bcd([m[self._a_lives]]),
            "score": _bcd(st[f["score"].offset:f["score"].offset + 4]),
            "level": st[f["game_level"].offset],
            "ship_x": u16(f["ship_x"].offset) / 256.0,
            "ship_y": u16(f["ship_y"].offset) / 256.0,
            "ship_z": u16(f["ship_z"].offset) / 256.0,
            "ship_vz": (vz - 65536 if vz >= 32768 else vz) / 256.0,
            "ship_state_raw": st[f["ship_state"].offset],
            "ship_counter": st[f["ship_counter"].offset],
            "paused": int(bool(st[f["paused"].offset])),
            "speed": u16(self._f_speed) / 256.0,
            "scx_ptr": u16(f["p_scx_table_frame_base"].offset),
            "obs_first": st[f["obstacles_active_first"].offset],
            "obs_count": st[f["obstacles_active_count"].offset],
        }
        base = f["obstacles"].offset
        # obs_ent = {fixed y (2 bytes), uint8 type}; y pixel = high byte
        raw["obs_slots"] = [(st[base + 3 * i + 1], st[base + 3 * i + 2]) for i in range(OBSTACLES_COUNT)]
        return raw

    def _canyon_left(self, ptr: int, y: int) -> int:
        # scroll table lives in ROM bank 0; entries are x scroll offsets (uint8 wrap)
        return (CANYON_LEFT_X_BASE - self._pb.memory[ptr + (y & 0xFF)]) & 0xFF

    def _state_obs(self, raw) -> dict[str, Any]:
        ptr = raw["scx_ptr"]
        rows = []
        for k in range(min(raw["obs_count"], OBSTACLES_COUNT)):
            idx = (raw["obs_first"] + k) % OBSTACLES_COUNT
            y, t = raw["obs_slots"][idx]
            if t & OBJECTS_FLAG_HIDDEN_BIT:
                continue
            tt = 4 if (t & 0x07) == OBJECT_TYPE_ITEM_PLUS_1 else (t & 0x03)
            x0 = self._canyon_left(ptr, y)
            lo, hi = OBST_HITBOX_X[tt]
            rows.append((1.0, tt, min(x0 + lo, 255), min(x0 + hi, 255), y))
        rows.sort(key=lambda r: -r[4])  # nearest (largest y, closest to the ship) first
        obstacles = np.zeros((OBSTACLES_COUNT, 5), np.float32)
        for i, r in enumerate(rows[:OBSTACLES_COUNT]):
            obstacles[i] = r
        return {
            "ship": np.array([raw["ship_x"], raw["ship_y"], raw["ship_z"], raw["ship_vz"]], np.float32),
            "ship_state": np.int64(SHIP_STATE_NAMES.get(raw["ship_state_raw"], 0)),
            "canyon_left": np.array([self._canyon_left(ptr, int(y)) for y in CANYON_SAMPLE_Y], np.float32),
            "lives": np.int64(min(raw["lives"], 99)),
            "score": np.array([min(raw["score"], 99999)], np.float32),
            "level": np.int64(raw["level"]),
            "speed": np.array([raw["speed"]], np.float32),
            "obstacles": obstacles,
        }

    def _pixels(self) -> np.ndarray:
        return np.ascontiguousarray(self._pb.screen.ndarray[:, :, :3])

    @staticmethod
    def flatten_state(s: dict[str, Any]) -> np.ndarray:
        """Dict state -> engineered float32 feature vector, layout in OBS_LAYOUT."""
        x, y, z, vz = (float(v) for v in s["ship"])
        f = [x / POS_SCALE, y / POS_SCALE, z / POS_SCALE, vz / VZ_SCALE, float(int(s["ship_state"]) in SHIP_CRASHING)]
        f += [(float(c) - x) / POS_SCALE for c in s["canyon_left"]]
        for present, typ, xl, xr, oy in s["obstacles"][:OBSTACLES_IN_OBS]:
            if present:
                onehot = [1.0 if int(typ) == k else 0.0 for k in range(5)]
                f += [1.0, *onehot, (float(xl) - x) / POS_SCALE, (float(xr) - x) / POS_SCALE, (y - float(oy)) / POS_SCALE]
            else:
                f += [0.0] * OBSTACLE_FEATURES
        return np.array(f, np.float32)

    def _observe(self, raw):
        if self.observation_mode == "pixels":
            return self._pixels()
        if self.observation_mode == "box":
            return self.flatten_state(self._state_obs(raw))
        obs = self._state_obs(raw)
        if self.observation_mode == "both":
            obs["pixels"] = self._pixels()
        return obs

    def _info(self, raw) -> dict[str, Any]:
        return {
            "steps": self._steps, "frames": self._frames,
            "score": raw["score"], "lives": raw["lives"], "level": raw["level"],
            "ship_state": SHIP_STATE_NAMES.get(raw["ship_state_raw"], -1),
            "paused": bool(raw["paused"]), "game_state": raw["game_state"],
            "game_seed": self._game_seed,
        }

    # ---------------------------------------------------------------- input
    def _buttons_for(self, action) -> set[str]:
        if self.action_mode == "discrete":
            return set(self._action_table[int(action)])
        a = np.asarray(action).reshape(-1)
        if self.action_mode == "box":
            return {BUTTON_ORDER[i] for i in range(8) if a[i] > BOX_PRESS_THRESHOLD}
        return {BUTTON_ORDER[i] for i in range(8) if a[i]}

    def _set_buttons(self, want: set[str]):
        for b in self._held - want:
            self._pb.button_release(b)
        for b in want - self._held:
            self._pb.button_press(b)
        self._held = set(want)

    def advance(self, buttons, frames: int):
        """Hold ``buttons`` for ``frames`` emulator frames (no reward bookkeeping).

        With ``action_offset`` = k > 0 the buttons chosen now only reach the emulator k frames later (the first k
        frames after a reset see no buttons); every frame consumes one entry from a FIFO of requested button sets.
        """
        if self.action_offset == 0:
            self._set_buttons(set(buttons))
            if frames > 1:
                self._pb.tick(frames - 1, False)
            self._pb.tick(1, self._need_pixels)
        else:
            for i in range(frames):
                self._queue.append(set(buttons))
                self._set_buttons(self._queue.popleft() if len(self._queue) > self.action_offset else set())
                self._pb.tick(1, self._need_pixels and i == frames - 1)
        self._frames += frames

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._ensure_pyboy()
        self._set_buttons(set())
        self._queue.clear()
        self._pb.load_state(io.BytesIO(self._snapshot))
        options = options or {}
        gs = options["game_seed"] if "game_seed" in options else int(self.np_random.integers(1, 65536))
        self._game_seed = gs
        if gs is not None:  # see module docstring
            m = self._pb.memory
            for a in (self._a_rand_seed, self._a_rand_copy):
                m[a] = gs & 0xFF
                m[a + 1] = (gs >> 8) & 0xFF
            m[self._a_state + self._f["obstacles_next_type"].offset] = ((gs * 2654435761) >> 7) & 0x03
        self._steps = 0
        self._frames = 0
        self._pb.tick(1, self._need_pixels)  # make the screen buffer valid
        raw = self._read_raw()
        self._prev = raw
        return self._observe(raw), self._info(raw)

    def step(self, action):
        return self.step_buttons(self._buttons_for(action), self.frame_skip)

    def step_buttons(self, buttons, frames: int):
        """Like step() but with explicit button names and frame count (used by the MCP server)."""
        self.advance(buttons, frames)
        self._steps += 1
        raw = self._read_raw()
        prev = self._prev
        reward = (raw["score"] - prev["score"]) * self.score_reward_scale
        dl = raw["lives"] - prev["lives"]
        reward += self.life_gain_bonus * dl if dl > 0 else self.life_loss_penalty * dl
        was_crash = SHIP_STATE_NAMES.get(prev["ship_state_raw"], 0) in SHIP_CRASHING
        now_crash = SHIP_STATE_NAMES.get(raw["ship_state_raw"], 0) in SHIP_CRASHING
        crashed = now_crash and not was_crash
        if crashed:
            reward -= self.crash_penalty
        terminated = raw["game_state"] != GAME_STATE_RUNNING
        if terminated:
            reward -= self.game_over_penalty
        truncated = (not terminated) and self._steps >= self.max_steps
        self._prev = raw
        info = self._info(raw)
        info["crashed"] = crashed
        return self._observe(raw), float(reward), bool(terminated), bool(truncated), info

    def get_raw_state(self) -> dict[str, Any]:
        self._ensure_pyboy()
        return self._read_raw()

    def render(self):
        return self._pixels() if self.render_mode == "rgb_array" else None

    def screen(self) -> np.ndarray:
        """Current screen as (144,160,3) uint8 (fresh only if pixels are being rendered)."""
        return self._pixels()

    def close(self):
        if self._pb is not None:
            self._pb.stop(save=False)
            self._pb = None
