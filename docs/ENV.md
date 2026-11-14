# Canyon Racer environment

`gym.make("CanyonRacer-v0")` -> `canyon_jumperl.env.CanyonRacerEnv` (PyBoy headless, state read from emulator RAM).
Defaults are `observation_mode="box"` (flat `Box(44)`) and `action_mode="box"` (`Box(8)`). The `"state"` Dict, `"pixels"`, `"discrete"` and
`"multibinary"` modes are options. The tables below describe the Dict/discrete modes. The Box layout is in the last section.
ROM: `roms/canyon_racer_0.7.7_mbc5_rumble.gb` (+ `.noi`/`.cdb` next to it; override with `rom_path=` or `$CANYON_RACER_ROM`).

```python
import gymnasium as gym, canyon_jumperl
env = gym.make("CanyonRacer-v0", frame_skip=4, max_steps=2000)
obs, info = env.reset(seed=0)
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
```

## Constructor args
`rom_path, observation_mode="box"|"state"|"pixels"|"both", action_mode="box"|"discrete"|"multibinary", allow_save_button=False,
action_offset=0, frame_skip=4, max_steps=2000, render_mode=None|"rgb_array", crash_penalty=1.0, life_loss_penalty=0.5,
life_gain_bonus=1.0, score_reward_scale=0.1, game_over_penalty=1.0, settle_frames=60`.

## Actions -> real joypad buttons
Buttons are held for all `frame_skip` frames of a step (GBDK bits: RIGHT 01, LEFT 02, UP 04, DOWN 08, A 10, B 20, SELECT 40, START 80).

`multibinary`: `MultiBinary(8)` in order `[RIGHT, LEFT, UP, DOWN, A, B, SELECT, START]`. START pauses the game (toggle).

`discrete`:
| id | buttons | id | buttons |
|---|---|---|---|
| 0 | noop | 6 | A+LEFT |
| 1 | LEFT | 7 | A+RIGHT |
| 2 | RIGHT | 8 | A+UP |
| 3 | UP | 9 | A+UP+LEFT |
| 4 | DOWN | 10 | A+UP+RIGHT |
| 5 | A (jump; hold = higher) | 11 | B (only with `allow_save_button=True`) |

Verified against the real ROM in `tests/test_wiring.py`: D-pad moves x/y (0.5 px/frame), A raises z and sets ship_state=JUMP
(held A jumps longer than a tap), B costs exactly one life (edge-triggered), START pauses (ship frozen, `info["paused"]`).
**Finding:** the ROM source (`entity_ship.c`) has no separate "A+UP extra-long jump"; UP just moves the ship up while the normal
jump happens (the jump profile of A and A+UP is identical, and a test asserts this). Actions 8-10 are kept for convenience.

## Observation (`observation_mode="state"`: Dict; `"box"` is this dict flattened, see the last section)
| key | space | source (RAM) |
|---|---|---|
| `ship` | Box(4) x, y, z (px), vz (px/frame) | `state.ship_x/ship_y/ship_z` (8.8 fixed), `ship_jump_velocity` |
| `ship_state` | Discrete(4): 0 ground, 1 jump, 2 crashing, 3 crash-ended | `state.ship_state` (0x10/0x11/0x22/0x23) |
| `canyon_left` | Box(10) left wall x at y=0,16,..,144 | ROM scroll table via `state.p_scx_table_frame_base`: `64 - table[y]` |
| `lives` | Discrete(100) | `lives_count` (BCD) = remaining save states/rewinds |
| `score` | Box(1) | `state.score` (4-byte BCD, +10 per obstacle cleared) |
| `level` | Discrete(256) | `state.game_level` |
| `speed` | Box(1) obstacle px/frame | `state.cur_level.obst_speed` (8.8) |
| `obstacles` | Box(5,5): `[present, type, hit_x_left, hit_x_right, y]`, nearest first | `state.obstacles[]` ring buffer (`obstacles_active_first/count`); type 0 L,1 R,2 mid,3 full,4 +1 item; x from canyon table + ROM hitbox offsets |

`pixels`: Box(144,160,3) uint8; `both`: state dict plus `pixels`.

RAM symbols (from `.noi`/`.cdb`, parsed in `src/canyon_jumperl/symbols.py`, nothing hardcoded; addresses for this build):
`state` 0xC489 (struct `game_state_data`, 81 bytes; field offsets from the .cdb, e.g. ship_x +15, ship_y +17, ship_z +19, ship_state +21,
ship_jump_velocity +23, obstacles +25, score +77, game_level +3, paused +2, cur_level +5), `lives_count` 0xC537, `game_state` 0xC53D
(4 = RUNNING, other = not in a round), `__rand_seed` 0xC89B, `rand_seed_copy` 0xC52B.

## Reward / termination
`reward = dScore*score_reward_scale (default 0.1 = +1 per obstacle; 1.0 makes the episode return equal the game score) + life_gain_bonus*lives_gained - life_loss_penalty*lives_lost - crash_penalty (once, when crash starts) - game_over_penalty (on terminal)`.
`terminated` when the ROM leaves the running state (game over: crash with 0 lives). `truncated` after `max_steps` agent steps.
A crash auto-rewinds to the last save point (costing a life) after ~2-3 s of non-controllable frames. `info`: steps, frames, score, lives, level,
ship_state, paused, game_state, game_seed, crashed.

## Reset / determinism
reset() restores a PyBoy snapshot taken once gameplay has started (title/intro skipped by pressing START only while not in-game).
The ROM does `initrand(0x1234)` each round, so the stock game is deterministic. `reset(seed=s)` overwrites the game RNG
(`__rand_seed`, its save-point copy, and the queued first obstacle type) from `s`; same seed + same actions => identical trajectory
(tested across resets and across fresh env instances). `reset(options={"game_seed": None})` keeps the stock sequence. `reset()` with no
seed draws a new game seed from `env.np_random`. Not randomised: first-obstacle delay (180 frames) and initial map scroll.

## MCP server (`canyon-racer-mcp`, `src/canyon_jumperl/mcp_server.py`)
A *game interface*, not a training environment. It serves the canonical representation of `gym.make("CanyonRacer-v0")` with default
settings (`frame_skip=4`; see the last section for the observation and action encoding) and reports game facts. There is **no reward,
no truncation and no step budget**: those belong in the training code, computed from obs/info. `terminated` means the game is over.

Flags (operational only): `--rom PATH`, `--backend emulator|hardware`, `--http [--host H] [--port P]` (streamable HTTP at `/mcp`; stdio by default).
Uses mcp 2.x (`mcp.server.mcpserver.MCPServer`, formerly FastMCP).

Tools: `reset(seed)` (new game) and `step(action)` only. The action encoding is in the `step` tool description.
`step(action)`: a list of 8 floats in [0,1] (the Box encoding, pressed if > 0.5) in the order `[RIGHT,LEFT,UP,DOWN,A,B,SELECT,START]`.
Buttons are held for the default frame count. Returns `{"observation", "terminated", "info"}`; `info` = `lives, score, level, ship_state, paused,
game_over, game_state, frames`.

Wire encoding (lossless; `encode`/`decode` in `mcp_server.py`): numpy array/scalar -> `{"dtype","shape","data"}` (scalars shape `[]`, e.g. Discrete
observation entries are int64); float32 -> JSON double is exact, so decoded float32 is bit-identical. Plain JSON for everything else.

`EmulatorBackend` wraps `CanyonRacerEnv` (its own truncation disabled). `HardwareBackend` is an explicit stub with the same interface,
documenting the planned mapping to link-cable command bytes and link state packet -> identical obs dict (`docs/LINK_PROTOCOL.md` does not exist yet; the protocol is implemented in `rom/canyon-racer-link/src/link.h` and `rom/tools/link_client.py`).

Portability to hardware: portable from a small packet: `ship`, `ship_state`, `lives`, `score`, `level`, `speed`. Needs extra ROM packing or host
reconstruction: `obstacles`, `canyon_left`. Not available: `pixels`. Reward/truncation are the user's (host-side) anyway; `reset(seed)` needs a
ROM-side restart + RNG seed command.

Example: `examples/play_via_mcp.py` (policy drives the game via MCP; it computes its own reward and truncation).

## Tests
`cd ~/canyon-jumpeRL && .venv/bin/python -m pytest tests -q` (~25 s): `test_env.py` (symbols, check_env, determinism, random smoke,
game over, truncation), `test_wiring.py` (buttons -> game, RAM -> obs), `test_mcp.py` (one gym-vs-MCP parity run through a real stdio client: obs bit-for-bit, game over, game facts; plus spaces/errors).
No training code is included.

## Box spaces and action offset (current defaults)

`CanyonRacer-v0` now defaults to Box spaces (`observation_mode="box"`, `action_mode="box"`); the older `"state"` (Dict),
`"pixels"`, `"both"`, `"discrete"` and `"multibinary"` modes remain available as options.

* **Observation:** `Box(float32, (44,))`. Layout (`canyon_jumperl.env.OBS_LAYOUT`): `ship` [0:4] = x, y, z, vz;
  `ship_state` [4]; `canyon_left` [5:15]; `lives` [15]; `score` [16]; `level` [17]; `speed` [18]; `obstacles` [19:44]
  = 5 rows x [present, type, hit_x_left, hit_x_right, y], nearest first. Bounds are in `observation_space`.
* **Action:** `Box(0, 1, (8,), float32)` in `BUTTON_ORDER` = RIGHT, LEFT, UP, DOWN, A, B, SELECT, START. A button is
  pressed when its value is > 0.5 (`BOX_PRESS_THRESHOLD`). B costs a life and START pauses, so a policy can hurt itself.
* **`action_offset=k` (frames, default 0):** latency simulation. The buttons chosen at a step reach the emulator k frames
  later (a FIFO fed one entry per frame); for the first k frames after a reset no buttons are pressed. The game samples
  input on its own frame phase, so the visible reaction can lag one frame more than k.
* The MCP server serves these defaults: `step(action)` takes the 8 floats and returns the 44-vector observation.
  It has no offset flag; on real hardware the latency is physical.
