"""MCP server exposing the game interface (actions + representation) of ``CanyonRacerEnv``.

The server is a *game interface*, not a training environment: it serves the canonical observation
dict and action encodings of ``gym.make("CanyonRacer-v0")`` with its default settings, and reports game
facts.  There is NO reward, NO truncation and NO step budget here: those belong to the training code,
which computes them from obs/info.  ``terminated`` means the GAME is over.

    canyon-racer-mcp [--rom PATH] [--backend emulator|hardware]            # stdio
    canyon-racer-mcp --http --host 127.0.0.1 --port 8000                    # streamable HTTP at /mcp

Tools: reset(seed) and step(action) only. The action encoding is in step's description.
step(action): a list of 7 floats in [0,1] (pressed if > 0.5) in order
[RIGHT,LEFT,UP,DOWN,A,B,SELECT]. There is no START button (pause is not available).  Buttons are held for the env's default frame count (frame_skip=4).
Results: {"observation", "terminated", "info"}; info = {lives, score, level, ship_state, paused, game_over,
game_state, frames}.

Wire encoding of numpy values (lossless, see :func:`encode` / :func:`decode`)
-----------------------------------------------------------------------------
* ndarray / numpy scalar -> ``{"dtype": "float32", "shape": [..], "data": <nested list>}`` (scalars: shape []).
  float32 -> JSON double is exact, so decoding gives bit-identical float32.
* uint8 arrays with > 4096 elements use ``{"dtype", "shape", "b64"}`` (not used by the default state observation).
Written for mcp 2.x (``mcp.server.mcpserver.MCPServer``, called ``FastMCP`` in 1.x).
"""
from __future__ import annotations

import argparse
import base64
import io
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from gymnasium import spaces
from mcp.server.mcpserver import Image, MCPServer

from .env import BOX_PRESS_THRESHOLD, BUTTON_ORDER, GAME_STATE_RUNNING, N_BUTTONS, OBS_LAYOUT, CanyonRacerEnv

B64_THRESHOLD = 4096


# ------------------------------------------------------------------ codec
def encode(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: encode(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [encode(v) for v in x]
    if isinstance(x, np.generic):
        x = np.asarray(x)
    if isinstance(x, np.ndarray):
        if x.dtype == np.uint8 and x.size > B64_THRESHOLD:
            return {"dtype": "uint8", "shape": list(x.shape),
                    "b64": base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()}
        return {"dtype": str(x.dtype), "shape": list(x.shape), "data": x.tolist()}
    return x


def decode(x: Any) -> Any:
    """Inverse of :func:`encode` (dicts without dtype/shape keys are plain dicts)."""
    if isinstance(x, dict):
        if "dtype" in x and "shape" in x and ("data" in x or "b64" in x):
            if "b64" in x:
                a = np.frombuffer(base64.b64decode(x["b64"]), dtype=x["dtype"])
                return a.reshape(x["shape"]).copy()
            return np.array(x["data"], dtype=x["dtype"]).reshape(x["shape"])
        return {k: decode(v) for k, v in x.items()}
    if isinstance(x, list):
        return [decode(v) for v in x]
    return x


def describe_space(sp: spaces.Space) -> dict[str, Any]:
    """Machine-readable space description."""
    if isinstance(sp, spaces.Dict):
        return {"type": "Dict", "spaces": {k: describe_space(v) for k, v in sp.spaces.items()}}
    if isinstance(sp, spaces.Box):
        return {"type": "Box", "shape": list(sp.shape), "dtype": str(sp.dtype),
                "low": np.unique(sp.low).tolist(), "high": np.unique(sp.high).tolist()}
    if isinstance(sp, spaces.Discrete):
        return {"type": "Discrete", "n": int(sp.n), "start": int(sp.start), "dtype": str(sp.dtype)}
    if isinstance(sp, spaces.MultiBinary):
        return {"type": "MultiBinary", "n": int(np.prod(sp.n)), "shape": list(sp.shape), "dtype": str(sp.dtype)}
    raise TypeError(f"unsupported space {sp!r}")


# ------------------------------------------------------------------ backends
INFO_KEYS = ("lives", "score", "level", "ship_state", "paused", "game_state", "frames")


def game_facts(info: dict) -> dict[str, Any]:
    """Game facts only (no training signals)."""
    out = {k: info[k] for k in INFO_KEYS}
    out["game_over"] = info["game_state"] != GAME_STATE_RUNNING
    return out


class GameBackend(ABC):
    """Game interface: canonical observation + actions, game-over flag, game facts. No reward."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[Any, dict]: ...

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, bool, dict]:
        """action: int (Discrete index) or list of 7 bits (MultiBinary). -> (obs, game_over, info)."""

    @abstractmethod
    def step_buttons(self, buttons: list[str]) -> tuple[Any, bool, dict]: ...

    @abstractmethod
    def spaces(self) -> dict[str, Any]: ...

    @abstractmethod
    def screen_png(self) -> bytes: ...

    def close(self) -> None:  # pragma: no cover
        pass


class EmulatorBackend(GameBackend):
    """Wraps CanyonRacerEnv with its canonical default settings (state obs, frame_skip=4)."""

    def __init__(self, rom_path: str | None = None):
        # max_steps only disables the env's own truncation (not a game concept); render_mode only enables screen capture.
        self.env = CanyonRacerEnv(rom_path=rom_path, max_steps=10**9, render_mode="rgb_array")
        self._started = False

    def reset(self, seed=None):
        self._started = True
        obs, info = self.env.reset(seed=seed)
        return obs, game_facts(info)

    def _ensure(self):
        if not self._started:
            self.reset()

    def _step(self, buttons):
        self._ensure()
        obs, _reward, terminated, _trunc, info = self.env.step_buttons(set(buttons), self.env.frame_skip)
        return obs, bool(terminated), game_facts(info)

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (N_BUTTONS,) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
            raise ValueError("action must be a list of 7 floats in [0,1]: [RIGHT,LEFT,UP,DOWN,A,B,SELECT]")
        return self._step({BUTTON_ORDER[i] for i in range(N_BUTTONS) if a[i] > BOX_PRESS_THRESHOLD})

    def step_buttons(self, buttons):
        bad = [b for b in buttons if b not in BUTTON_ORDER]
        if bad:
            raise ValueError(f"unknown buttons {bad}; valid: {BUTTON_ORDER}")
        return self._step(set(buttons))

    def spaces(self):
        env = self.env
        return {
            "observation_space": describe_space(env.observation_space),
            "observation_layout": {k: ([v.start, v.stop] if isinstance(v, slice) else v)
                                   for k, v in OBS_LAYOUT.items()},
            "action_space": {**describe_space(env.action_space), "order": BUTTON_ORDER,
                             "press_threshold": BOX_PRESS_THRESHOLD},
            "backend": "emulator",
        }

    def screen_png(self):
        from PIL import Image as PILImage
        self._ensure()
        buf = io.BytesIO()
        PILImage.fromarray(self.env.screen()).resize((320, 288), PILImage.NEAREST).save(buf, format="PNG")
        return buf.getvalue()

    def close(self):
        self.env.close()


class HardwareBackend(GameBackend):
    """Explicit stub for a real console (ModRetro Chromatic) driven over the link cable.

    Planned mapping (to be aligned with docs/LINK_PROTOCOL.md, which does not exist yet):

    * step(action): Discrete index -> button set via the same table, or MultiBinary(7) bits in
      BUTTON_ORDER -> the 8-bit GBDK joypad byte (RIGHT 0x01 LEFT 0x02 UP 0x04 DOWN 0x08 A 0x10 B 0x20
      SELECT 0x40 START 0x80), sent as a link command byte; the ROM holds it for frame_skip frames and replies.
    * observation: the reply "link state packet" is parsed into the identical dict: ship (x,y,z,vz),
      ship_state, lives (BCD), score (BCD), level, speed.  ``obstacles`` / ``canyon_left`` require the ROM to
      pack the obstacle ring buffer / scroll samples (or the host reconstructs them).
    * info/terminated: game facts and game_over come straight from the packet (game_state, lives).
      reset(seed) needs a ROM-side "restart round + set RNG seed" command.
    """

    def _na(self):
        raise NotImplementedError("real-hardware backend is not implemented yet")

    def reset(self, seed=None): self._na()
    def step(self, action): self._na()
    def step_buttons(self, buttons): self._na()
    def spaces(self): self._na()
    def screen_png(self): self._na()


# ------------------------------------------------------------------ server
def build_server(backend: GameBackend, **server_kwargs) -> MCPServer:
    mcp = MCPServer("canyon-racer", **server_kwargs)

    def _pack(obs, terminated, info):
        return {"observation": encode(obs), "terminated": bool(terminated), "info": encode(info)}

    # The encodings live in the step tool's description, so no separate "spaces" tool is needed.
    layout = "; ".join(f"{k}={'[%d:%d]' % (v.start, v.stop) if isinstance(v, slice) else '[%d]' % v}"
                       for k, v in OBS_LAYOUT.items())
    step_doc = (
        "Press buttons for 4 frames and return the result. `action` = list of 7 floats in [0,1] "
        "[RIGHT,LEFT,UP,DOWN,A,B,SELECT] (no START); a button is pressed when its value is > 0.5 (e.g. jump = A). "
        "Returns {observation, terminated (true when the game is over), info (lives, score, level, ship_state, "
        "paused, game_over, game_state, frames)}. `observation` is a flat float32 feature vector (shape [33]) with layout "
        f"{layout}. ship = x/100, y/100, z/100, vz/3; crashing = 1 while the ship is crashing (buttons do nothing); "
        "canyon_left_rel = (left wall x at y=0,16,..,144 - ship x)/100; obstacles = 2 nearest, 9 values each: present, "
        "one-hot type [left, right, middle, full width, +1 life item], (hit_x_left - ship_x)/100, (hit_x_right - ship_x)/100, "
        "(ship_y - obstacle_y)/100 (positive while the obstacle is still ahead of the ship); all zeros if absent. "
        "No reward or truncation: compute those yourself."
    )

    @mcp.tool()
    def reset(seed: int | None = None) -> dict:
        """Start a new game. Same seed => same game for the same inputs. Returns the same shape as step()."""
        obs, info = backend.reset(seed)
        return _pack(obs, info["game_over"], info)

    @mcp.tool(description=step_doc)
    def step(action: list[float]) -> dict:
        return _pack(*backend.step(action))

    return mcp


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Canyon Racer MCP game interface")
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backend", choices=["emulator", "hardware"], default="emulator")
    ap.add_argument("--rom", default=None)
    args = ap.parse_args(argv)
    backend = EmulatorBackend(args.rom) if args.backend == "emulator" else HardwareBackend()
    server = build_server(backend)
    if args.http:
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
