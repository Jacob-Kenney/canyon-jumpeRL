# canyon-jumpeRL

A reinforcement-learning environment for [Canyon Racer](https://github.com/bbbbbr/canyon-racer), a Game Boy game by bbbbbr.
The ROM runs headless in [PyBoy](https://github.com/Baekalfen/PyBoy). The game state is read straight from emulator RAM and exposed as a
[Gymnasium](https://gymnasium.farama.org/) environment and as an MCP server.

| Path | What it is |
|---|---|
| `src/canyon_jumperl/` | `env.py` (the `CanyonRacer-v0` Gymnasium env), `symbols.py` (RAM symbols from `.noi`/`.cdb`), `mcp_server.py` (MCP game interface) |
| `docs/ENV.md` | Full reference: observation and action layout, reward, determinism, MCP tools |
| `roms/` | The built ROM `canyon_racer_0.7.7_mbc5_rumble.gb` and its symbol files |
| `rom/` | ROM sources and tools. See [rom/README.md](rom/README.md) |
| `harbor/canyon-racer-task/` | A [Harbor](https://github.com/laude-institute/harbor) task that has an agent play over MCP. See its [README](harbor/canyon-racer-task/README.md) |
| `examples/` | `play_via_mcp.py`, a policy driving the game through the MCP server |
| `tests/` | pytest suite |
| `TASKS.md` | Original planning notes. Parts are out of date |

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Use

```python
import gymnasium as gym, canyon_jumperl

env = gym.make("CanyonRacer-v0")        # Box(33) feature vector observation, Box(7) action (no START)
obs, info = env.reset(seed=0)
terminated = truncated = False
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

MCP server (stdio by default, or `--http` for streamable HTTP at `/mcp`):

```bash
uv run canyon-racer-mcp [--rom PATH] [--http --host 127.0.0.1 --port 8000] [--backend emulator|hardware]
uv run python examples/play_via_mcp.py
```

The `hardware` backend is a stub. It is the planned mapping to a real Game Boy over the link cable.

## Tests

```bash
uv run pytest tests -q     # about 25 s
```

## Status

- The environment, MCP server, and Harbor task work. No training code is included.
- The ROM has a link-cable fork under `rom/canyon-racer-link` (commands in, state out). It is not wired to hardware yet.
  The protocol doc `docs/LINK_PROTOCOL.md` that the code refers to has not been written.

## License and credits

The game is Canyon Racer, Copyright 2023 bbbbbr, licensed **CC-BY-NC-SA 4.0**. The ROM in `roms/` and the sources in `rom/` are derived from it
and carry that license: attribution, non-commercial use only, share-alike. The Python code in this repo is shared under the same terms.
