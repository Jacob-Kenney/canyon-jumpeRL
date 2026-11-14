# Canyon Racer Harbor task

Run: `harbor run -p ./canyon-racer-task -a opencode -m <provider/model>` (needs that provider's API key; not tested here).
Oracle: `harbor run -p ./canyon-racer-task -a oracle`; baseline: `-a nop`.

Layout: `environment/` = agent container (python + mcp client only, no game code/ROM) + sidecar `mcp-server` (PyBoy + ROM +
`canyon-racer-mcp` wrapped by `server.py`, which logs every action to `/trace/trace.jsonl` and enforces 1500 steps, fixed seed 0).
The MCP server is declared in `task.toml` (`[[environment.mcp_servers]]`), which Harbor's OpenCode agent writes into
`opencode.json` as a remote MCP server. Run `./sync.sh` to re-vendor `src/` + ROM + pinned `constraints.txt` (from uv.lock) after
the project changes (needed before building).

Score: `reward = best raw in-game score over any episode` (10 per obstacle cleared; no normalisation or cap).

Trust model: the agent only reaches the sidecar over MCP; it cannot touch the emulator, ROM, trace or verifier. The trace is
copied (artifact, service `mcp-server`) into a separate verifier container (own image with the game and `tests/` baked in), which
replays it in a fresh emulator, so the reported in-game score is never trusted. Residual risks: the agent could exploit game bugs
via legal inputs (that is play); an agent with internet could download the ROM and plan offline (the seed is fixed, so offline
search is possible; this is not prevented); the ROM is CC-BY-NC-SA 4.0, non-commercial only.
