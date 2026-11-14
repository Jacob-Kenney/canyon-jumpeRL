# Canyon Racer RL — task list (rev 2)

Repo: ~/canyon-jumpeRL (uv, Python 3.14; gymnasium 1.2.2, pyboy 2.9.0, ray[rllib], torch already installed).
VM currently has NO gcc/make/rgbds/GBDK/docker/node.

## Inputs (confirmed)
- Game source: https://github.com/bbbbbr/canyon-racer  (C + GBDK-2020, Makefile build, v0.7.7; src/input.c, gameplay.c,
  entity_ship.c, lives.c, score.c; license is "other"/NOASSERTION -> read LICENSE before redistributing the ROM)
- Hardware: ModRetro Chromatic + "shiny relay" (link-cable <-> Wi-Fi relay; exact protocol still to be identified, see Q1)
- Harbor: https://github.com/laude-institute/harbor. Task = instruction.md, task.toml, environment/Dockerfile, tests/, solution/.
  Run with `harbor run -p ./my-task -a opencode -m <provider/model>`.
  Verifier must write ONE score in [0,1] to /logs/verifier/reward.txt (partial credit welcome).

## 0. Shared interface spec (do first; unblocks 1, 2, 3, 4)
- [ ] docs/SPEC.md = single source of truth for:
  - actions: A (jump), A+Up (extra long jump), B (save state, costs a life), D-pad, Start, Select; per-frame encoding
  - state packet: ship x/y, jump state, obstacle list, lives, score, crashed/rewind flags, level, frame counter; byte layout
  - link-cable command byte format + handshake (master/slave, clock, timing)
  - Wi-Fi stream format (transport, rate, schema, versioning) matching what the shiny relay can carry
  - reward + termination definition (score delta, life loss, game over)
- [ ] Inspect canyon-racer source to locate RAM symbols for the state fields (build a .map/.sym) -> shared by tasks 1, 2

## 1. ROM fork: link-cable commands in, game state out
- [ ] Install GBDK-2020 + make/gcc on the VM; reproduce the stock ROM build
- [ ] Fork bbbbbr/canyon-racer into rom/; read button commands from the serial port in src/input.c (keep joypad as fallback)
- [ ] Emit the state packet over the link port each frame (or each N frames)
- [ ] Identify the ModRetro shiny relay protocol (Q1) and match SPEC.md to it
- [ ] Host-side Wi-Fi client library (Python) for the stream + command channel
- [ ] Tests: ROM runs headless in PyBoy with a fake link peer; bytes match SPEC.md
- [ ] On-device test on the Chromatic once the above passes (needs your hardware)

## 2. PyBoy + Gymnasium environment
- [ ] CanyonRacerEnv(gymnasium.Env): observation_space/action_space from SPEC.md
- [ ] State from emulator RAM; actions -> PyBoy button events; frame-skip option
- [ ] Reward, termination/truncation, seeded deterministic reset
- [ ] Register env id; check_env + determinism tests
- [ ] Optional link-peer mode so the env speaks the same protocol as real hardware
- [ ] Smoke run: random policy + one RLlib/torch training step

## 3. MCP server interface (new)
- [ ] Pick the MCP Python SDK (FastMCP) and transport (stdio for local agents, streamable HTTP for remote); confirm with docs
- [ ] Tools (backed by CanyonRacerEnv): reset(seed), step(buttons, frames), get_state(), get_screen() (image), get_info()
- [ ] Resources: SPEC/controls help, current episode stats
- [ ] Backend switch: emulator (default) vs real Chromatic over the Wi-Fi relay
- [ ] Tests with an MCP client; confirm the tool schema is stable and the server is deterministic given a seed
- [ ] Check whether Harbor/OpenCode can declare MCP servers in task.toml; if so wire it in task 4

## 4. Harbor task (portability + submission format)
- [ ] Install harbor (`uv tool install harbor`) and docker on the VM, or test where docker exists
- [ ] Task layout: instruction.md, task.toml, environment/Dockerfile (pinned deps from uv.lock + built ROM), tests/, solution/
- [ ] tests/ verifier writes a single float in [0,1] to /logs/verifier/reward.txt (e.g. normalized score over N seeded episodes, partial credit)
- [ ] solution/ reference agent + oracle run to validate end to end
- [ ] Run with OpenCode: `harbor run -p ./canyon-racer-task -a opencode -m <provider/model>`
- [ ] Clean-machine check

## Dependencies / parallelism
- 0 first. Then 1 and 2 in parallel (2 uses emulated state, no hardware). 3 after 2's API is stable. 4 after 2 (and 3 if MCP is used).
- Each subagent writes only under its own dir: rom/ (1), src/canyon_jumperl/ (2), mcp/ (3), harbor/ (4).

## Open questions
1. ModRetro "shiny relay": what is it exactly (docs/link)? I need its protocol/API before finalizing the Wi-Fi format.
2. Should the Harbor agent play via the MCP server, or interact with the environment another way (e.g. CLI/files in the container)?
3. Is the ROM allowed to be bundled in the submission, given the repo's license?
