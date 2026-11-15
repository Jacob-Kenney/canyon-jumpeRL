# Running Canyon Racer in Harbor with the trained PPO policy

The task lives in `canyon-racer-task/` (see its README for the sandbox and trust model). This page covers running it with our
trained policy. Commands run from this directory (`harbor/`). Jobs are written to `jobs/`.

## How the policy gets into the run

- `canyon-racer-task/environment/policy/policy.npz`: the exported network weights (numpy, no torch or ray needed).
- `canyon-racer-task/environment/policy/ppo_play.py`: standalone script that resets the game (seed 0), then loops
  `observation -> network -> buttons -> step` through the MCP server. It needs only `numpy` and `mcp`.
- The agent `Dockerfile` copies both into the agent container at `/opt/policy`.
- `instruction.md` tells the agent the policy is there and to run `python /opt/policy/ppo_play.py`.

The agent container never has the game or ROM. It only reaches the game through the `mcp-server` sidecar
(`http://mcp-server:8000/mcp`), which enforces the 1500-step budget and records every action. The verifier replays that
trace in a fresh emulator. `reward` is the best raw in-game score (10 per obstacle).

## Run it

Needs Docker and `harbor` on the PATH.

### 1. With an OpenCode agent (an LLM drives, and runs the policy)

```bash
export <PROVIDER>_API_KEY=...        # the key for the provider you pick, e.g. ANTHROPIC_API_KEY
harbor run -p ./canyon-racer-task -a opencode -m <provider/model>
```

Harbor writes the MCP server from `task.toml` into OpenCode's config, so the agent sees the `reset` and `step` tools. The agent
can run `/opt/policy/ppo_play.py`, or read it and `policy.npz` and build on them.

Status: this path has **not been run yet** (no API key was available when it was set up). Parts 2 and 3 below have been run.

Expect about the same score as part 2 if the agent just runs the script. The policy is deterministic and the budget is 1500
steps, so one run uses all of it.

### 2. Oracle, no LLM, no API key (tests the policy and the plumbing)

```bash
harbor run -p ./canyon-racer-task -a oracle --ae POLICY=ppo
```

`solution/solve.sh` runs `/opt/policy/ppo_play.py` when `POLICY=ppo`. Without that variable it runs the heuristic (`play.py`).
Last result: reward **230** (`jobs/2026-11-15__07-53-01/`). The earlier heuristic-only runs scored 0.575, from before the grader
reported raw score, so don't compare them directly.

### 3. Baselines

```bash
harbor run -p ./canyon-racer-task -a oracle          # heuristic policy
harbor run -p ./canyon-racer-task -a nop             # does nothing
```

## Read the results

```bash
cat jobs/<job>/result.json                          # mean reward
cat jobs/<job>/*/verifier/reward.txt                # the score
cat jobs/<job>/*/agent/oracle.txt                   # oracle's final game info
harbor view jobs                                    # browse in the UI
```

`jobs/<job>/*/artifacts/trace.jsonl` is the recorded action trace (`{"op": "step", "buttons": ["a"]}` per step).

## Use a different trained model

1. Train: `uv run python src/training/ppo.py --lr 3e-5 --iters 100 --out runs/<name>` from the repo root. This also writes
   `runs/<name>/stage2_ppo`.
2. Export it (use an absolute checkpoint path, since RLlib rejects relative ones):

   ```bash
   uv run python src/training/export_policy.py $PWD/runs/<name>/stage2_ppo harbor/canyon-racer-task/environment/policy/policy.npz
   ```

   The exporter checks the numpy network against the real RLlib module before saving.
3. Run part 2 or 1 above. The agent image rebuilds automatically, because `policy/` is copied in the Dockerfile.

The exported policy is deterministic: it presses button *i* when the network's action mean for *i* is > 0. The observation
(33 floats) and action (7 buttons) layouts are in `docs/ENV.md`. The network is a 256x256 tanh MLP, so it must be trained with the
default RLlib `fcnet_hiddens` for the exporter to read it. For other shapes, update `export_policy.py` and `ppo_play.py`.

## Notes

- The vendored game source in `canyon-racer-task/environment/mcp-server/game/` and `tests/game/` is a snapshot made by
  `canyon-racer-task/sync.sh`. Run `./canyon-racer-task/sync.sh` after changing `src/` or the ROM. Changing only the policy needs
  no sync.
- The current policy is a PPO fine-tune of an imitation of the heuristic. It scores about the same as the heuristic (a game score of
  about 100 in training, 230 best episode in the Harbor run). It is a baseline, not an improvement.
- The ROM is CC-BY-NC-SA 4.0, for non-commercial use only.
