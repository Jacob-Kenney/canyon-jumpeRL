"""Authoritative scoring: replay the sidecar's action trace in a fresh emulator (seed 0) and score it.
reward = best raw in-game score over any episode (no normalisation, no cap)."""
import json, os, sys
from canyon_jumperl.mcp_server import EmulatorBackend

MAX_STEPS = 1500
trace, out = sys.argv[1], sys.argv[2]
recs = []
try:
    recs = [json.loads(l) for l in open(trace) if l.strip()]
except Exception as e:
    print("no/bad trace:", e)
be = EmulatorBackend(os.environ["CANYON_RACER_ROM"])
best, steps = 0, 0
for r in recs:
    if r["op"] == "reset":
        be.reset(0)
    elif r["op"] == "step":
        if steps >= MAX_STEPS:
            break
        steps += 1
        _, over, info = be.step_buttons(r["buttons"])
        best = max(best, info["score"])
        if over:
            pass  # game over: further steps in this episode are no-ops for scoring
reward = float(best)
print(f"steps={steps} best_score={best} reward={reward}")
open(out, "w").write(f"{reward}\n")
