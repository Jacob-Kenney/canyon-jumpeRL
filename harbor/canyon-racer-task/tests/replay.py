"""Authoritative scoring: replay the sidecar's action trace in a fresh emulator (seed 0) and score it.
score = best in-game score over any episode / 10 = obstacles cleared. reward = min(1, cleared / REF)."""
import json, os, sys
from canyon_jumperl.mcp_server import EmulatorBackend

REF = float(os.environ.get("REF_OBSTACLES", "40"))
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
cleared = best / 10
reward = max(0.0, min(1.0, cleared / REF))
print(f"steps={steps} best_score={best} obstacles_cleared={cleared} ref={REF} reward={reward:.4f}")
open(out, "w").write(f"{reward:.4f}\n")
