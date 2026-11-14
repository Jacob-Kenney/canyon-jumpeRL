"""Trace-logging wrapper around canyon-racer-mcp (sidecar; the agent cannot touch this container).
Every reset/step is appended to /trace/trace.jsonl BEFORE the game advances. The verifier replays that trace in
its own fresh emulator. Hard step budget; seed is fixed server-side."""
import json, os
from canyon_jumperl.mcp_server import EmulatorBackend, build_server

SEED = 0
MAX_STEPS = int(os.environ.get("MAX_STEPS", "1500"))
LOG = "/trace/trace.jsonl"
open(LOG, "w").close()


class TracedBackend(EmulatorBackend):
    n = 0

    def _log(self, rec):
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())

    def reset(self, seed=None):  # agent-chosen seed ignored
        self._log({"op": "reset", "seed": SEED})
        return super().reset(SEED)

    def _step(self, buttons):
        if self.n >= MAX_STEPS:
            raise ValueError(f"step budget of {MAX_STEPS} steps exhausted")
        self._ensure()
        self.n += 1
        self._log({"op": "step", "buttons": sorted(buttons)})
        return super()._step(buttons)


if __name__ == "__main__":
    build_server(TracedBackend(os.environ["CANYON_RACER_ROM"])).run("streamable-http", host="0.0.0.0", port=8000)
