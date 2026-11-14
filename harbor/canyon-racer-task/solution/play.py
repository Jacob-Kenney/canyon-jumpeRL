"""Reference policy: jump when a non-item obstacle is about to reach the ship. Plays via MCP (streamable HTTP)."""
import asyncio, json, os
import numpy as np
from mcp.client.session import ClientSession
try:
    from mcp.client.streamable_http import streamable_http_client as http_client
except ImportError:
    from mcp.client.streamable_http import streamablehttp_client as http_client

URL = os.environ.get("MCP_URL", "http://mcp-server:8000/mcp")
TH, HOLD = int(os.environ.get("TH", 30)), int(os.environ.get("HOLD", 2))
STEPS = int(os.environ.get("STEPS", 1500))
# observation vector layout (shape [33], positions /100): ship [0:4]=x,y,z,vz; crashing [4]; canyon_left_rel [5:15];
# obstacles [15:33] = 2 rows x [present, one-hot type x5 (4 = +1 item), dx_left, dx_right, dy = (ship_y - obstacle_y)]
Z, CRASHING, OBST = 2, 4, slice(15, 33)
NOOP = [0.0] * 8
JUMP = [0, 0, 0, 0, 1, 0, 0, 0]  # A: [RIGHT,LEFT,UP,DOWN,A,B,SELECT,START]


def data(res):
    assert not res.is_error, res.content
    return res.structured_content if res.structured_content is not None else json.loads(res.content[0].text)


def dec(x):
    return np.array(x["data"], dtype=x["dtype"]).reshape(x["shape"])


class Policy:
    def __init__(self): self.hold = 0
    def __call__(self, obs):
        if self.hold > 0:
            self.hold -= 1; return JUMP
        if obs[Z] < 1e-6 and obs[CRASHING] == 0:  # on the ground, not crashing
            for o in obs[OBST].reshape(2, 9):
                dy = float(o[8]) * 100  # ship_y - obstacle_y in pixels; same window as before: TH <= dy <= TH + 8
                if o[0] and o[5] == 0 and TH - 1e-3 <= dy <= TH + 8 + 1e-3:
                    self.hold = HOLD - 1; return JUMP
        return NOOP


async def main():
    async with http_client(URL) as streams:
        async with ClientSession(streams[0], streams[1]) as s:
            await s.initialize()
            d = data(await s.call_tool("reset", {"seed": 0}))
            pol, info = Policy(), d["info"]
            for _ in range(STEPS):
                obs = dec(d["observation"])
                d = data(await s.call_tool("step", {"action": [float(v) for v in pol(obs)]}))
                info = d["info"]
                if d["terminated"]: break
            print("final info", info)

if __name__ == "__main__":
    asyncio.run(main())
