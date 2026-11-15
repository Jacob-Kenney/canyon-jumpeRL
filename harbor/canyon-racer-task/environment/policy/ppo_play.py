"""Play Canyon Racer through the MCP server with the pre-trained PPO policy (policy.npz next to this file).

    python /opt/policy/ppo_play.py            # reset(seed 0) and play up to STEPS (default 1500) steps

Deterministic: presses button i when the network's action mean for i is > 0. Needs only numpy + the `mcp` package.
"""
import asyncio, json, os
import numpy as np
from mcp.client.session import ClientSession
try:
    from mcp.client.streamable_http import streamable_http_client as http_client
except ImportError:
    from mcp.client.streamable_http import streamablehttp_client as http_client

URL = os.environ.get("MCP_URL", "http://mcp-server:8000/mcp")
STEPS = int(os.environ.get("STEPS", 1500))
W = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.npz"))


def data(res):
    assert not res.is_error, res.content
    return res.structured_content if res.structured_content is not None else json.loads(res.content[0].text)


def dec(x):
    return np.array(x["data"], dtype=x["dtype"]).reshape(x["shape"])


def policy(obs):
    h = np.tanh(W["w0"] @ obs + W["b0"])
    h = np.tanh(W["w1"] @ h + W["b1"])
    out = W["wp"] @ h + W["bp"]
    mean = out[:len(out) // 2]  # RLlib action means in [-1,1]; env action = (mean+1)/2, so pressed when mean > 0
    return [1.0 if m > 0 else 0.0 for m in mean]


async def main():
    async with http_client(URL) as streams:
        async with ClientSession(streams[0], streams[1]) as s:
            await s.initialize()
            d = data(await s.call_tool("reset", {"seed": 0}))
            info = d["info"]
            for _ in range(STEPS):
                d = data(await s.call_tool("step", {"action": policy(dec(d["observation"]).astype(np.float64))}))
                info = d["info"]
                if d["terminated"]: break
            print("final info", info)

if __name__ == "__main__":
    asyncio.run(main())
