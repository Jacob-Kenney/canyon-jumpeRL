"""Play via MCP with the trained PPO policy (solution/policy.npz, exported by src/training/export_policy.py)."""
import asyncio, os
import numpy as np
from mcp.client.session import ClientSession
from play import URL, STEPS, data, dec, http_client

W = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.npz"))


def policy(obs):
    h = np.tanh(W["w0"] @ obs + W["b0"])
    h = np.tanh(W["w1"] @ h + W["b1"])
    mean = (W["wp"] @ h + W["bp"])[:8]  # action means; pressed when > 0.5
    return [1.0 if m > 0.5 else 0.0 for m in mean]


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
