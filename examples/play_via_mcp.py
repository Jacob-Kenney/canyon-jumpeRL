"""Drive Canyon Racer through the MCP game interface with any policy trained on the gym env.

    python examples/play_via_mcp.py [--rom PATH]

The server only provides the game: observation (same Box(44) vector as gym.make("CanyonRacer-v0")), `terminated`
(= game over) and game facts in `info`.  Reward shaping, truncation and step budgets live HERE, on the
client/training side, computed from obs/info.
"""
import json
import sys

import anyio
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from canyon_jumperl.env import BUTTON_ORDER, OBS_LAYOUT
from canyon_jumperl.mcp_server import decode

MAX_STEPS = 300  # client-side truncation (training-file concern)


def policy(obs) -> list[float]:
    """Any callable obs -> 8 button floats in BUTTON_ORDER. Swap in your trained network."""
    ship_x, ship_y, ship_z, _ = obs[OBS_LAYOUT["ship"]]
    ahead = obs[OBS_LAYOUT["obstacles"]].reshape(5, 5)[0]  # [present, type, hit_x_left, hit_x_right, y]
    action = [0.0] * 8
    if ahead[0] and ahead[4] > ship_y - 40 and ahead[2] - 8 <= ship_x <= ahead[3]:
        action[BUTTON_ORDER.index("a")] = 1.0  # jump
    return action


def reward_fn(prev, cur) -> float:
    """Client-side reward from game facts (example shaping: +1 per obstacle, -1 per life lost)."""
    return (cur["score"] - prev["score"]) / 10.0 - 1.0 * max(0, prev["lives"] - cur["lives"])


def _data(res):
    assert not res.is_error, res.content
    return res.structured_content if res.structured_content is not None else json.loads(res.content[0].text)


async def main():
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "canyon_jumperl.mcp_server", *sys.argv[1:]])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            d = _data(await s.call_tool("reset", {"seed": 0}))
            obs, info, total = decode(d["observation"]), decode(d["info"]), 0.0
            for step in range(MAX_STEPS):
                d = _data(await s.call_tool("step", {"action": policy(obs)}))
                obs, new_info = decode(d["observation"]), decode(d["info"])
                total += reward_fn(info, new_info)
                info = new_info
                if d["terminated"]:  # game over, reported by the game
                    break
            truncated = (not d["terminated"]) and step + 1 >= MAX_STEPS
            print(f"return={total} steps={step + 1} game_over={d['terminated']} truncated={truncated} info={info}")


if __name__ == "__main__":
    anyio.run(main)
