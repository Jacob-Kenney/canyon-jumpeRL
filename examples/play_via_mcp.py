"""Drive Canyon Racer through the MCP game interface with any policy trained on the gym env.

    python examples/play_via_mcp.py [--rom PATH]

The server only provides the game: observation (same Box(33) feature vector as gym.make("CanyonRacer-v0")), `terminated`
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
    ahead = obs[OBS_LAYOUT["obstacles"]].reshape(2, 9)[0]  # present, one-hot type x5, dx_left, dx_right, dy (all relative to the ship, /100)
    dx_left, dx_right, dy = ahead[6] * 100, ahead[7] * 100, ahead[8] * 100
    action = [0.0] * 8
    if ahead[0] and ahead[5] == 0 and 0 < dy < 40 and dx_left - 8 <= 0 <= dx_right:  # ahead[5] = +1 item
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
