"""MCP <-> gym parity: the MCP game interface must match CanyonRacer-v0 (defaults) bit-for-bit."""
import json
import sys

import anyio
import gymnasium as gym
import numpy as np
import pytest
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

import canyon_jumperl  # noqa: F401
from canyon_jumperl.env import OBS_DIM
from canyon_jumperl.mcp_server import decode

STEPS = 200
FACTS = ("lives", "score", "level", "ship_state", "paused", "game_state", "frames")
PARAMS = StdioServerParameters(command=sys.executable, args=["-m", "canyon_jumperl.mcp_server"])


def _data(res):
    assert not res.is_error, res.content
    return res.structured_content if res.structured_content is not None else json.loads(res.content[0].text)


def _plan(env, seed, n):
    env.action_space.seed(seed + 1000)
    return [env.action_space.sample() for _ in range(n)]


def _wire(a):
    return a.tolist() if isinstance(a, np.ndarray) else int(a)


def _eq_obs(a, b):
    x, y = np.asarray(a), np.asarray(b)
    assert x.dtype == y.dtype == np.float32 and x.shape == y.shape == (OBS_DIM,)
    np.testing.assert_array_equal(x, y)  # bit-for-bit


def _facts(info):
    f = {k: info[k] for k in FACTS}
    f["game_over"] = info["game_state"] != 4
    return f


def _direct(env_kwargs, seed, actions):
    env = gym.make("CanyonRacer-v0", **env_kwargs)
    obs, info = env.reset(seed=seed)
    out = [(obs, False, _facts(info))]
    for a in actions:
        obs, _r, te, tr, info = env.step(a)
        out.append((obs, te, _facts(info)))
        if te or tr:
            break
    env.close()
    return out


async def _via_mcp(seeds, plans):
    results = {}
    async with stdio_client(PARAMS) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = {t.name for t in (await s.list_tools()).tools}
            assert tools == {"reset", "step"}
            for seed, actions in zip(seeds, plans):
                d = _data(await s.call_tool("reset", {"seed": seed}))
                assert "reward" not in d and "truncated" not in d
                out = [(decode(d["observation"]), d["terminated"], decode(d["info"]))]
                for a in actions:
                    d = _data(await s.call_tool("step", {"action": _wire(a)}))
                    assert set(d) == {"observation", "terminated", "info"}
                    out.append((decode(d["observation"]), d["terminated"], decode(d["info"])))
                    if d["terminated"]:
                        break
                results[seed] = out
    return results


@pytest.mark.parametrize("env_kwargs,seeds", [
    ({}, [0]),                                         # discrete index encoding, defaults
])
def test_parity_gym_vs_mcp(env_kwargs, seeds):
    probe = gym.make("CanyonRacer-v0", **env_kwargs)
    plans = [_plan(probe, s, STEPS) for s in seeds]
    probe.close()
    direct = {s: _direct(env_kwargs, s, p) for s, p in zip(seeds, plans)}
    via = anyio.run(_via_mcp, seeds, plans)
    for s in seeds:
        d, m = direct[s], via[s]
        assert len(d) == len(m), f"episode length differs for seed {s}"
        for (o1, t1, f1), (o2, t2, f2) in zip(d, m):
            _eq_obs(o1, o2)
            assert t1 == t2
            assert f1 == {k: f2[k] for k in f1}, (f1, f2)
            assert f2["game_over"] == t2


def test_tool_description_and_errors():
    async def run():
        async with stdio_client(PARAMS) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = {t.name: t for t in (await s.list_tools()).tools}
                assert "RIGHT,LEFT,UP,DOWN,A,B,SELECT,START" in tools["step"].description
                await s.call_tool("reset", {"seed": 1})
                assert (await s.call_tool("step", {"action": [2.0] * 8})).is_error
                assert (await s.call_tool("step", {"action": [1, 0]})).is_error
    anyio.run(run)
