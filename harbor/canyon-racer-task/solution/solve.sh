#!/bin/bash
# Oracle: play the game through the same MCP server. Default: heuristic policy. POLICY=ppo: the trained PPO policy shipped in the agent image.
cd /solution
if [ "$POLICY" = "ppo" ]; then python3 /opt/policy/ppo_play.py; else python3 play.py; fi
