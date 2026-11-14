#!/bin/bash
# Oracle: play the game through the same MCP server. Default: heuristic policy. POLICY=ppo: trained PPO policy (solution/policy.npz).
cd /solution
if [ "$POLICY" = "ppo" ]; then python3 play_trained.py; else python3 play.py; fi
