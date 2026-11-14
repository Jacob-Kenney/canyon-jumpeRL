# Canyon Racer

You are playing **Canyon Racer** (Game Boy, by bbbbbr, CC-BY-NC-SA 4.0) through the MCP server `canyon-racer`
(streamable HTTP, `http://mcp-server:8000/mcp`; already registered if your harness supports task MCP servers).
Your ship flies down a canyon; obstacles scroll toward it. Each obstacle you pass without crashing gives +10 score.
Crashing costs a life (the game rewinds to the last save point, score included); at 0 lives the game is over.

Tools: `reset(seed)` (seed is fixed server-side) and `step(action)`. `action` is a list of 7 floats in [0,1]
`[RIGHT, LEFT, UP, DOWN, A, B, SELECT]` (there is no START button); a button is pressed when its value is > 0.5 (jump = A, so
`[0,0,0,0,1,0,0]`). Each step holds the buttons for 4 frames and returns `{observation, terminated, info}`.
`observation` is a flat float32 feature vector of length 33 (positions are scaled by 1/100, not clipped):
`ship` [0:4] = x/100, y/100, z/100, vz/3 (the ship is jumping when z > 0); `crashing` [4] = 1 while the ship is crashing
(buttons do nothing until the game rewinds), else 0; `canyon_left_rel` [5:15] = (left canyon wall x at y = 0,16,..,144
minus ship x)/100; `obstacles` [15:33] = the 2 nearest obstacles, 9 values each: present, one-hot type
[left, right, middle, full width, +1 life item], (hit_x_left - ship_x)/100, (hit_x_right - ship_x)/100,
(ship_y - obstacle_y)/100 (positive while the obstacle is still ahead of the ship); all 0 if absent. Score and lives are in
`info`. Arrays are encoded as `{dtype, shape, data}`. Hint: jumping (A) clears obstacles; timing matters. Writing a script that
uses the `mcp` Python package (installed) to loop over `step` is far cheaper than calling tools by hand.

**Budget:** at most 1500 `step` calls in total (the server refuses more). Call `reset` once at the start; extra resets are
allowed but only your best episode counts.

**Scoring:** the server records every action; the grader replays them in a fresh emulator and computes
`reward = min(1, best_score/10 / 40)`, i.e. obstacles cleared in one run, 40 or more = full marks. Nothing else counts;
there is nothing to write to disk.
