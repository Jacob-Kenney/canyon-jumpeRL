# Canyon Racer

You are playing **Canyon Racer** (Game Boy, by bbbbbr, CC-BY-NC-SA 4.0) through the MCP server `canyon-racer`
(streamable HTTP, `http://mcp-server:8000/mcp`; already registered if your harness supports task MCP servers).
Your ship flies down a canyon; obstacles scroll toward it. Each obstacle you pass without crashing gives +10 score.
Crashing costs a life (the game rewinds to the last save point, score included); at 0 lives the game is over.

Tools: `reset(seed)` (seed is fixed server-side) and `step(action)`. `action` is a list of 8 floats in [0,1]
`[RIGHT, LEFT, UP, DOWN, A, B, SELECT, START]`; a button is pressed when its value is > 0.5 (jump = A, so
`[0,0,0,0,1,0,0,0]`). Each step holds the buttons for 4 frames and returns `{observation, terminated, info}`.
`observation` is a flat float32 vector of length 44: `ship` [0:4] = x, y, z, vz; `ship_state` [4] (0 ground, 1 jump,
2 crashing, 3 crash ended); `canyon_left` [5:15]; `lives` [15]; `score` [16]; `level` [17]; `speed` [18]; `obstacles`
[19:44] = 5 rows x [present, type (0 left, 1 right, 2 middle, 3 full width, 4 +1 life item), hit_x_left,
hit_x_right, y], nearest first. Arrays are encoded as `{dtype, shape, data}`. Hint: jumping (A) clears obstacles; timing matters. Writing a script that
uses the `mcp` Python package (installed) to loop over `step` is far cheaper than calling tools by hand.

**Budget:** at most 1500 `step` calls in total (the server refuses more). Call `reset` once at the start; extra resets are
allowed but only your best episode counts.

**Scoring:** the server records every action; the grader replays them in a fresh emulator and computes
`reward = min(1, best_score/10 / 40)`, i.e. obstacles cleared in one run, 40 or more = full marks. Nothing else counts;
there is nothing to write to disk.
