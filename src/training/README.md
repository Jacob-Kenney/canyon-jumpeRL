# src/training

`ppo.py`: RLlib PPO on `CanyonRacerEnv` (Box(44) observation, Box(8) action), default hyperparameters.

```bash
uv run python src/training/ppo.py --iters 100 --env-runners 2 --out runs/ppo
```

Each iteration prints the game score (the game's own score, +10 per obstacle, mean over the last 100 episodes) and the training return
(the env reward, which includes crash and life penalties). Saves the RLlib checkpoint to `--out` and a graph of both curves per
iteration to `<out>.png` (here `runs/ppo.png`).
