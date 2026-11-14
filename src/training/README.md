# src/training

`ppo.py`: PPO from RLlib on `CanyonRacerEnv`, using the env's default Box(44) observation and Box(8) action spaces.

```bash
uv run python src/training/ppo.py --iters 100 --env-runners 2 --out runs/ppo
```

Prints the mean episode return each iteration and saves an RLlib checkpoint to `--out` (absolute path recommended).
The policy can press any button, including B (costs a life) and START (pauses), and has to learn to avoid them.
