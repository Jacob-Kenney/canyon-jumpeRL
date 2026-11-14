# src/training

`ppo.py`: PPO from stable-baselines3 on `CanyonRacer-v0` with its default Box(44) observation and Box(8) action spaces.

```bash
uv run python src/training/ppo.py --steps 2000000 --envs 4 --out runs/ppo
```

Saves `model.zip` and `vecnormalize.pkl` (observation/reward statistics, needed to run the policy) in `--out`.
The policy can press any button, including B (costs a life) and START (pauses), and has to learn to avoid them.
