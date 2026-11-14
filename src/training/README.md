# src/training

`ppo.py` is a self-contained PPO (clipped, GAE) in PyTorch for `CanyonRacer-v0`. It does not use RLlib.

```bash
uv run python src/training/ppo.py --total-steps 2000000 --num-envs 4 --out runs/ppo
uv run python src/training/ppo.py --help       # all hyperparameters
```

- **Actions:** the policy sets 5 independent button bits (RIGHT, LEFT, UP, DOWN, A). B, SELECT and START are always released.
- **Observation and reward:** the observation is the Box(44) vector with a running mean/std. The reward is the env's own, scaled by the running std of the return.
- **Truncation:** episodes cut off by `--max-steps` bootstrap from the value estimate. Game over does not.
- **Outputs** in `--out`: `log.csv` (per update), `latest.pt`, and `best.pt` (best periodic eval). `--resume CKPT` continues from a checkpoint.
- **Loading a policy:** `from ppo import load_policy, evaluate` (with `src/training` on the path) returns `(agent, normalizer)`.
- **Speed:** about 200 steps/s on 2 CPU cores, so 2M steps takes roughly 3 hours. Use more `--num-envs` on a bigger machine.
