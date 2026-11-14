"""Export a trained RLlib PPO checkpoint to a small numpy file the Harbor solution can run (no ray/torch needed).

    uv run python src/training/export_policy.py runs/ppo harbor/canyon-racer-task/solution/policy.npz

The exported policy is deterministic: it presses button i when the Gaussian action mean for i is > 0 (RLlib maps the mean in [-1,1] to the env action (mean+1)/2).
"""
import sys

import numpy as np
from ray.rllib.core.rl_module.rl_module import RLModule


def numpy_forward(w, obs):
    h = np.tanh(w["w0"] @ obs + w["b0"])
    h = np.tanh(w["w1"] @ h + w["b1"])
    out = w["wp"] @ h + w["bp"]
    return out[:len(out) // 2]  # the head outputs the action means, then the log-stds


if __name__ == "__main__":
    ckpt, out = sys.argv[1:3]
    m = RLModule.from_checkpoint(f"{ckpt}/learner_group/learner/rl_module/default_policy")
    s = {k: v.detach().cpu().numpy() for k, v in m.state_dict().items()}
    w = {"w0": s["encoder.actor_encoder.net.mlp.0.weight"], "b0": s["encoder.actor_encoder.net.mlp.0.bias"],
         "w1": s["encoder.actor_encoder.net.mlp.2.weight"], "b1": s["encoder.actor_encoder.net.mlp.2.bias"],
         "wp": s["pi.net.mlp.0.weight"], "bp": s["pi.net.mlp.0.bias"]}
    # sanity check against the real module
    import torch
    obs = np.random.default_rng(0).uniform(-1, 1, w["w0"].shape[1]).astype(np.float32)
    with torch.no_grad():
        ref = m.forward_inference({"obs": torch.as_tensor(obs)[None]})["action_dist_inputs"][0].numpy()[:7]
    assert np.allclose(ref, numpy_forward(w, obs), atol=1e-3), (ref, numpy_forward(w, obs))
    np.savez(out, **w)
    print("saved", out)
