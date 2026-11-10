"""Gymnasium + Stable-Baselines3 PPO: the training monitor callback and the ONNX export.

What the env provides (through the VecEnv's get_attr and the step infos):
  reward_scales   reward term -> weight; each step's info has every term's unweighted value per second
  control_dt      seconds per policy step
  info["is_failure"], info["is_success"]  at the end of an episode (or pass `failure` / `success`)

The exported model.onnx is the deterministic actor with the VecNormalize statistics baked in: input "input"
[1, obs_dim] float32, output "output" [1, act_dim] in [-1, 1]. With a privileged critic (observations followed
by state only the critic sees), obs_dim is the actor's part only.
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

from .viz import RLViz

# rl-viz TRAIN_COLUMNS -> SB3 logger keys
TRAIN_KEYS = {
    "std": "train/std",
    "entropy": "train/entropy_loss",
    "value_loss": "train/value_loss",
    "pg_loss": "train/policy_gradient_loss",
    "approx_kl": "train/approx_kl",
    "clip_frac": "train/clip_fraction",
    "explained_var": "train/explained_variance",
    "lr": "train/learning_rate",
}


def is_failure(info):
    """The env marked the episode a failure (the default)."""
    return bool(info.get("is_failure", False))


def is_success(info):
    return bool(info.get("is_success", False))


def ended_early(info):
    """Any end but the time limit is a failure (a fall, the object lost)."""
    return not info.get("TimeLimit.truncated", False)


class RLVizCallback(BaseCallback):
    """Feeds an RLViz from SB3: one row per rollout, the snapshots, the checkpoints.

    rollout(policy, render) -> (frames, stats): the block's snapshot, playing `policy(obs) -> action` (the
        current deterministic policy, on raw observations).
    failure(info), success(info): how an episode ended, from its last info. success=None: the task doesn't
        count successes (no success_rate column).
    header: more run header fields (e.g. the snapshot's fixed command).
    """

    def __init__(self, viz: RLViz, rollout, failure=is_failure, success=is_success, header=None):
        super().__init__()
        self.viz = viz
        self.failure, self.success = failure, success
        self.header = header
        viz.rollout = lambda render: rollout(self.deterministic_policy(), render)

    def _on_training_start(self):
        self.steps_iter = self.num_timesteps
        self.weights = self.training_env.get_attr("reward_scales", indices=0)[0]
        self.term_names = list(self.weights)
        control_dt = self.training_env.get_attr("control_dt", indices=0)[0]
        self._reset_counts()
        self.viz.start(control_dt, self.weights, success=self.success is not None, header=self.header,
                       step=self.num_timesteps)
        self.t_iter = self.viz.t0

    def _reset_counts(self):
        self.term_sums = np.zeros(len(self.term_names))
        self.term_count = 0
        self.episodes = self.failures = self.successes = 0

    def _on_step(self):
        infos, dones = self.locals["infos"], self.locals["dones"]
        for info, done in zip(infos, dones):
            for j, k in enumerate(self.term_names):
                self.term_sums[j] += info.get(k, 0.0)
            if done:
                self.episodes += 1
                self.failures += int(self.failure(info))
                if self.success is not None:
                    self.successes += int(self.success(info))
        self.term_count += len(infos)
        return True

    def _on_rollout_end(self):
        now = time.time()
        buf = self.model.ep_info_buffer
        logged = self.logger.name_to_value  # train/* from the previous update
        row = dict(
            ep_rew=float(np.mean([e["r"] for e in buf])) if buf else None,
            ep_len=float(np.mean([e["l"] for e in buf])) if buf else None,
            fps=int((self.num_timesteps - self.steps_iter) / max(now - self.t_iter, 1e-6)),
            fall_rate=self.failures / self.episodes if self.episodes else None,
            success_rate=self.successes / self.episodes if self.episodes and self.success is not None else None,
            **{k: logged.get(v) for k, v in TRAIN_KEYS.items()},
        )
        # Weighted contribution of each reward term, in reward per second of simulated time
        means = self.term_sums / max(self.term_count, 1)
        terms = {k: self.weights[k] * means[j] for j, k in enumerate(self.term_names)}
        self.t_iter, self.steps_iter = now, self.num_timesteps
        self._reset_counts()
        self.viz.log(self.num_timesteps, row, terms)

    def _on_training_end(self):
        self.viz.finish(self.num_timesteps)

    def deterministic_policy(self):
        """policy(obs) -> action on raw observations, with the current VecNormalize statistics (not updated).
        Observations shorter than the training's (the actor's part only) are padded with zeros."""
        vec_norm = self.training_env
        n = vec_norm.observation_space.shape[0]

        def policy(o):
            o = np.asarray(o)
            if len(o) < n:
                o = np.pad(o, (0, n - len(o)))
            return self.model.predict(vec_norm.normalize_obs(o), deterministic=True)[0]
        return policy


# ----------------------------------------------------------------------------------------------------------
# ONNX
# ----------------------------------------------------------------------------------------------------------

def actor_obs_dim(model, vec_norm):
    """The actor's inputs: the policy's mlp_extractor.actor_obs_dim if it has one, else every observation."""
    return int(getattr(model.policy.mlp_extractor, "actor_obs_dim", vec_norm.observation_space.shape[0]))


class ExportPolicy(nn.Module):
    """Normalize the actor's observations with frozen VecNormalize stats -> actor MLP -> action in [-1, 1].

    If the MLP's first layer takes more inputs than obs_dim (an actor trained on the full observation, with the
    critic's extra state after the actor's), it is cut to the first obs_dim inputs.
    """

    def __init__(self, mlp, action_net, obs_rms, clip_obs, epsilon, obs_dim):
        super().__init__()
        scale = 1.0 / np.sqrt(obs_rms.var[:obs_dim] + epsilon)
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("bias", torch.tensor(-obs_rms.mean[:obs_dim] * scale, dtype=torch.float32))
        self.clip_obs = float(clip_obs)
        self.mlp = mlp
        first = mlp[0] if isinstance(mlp, nn.Sequential) and len(mlp) else None
        if isinstance(first, nn.Linear) and first.in_features > obs_dim:
            sliced = nn.Linear(obs_dim, first.out_features)
            with torch.no_grad():
                sliced.weight.copy_(first.weight[:, :obs_dim])
                sliced.bias.copy_(first.bias)
            self.mlp[0] = sliced
        self.action_net = action_net

    def forward(self, obs):
        x = torch.clamp(obs * self.scale + self.bias, -self.clip_obs, self.clip_obs)
        return torch.clamp(self.action_net(self.mlp(x)), -1.0, 1.0)


def export_onnx(model, vec_norm, path, obs_dim=None):
    """Write the deterministic actor to `path` (opset 10, input "input", output "output"). Returns obs_dim.

    Exports copies of the actor's layers: a checkpoint exports while training goes on (eval() would switch the
    live policy's mode), and the policy also holds the last training batch's action distribution.
    """
    obs_dim = obs_dim or actor_obs_dim(model, vec_norm)
    mlp, head = (copy.deepcopy(m).cpu() for m in (model.policy.mlp_extractor.policy_net, model.policy.action_net))
    export = ExportPolicy(mlp, head, vec_norm.obs_rms, vec_norm.clip_obs, vec_norm.epsilon, obs_dim).eval()
    torch.onnx.export(export, torch.zeros(1, obs_dim), str(path), export_params=True, opset_version=10,
                      do_constant_folding=True, dynamo=False, external_data=False,
                      input_names=["input"], output_names=["output"])
    return obs_dim


def verify_onnx(model, vec_norm, path, env, obs_dim=None, n=256, seed=123, tolerance=1e-4):
    """Check the ONNX model matches SB3's deterministic predict on `n` observations of `env` (a Gymnasium env
    making the training's observations), played by the trained policy."""
    from ..core.onnx import onnx_policy

    obs_dim = obs_dim or actor_obs_dim(model, vec_norm)
    obs, _ = env.reset(seed=seed)
    batch = []
    for _ in range(n):
        batch.append(obs)
        action, _ = model.predict(vec_norm.normalize_obs(obs), deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
    batch = np.asarray(batch, dtype=np.float32)
    policy = onnx_policy(path)
    onnx_out = np.stack([policy(o[:obs_dim]) for o in batch])
    sb3_out, _ = model.predict(vec_norm.normalize_obs(batch), deterministic=True)
    err = float(np.max(np.abs(onnx_out - np.clip(sb3_out, -1, 1))))
    print(f"ONNX vs SB3 max abs diff over {n} observations: {err:.2e}")
    if err > tolerance:
        raise RuntimeError("exported ONNX model does not match the trained policy")
    return err
