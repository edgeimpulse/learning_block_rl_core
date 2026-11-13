# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""mjlab + rsl_rl: the training monitor hook, the time budget and the live checkpoints.

rsl_rl has no callback API, so RslRlViz wraps `runner.logger.log`, which `learn()` calls once per iteration
after the PPO update, and reads the episode statistics before the logger clears them. From mjlab's managers it
takes the reward weights (the rl-viz reward terms, from `Episode_Reward/<term>`) and which terminations are
time-outs (the others count as failures in fall_rate).

    viz = RLViz(custom_data, total_timesteps=iterations * num_steps_per_env * num_envs, ...,
                checkpoint=checkpoint(runner, custom_data))
    hook = RslRlViz(viz, rollout=my_rollout)   # rollout(policy, render) -> (frames, stats)
    hook.attach(runner, env)
    checkpoint_file = hook.learn(iterations, init_at_random_ep_len=True)
    hook.finish()

The rollout is the block's: what a snapshot plays and measures depends on the task. It gets the current
deterministic policy (the runner in eval mode) and plays it in its own env.

Install with the `[mjlab]` extra; the robot package pins the exact mjlab version, which brings rsl_rl and
mujoco. The hook itself only touches the runner, its logger and mjlab's managers, so it imports none of them.
It reads rsl_rl's logger internals (`log()`'s keyword arguments, `ep_extras`, `rewbuffer`, `lenbuffer`) and
the managers' term lists, which are not a public API. Tested with mjlab 1.3.0 and rsl-rl-lib 5.0.1.
"""

import contextlib
import io
import os
import statistics
import time
from pathlib import Path

import numpy as np


class StopTraining(Exception):
    """Raised from the logger hook when the wall-clock budget is spent; RslRlViz.learn catches it."""


class RslRlViz:
    """Feeds an RLViz from an rsl_rl runner training an mjlab task: one viz.log() per PPO iteration.

    viz: the RLViz, with total_timesteps = iterations x num_steps_per_env x num_envs.
    rollout: rollout(policy, render) -> (frames, stats), for the snapshots (None: no snapshots).
    print_every: rsl_rl's own console block every this many iterations (every iteration would drown the
        job log).
    """

    def __init__(self, viz, rollout=None, print_every=50):
        self.viz = viz
        self.rollout = rollout
        self.print_every = print_every
        self.iterations = 0
        self.stopped_early = False

    @property
    def step(self):
        return self.iterations * self.steps_per_iter

    def attach(self, runner, train_env, header=None):
        """Start the run (header, first snapshot) and hook the runner's logger.

        train_env: the RslRlVecEnvWrapper the runner trains on. header: more run header fields (n_envs is added).
        """
        missing = [a for a in ("log", "ep_extras", "rewbuffer", "lenbuffer", "log_dir")
                   if not hasattr(runner.logger, a)]
        if missing:
            raise RuntimeError(f"ei_rl.mjlab: this rsl_rl's logger has no {', '.join(missing)}; the hook is tested "
                               "with rsl-rl-lib 5.0.1")
        self.runner = runner
        u = train_env.unwrapped
        self.steps_per_iter = runner.cfg["num_steps_per_env"] * u.num_envs
        rm = u.reward_manager
        weights = {k: float(c.weight) for k, c in zip(rm._term_names, rm._term_cfgs)}
        tm = u.termination_manager
        self.timeout_terms = {k for k, c in zip(tm._term_names, tm._term_cfgs) if getattr(c, "time_out", False)}
        if self.rollout is not None:
            self.viz.rollout = self._rollout
        self.viz.start(float(u.step_dt), weights, header=dict(n_envs=u.num_envs, **(header or {})))
        self.t_iter = time.time()

        original = runner.logger.log

        def log(*args, **kwargs):
            row, terms = self._iteration(kwargs)  # before the original, which clears the episode extras
            if self.print_every and kwargs["it"] % self.print_every == 0:
                original(*args, **kwargs)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    original(*args, **kwargs)
            self.viz.log(self.step, row, terms)  # and snapshot, checkpoint if due
            if self.viz.out_of_time():
                self.stopped_early = True
                raise StopTraining()

        runner.logger.log = log

    def learn(self, num_learning_iterations, **kwargs):
        """runner.learn(), stopped at the time budget. Returns the path of the final rsl_rl checkpoint (.pt)."""
        runner = self.runner
        try:
            runner.learn(num_learning_iterations=num_learning_iterations, **kwargs)
        except StopTraining:
            print(f"\nTime budget of {self.viz.max_seconds / 60:g} min reached after {self.iterations} iterations; "
                  "stopping.")
            runner.save(os.path.join(runner.logger.log_dir, f"model_{runner.current_learning_iteration}.pt"))
            runner.logger.stop_logging_writer()
        return Path(runner.logger.log_dir) / f"model_{runner.current_learning_iteration}.pt"

    def finish(self):
        """The final snapshot and checkpoint, and the run marked trained."""
        self.viz.finish(self.step)

    def _iteration(self, kw):
        now = time.time()
        self.iterations += 1
        logger = self.runner.logger
        extras = logger.ep_extras
        terms, ended, early = {}, 0, 0
        for k in self.viz.term_names:
            vals = [float(e[f"Episode_Reward/{k}"]) for e in extras if f"Episode_Reward/{k}" in e]
            terms[k] = np.mean(vals) if vals else None
        for e in extras:
            for key, v in e.items():
                if key.startswith("Episode_Termination/"):
                    n = int(v.item() if hasattr(v, "item") else v)
                    ended += n
                    if key.split("/", 1)[1] not in self.timeout_terms:
                        early += n
        loss = kw.get("loss_dict") or {}
        std = kw.get("action_std")
        row = dict(
            ep_rew=statistics.mean(logger.rewbuffer) if len(logger.rewbuffer) else None,
            ep_len=statistics.mean(logger.lenbuffer) if len(logger.lenbuffer) else None,
            fps=int(self.steps_per_iter / max(now - self.t_iter, 1e-6)),
            # Resets this iteration that were not time-outs. Counts terminations, not whole episodes: an env
            # that fails twice in one iteration counts twice.
            fall_rate=early / ended if ended else None,
            std=std.mean().item() if std is not None else None,
            entropy=loss.get("entropy"),
            value_loss=loss.get("value", loss.get("value_function")),
            pg_loss=loss.get("surrogate"),
            lr=kw.get("learning_rate"),
        )
        self.t_iter = now
        return row, terms

    def _rollout(self, render):
        alg = self.runner.alg
        policy = alg.get_policy()
        alg.eval_mode()
        try:
            return self.rollout(policy, render)
        finally:
            alg.train_mode()


def checkpoint(runner, out_dir, live=True):
    """RLViz's checkpoint callback: the current policy exported as mjlab exports it (runner.export_policy_to_onnx,
    with mjlab's metadata) to <out_dir>/checkpoints/, and announced if `live` (see ei_rl.bundle)."""

    def save(step, iteration, label):
        from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

        from .bundle import save_checkpoint

        def export(tmp):
            runner.export_policy_to_onnx(str(tmp.parent), tmp.name)
            attach_metadata_to_onnx(str(tmp), get_base_metadata(runner.env.unwrapped,
                                                                run_path=f"iteration_{iteration}"))

        save_checkpoint(out_dir, step, iteration, label, export, live=live)

    return save


__all__ = ["RslRlViz", "StopTraining", "checkpoint"]
