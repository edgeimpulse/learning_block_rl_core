# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""The files blocks write for Studio and the extensions: the shapes the readers expect."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ei_rl.core import parse_args, parser, write_samples, write_training_metrics


def quiet(f, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        r = f(*a, **kw)
    return r, out.getvalue()


class CoreTest(unittest.TestCase):
    def test_unknown_arguments_ignored(self):
        args, log = quiet(parse_args, parser("t"), ["--out-directory", "o", "--info-file", "x.json"])
        self.assertEqual(args.out_directory, "o")
        self.assertIn("--info-file", log)

    def test_training_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            quiet(write_training_metrics, d, {"a": 1.234567, "b": None, "c": 3})
            m = json.loads((Path(d) / "custom_training_metrics.json").read_text())
        self.assertEqual(m["type"], "metrics")
        self.assertEqual(m["metrics"], [{"name": "a", "value": 1.2346}, {"name": "c", "value": 3}])

    def test_samples(self):
        with tempfile.TemporaryDirectory() as d:
            files = write_samples(d, [("s0", np.zeros(3), None), ("s1", np.ones(3), "x")], ["a", "b", "c"],
                                  "dev", "SIM", 50)
            sample = json.loads((Path(d) / "s1.json").read_text())
            labels = json.loads((Path(d) / "info.labels").read_text())
        self.assertEqual(sample["payload"]["values"], [[1.0, 1.0, 1.0]])
        self.assertEqual([s["name"] for s in sample["payload"]["sensors"]], ["a", "b", "c"])
        self.assertEqual(labels["files"], files)
        self.assertEqual(files[0]["label"], {"type": "unlabeled"})
        self.assertEqual(files[1]["label"], {"type": "label", "label": "x"})


class RLVizTest(unittest.TestCase):
    def run_viz(self, d, **kw):
        from ei_rl.viz import RLViz

        checkpoints = []
        viz = RLViz(d, total_timesteps=100, checkpoint=lambda *a: checkpoints.append(a), **kw)
        viz.rollout = lambda render: ([], {"outcome": "ok", "sim_return": 1.0})
        quiet(viz.start, 0.02, {"alive": 1.0, "fall": -2.0}, success=True)
        for step in (50, 100):
            quiet(viz.log, step, {"ep_rew": 1.23456789, "fall_rate": 0.33333, "success_rate": 0.5, "fps": 1000.7},
                  {"alive": 0.98765, "fall": -0.1})
        quiet(viz.finish)
        return viz, checkpoints

    def test_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            viz, checkpoints = self.run_viz(d, task={"name": "t"}, run_info={"seed": 0})
            manifest = json.loads((Path(d) / "manifest.json").read_text())
            metrics = json.loads((Path(d) / "metrics.json").read_text())
        self.assertEqual(manifest["format"], "rl-viz")
        self.assertEqual(manifest["status"], "trained")
        self.assertEqual(manifest["task"], {"name": "t"})
        self.assertEqual(manifest["seed"], 0)
        self.assertEqual(metrics["columns"][:7], ["step", "time", "ep_rew", "ep_len", "fps", "fall_rate",
                                                  "success_rate"])
        self.assertEqual(metrics["columns"][-2:], ["term_alive", "term_fall"])
        row = dict(zip(metrics["columns"], metrics["rows"][0]))
        self.assertEqual((row["step"], row["fps"], row["ep_rew"], row["fall_rate"]), (50, 1000, 1.2346, 0.333))
        self.assertEqual(row["term_alive"], 0.9877)
        self.assertEqual([s["step"] for s in manifest["snapshots"]], [0, 50, 100])
        self.assertEqual(manifest["snapshots"][-1]["label"], "Final")
        self.assertEqual([c[0] for c in checkpoints], [50, 100])  # not at step 0, once per step

    def test_eval_finishes_the_run(self):
        from ei_rl.viz import write_eval

        with tempfile.TemporaryDirectory() as d:
            self.run_viz(d)
            write_eval(d, [{"name": "a"}], 10.0, range(5))
            ev = json.loads((Path(d) / "eval.json").read_text())
            status = json.loads((Path(d) / "manifest.json").read_text())["status"]
        self.assertEqual((ev["seeds"], ev["caption"], status), (5, None, "finished"))


class RslRlVizTest(unittest.TestCase):
    """ei_rl.mjlab against a stand-in for an rsl_rl runner and an mjlab env (the same attributes, no torch)."""

    def fake(self, iterations):
        from types import SimpleNamespace as NS

        term = lambda w, **kw: NS(weight=w, **kw)  # noqa: E731
        u = NS(num_envs=4, step_dt=0.02,
               reward_manager=NS(_term_names=["alive", "fall"], _term_cfgs=[term(1.0), term(-2.0)]),
               termination_manager=NS(_term_names=["time_out", "fell"],
                                      _term_cfgs=[NS(time_out=True), NS(time_out=False)]))
        env = NS(unwrapped=u)
        saved, printed = [], []
        alg = NS(get_policy=lambda: "policy", eval_mode=lambda: printed.append("eval"),
                 train_mode=lambda: printed.append("train"))

        def log(**kw):
            print("rsl_rl block", kw["it"])
            runner.logger.ep_extras = []

        logger = NS(ep_extras=[], rewbuffer=[], lenbuffer=[], log_dir="/run", log=log,
                    stop_logging_writer=lambda: saved.append("stopped"))

        def learn(num_learning_iterations, init_at_random_ep_len=False):
            for it in range(num_learning_iterations):
                runner.current_learning_iteration = it
                logger.ep_extras = [{"Episode_Reward/alive": 0.5, "Episode_Reward/fall": -0.25,
                                     "Episode_Termination/time_out": 3, "Episode_Termination/fell": 1}]
                logger.rewbuffer, logger.lenbuffer = [1.0, 2.0], [10, 20]
                runner.logger.log(it=it, loss_dict={"value": 0.1, "surrogate": -0.2, "entropy": 1.5},
                                  learning_rate=1e-3)
            saved.append(f"model_{runner.current_learning_iteration}.pt")

        runner = NS(cfg={"num_steps_per_env": 5}, logger=logger, alg=alg, learn=learn, current_learning_iteration=0,
                    save=lambda path: saved.append(path))
        return runner, env, saved, printed

    def run_hook(self, d, iterations, **viz_kw):
        from ei_rl.mjlab import RslRlViz
        from ei_rl.viz import RLViz

        runner, env, saved, calls = self.fake(iterations)
        viz = RLViz(d, total_timesteps=iterations * 20, checkpoint=lambda *a: None, **viz_kw)
        hook = RslRlViz(viz, rollout=lambda policy, render: (calls.append(policy) or [], {"outcome": "ok"}),
                        print_every=2)
        quiet(hook.attach, runner, env)
        model_pt, log = quiet(hook.learn, iterations)
        quiet(hook.finish)
        return hook, model_pt, log, saved, calls

    def test_iterations(self):
        with tempfile.TemporaryDirectory() as d:
            hook, model_pt, log, saved, calls = self.run_hook(d, 3)
            metrics = json.loads((Path(d) / "metrics.json").read_text())
        self.assertEqual((hook.iterations, hook.steps_per_iter, hook.stopped_early), (3, 20, False))
        self.assertEqual(str(model_pt), "/run/model_2.pt")
        self.assertIn("rsl_rl block 0", log)
        self.assertNotIn("rsl_rl block 1", log)  # print_every=2
        row = dict(zip(metrics["columns"], metrics["rows"][0]))
        self.assertEqual((row["step"], row["ep_rew"], row["ep_len"], row["fall_rate"]), (20, 1.5, 15, 0.25))
        self.assertEqual((row["value_loss"], row["pg_loss"], row["entropy"], row["lr"]), (0.1, -0.2, 1.5, 0.001))
        self.assertEqual((row["term_alive"], row["term_fall"]), (0.5, -0.25))
        self.assertEqual(calls[:3], ["eval", "policy", "train"])  # snapshots play the policy in eval mode

    def test_time_budget(self):
        with tempfile.TemporaryDirectory() as d:
            hook, model_pt, log, saved, _ = self.run_hook(d, 50, max_minutes=1e-9)
        self.assertEqual((hook.iterations, hook.stopped_early), (1, True))
        self.assertIn("Time budget", log)
        self.assertEqual(saved, ["/run/model_0.pt", "stopped"])
        self.assertEqual(str(model_pt), "/run/model_0.pt")


if __name__ == "__main__":
    unittest.main()
