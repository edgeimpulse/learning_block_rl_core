"""The files blocks write for Studio and the extensions: the shapes the readers expect."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ei_robotics.core import parse_args, parser, write_samples, write_training_metrics

HAS_RL = all(importlib.util.find_spec(m) for m in ("mujoco", "PIL"))


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


@unittest.skipUnless(HAS_RL, "needs ei-robotics[rl]")
class RLVizTest(unittest.TestCase):
    def run_viz(self, d, **kw):
        from ei_robotics.rl.viz import RLViz

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
        from ei_robotics.rl.viz import write_eval

        with tempfile.TemporaryDirectory() as d:
            self.run_viz(d)
            write_eval(d, [{"name": "a"}], 10.0, range(5))
            ev = json.loads((Path(d) / "eval.json").read_text())
            status = json.loads((Path(d) / "manifest.json").read_text())["status"]
        self.assertEqual((ev["seeds"], ev["caption"], status), (5, None, "finished"))


if __name__ == "__main__":
    unittest.main()
