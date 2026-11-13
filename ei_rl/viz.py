# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""The rl-viz format: training telemetry for the RL training monitor Studio extension.

The format belongs to the extension (rl_training_monitor_extension). This module writes it; a new field or file
is agreed with the extension's owner first.

Two outputs, so the extension can show both a live view and a full report:

  * stdout, which Studio streams while the job runs:
      [rl-viz-run] {...}       the run header, once at the start
      [rl-viz] {...}           one row per training iteration, with each reward term's contribution
      [rl-viz-snapshot] {...}  one per snapshot, once its GIF is written
  * the custom data directory (/home/custom_data in Studio), which the extension reads through the API, even
    while the job runs:
      manifest.json            the run header, status, snapshot list (read first)
      metrics.json             every iteration, columnar: {"columns": [...], "rows": [[...], ...]}
      eval.json                the benchmark of the exported policy (write_eval)
      snapshots/step_<n>.gif   the policy at training milestones

RLViz is the trainer-agnostic part: a trainer integration (ei_rl.sb3, or a block's own) calls
start() once, log() once per iteration with the step count, and finish() at the end, and gives it `rollout`,
which plays the current policy for a snapshot. RLViz decides when to snapshot and checkpoint.

The run header's `task` tells the extension how to label this task: what success and failure mean, which
snapshot stats and evaluation columns to show. Blocks build it; see the extension's README.
"""

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .core.block import resolve_custom_data_dir
from .core.files import write_json

FORMAT = "rl-viz"
FORMAT_VERSION = 1
LOG_PREFIX = "[rl-viz]"
SNAPSHOT_PREFIX = "[rl-viz-snapshot]"
RUN_PREFIX = "[rl-viz-run]"

# Fraction of the run (by timesteps or wall-clock, whichever is further) at which to snapshot. Front-loaded
# because most of the visible learning happens early.
SNAPSHOT_AT = (0.0, 0.01, 0.03, 0.06, 0.1, 0.2, 0.35, 0.5, 0.75)
# Checkpoints come with the snapshots, at most this far apart, and at the end
CHECKPOINT_EVERY_S = 15 * 60

# metrics.json columns: these, then success_rate if the task counts successes, then TRAIN_COLUMNS, then one
# term_<name> per reward term
BASE_COLUMNS = ("step", "time", "ep_rew", "ep_len", "fps", "fall_rate")
TRAIN_COLUMNS = ("std", "entropy", "value_loss", "pg_loss", "approx_kl", "clip_frac", "explained_var", "lr")
# Significant digits per column (default 5); step and fps are integers
DIGITS = {"time": 6, "fall_rate": 3, "success_rate": 3}
TERM_DIGITS = 4


def round_sig(x, digits=5):
    """`digits` significant digits; None for None, NaN and infinities (JSON has no NaN)."""
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return float(f"{x:.{digits}g}")


def columns(term_names, success=False):
    return (list(BASE_COLUMNS) + (["success_rate"] if success else []) + list(TRAIN_COLUMNS)
            + [f"term_{k}" for k in term_names])


def print_line(prefix, obj):
    print(f"{prefix} " + json.dumps(obj, separators=(",", ":")), flush=True)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save_gif(frames, path, fps, colors=64):
    """One palette for the whole clip and no dithering: per-frame palettes or dither noise defeat GIF's
    frame-difference compression. Written atomically: the extension may read it while training runs."""
    from PIL import Image

    path = Path(path)
    palette = Image.fromarray(np.concatenate(frames[::6], axis=0)).quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT)
    images = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    tmp = path.with_suffix(path.suffix + ".tmp")
    images[0].save(tmp, format="GIF", save_all=True, append_images=images[1:], duration=int(round(1000 / fps)),
                   loop=0, optimize=True, disposal=1)
    os.replace(tmp, path)


class RLViz:
    """Writes the rl-viz outputs for one training run, and schedules its snapshots and checkpoints.

    custom_data_dir: where the files go (None: the log lines only).
    total_timesteps, max_minutes: the run's budget, for the progress at which to snapshot.
    task: the run header's task descriptor (None: none, the extension falls back to its defaults).
    run_info: more run header fields (the training settings).
    snapshots: play and record the policy at SNAPSHOT_AT. GIFs need the custom data directory and an OpenGL
        context (MUJOCO_GL=osmesa or egl in a container); without them a snapshot keeps its stats.
    checkpoint: checkpoint(step, iteration, label), called with the snapshots (not at step 0), at least every
        CHECKPOINT_EVERY_S, and at the end, whether or not snapshots are on. A failure stops the checkpoints,
        never the training.
    log_exclude: snapshot stats left out of the [rl-viz-snapshot] line (long traces; they stay in the manifest).

    The trainer integration sets `rollout`: rollout(render) -> (frames, stats), the current policy played for a
    snapshot. stats needs `fps` if there are frames; `outcome` (one line) is shown on the clip.
    """

    def __init__(self, custom_data_dir=None, total_timesteps=None, max_minutes=None, task=None, run_info=None,
                 snapshots=True, checkpoint=None, write_every=10, log_exclude=()):
        self.dir = resolve_custom_data_dir(custom_data_dir)
        self.total_timesteps = total_timesteps
        self.max_seconds = max_minutes * 60.0 if max_minutes else None
        self.task = task
        self.run_info = dict(run_info or {})
        self.snapshots_enabled = bool(snapshots)
        self.checkpoint_fn = checkpoint
        self.write_every = write_every
        self.log_exclude = set(log_exclude)
        self.rollout = None
        self.step = 0
        self.columns = None
        self.rows = []
        self.snapshots = []
        self.pending = list(SNAPSHOT_AT)
        self.render_ok = True
        self.checkpoint_step = 0
        self.t0 = self.checkpoint_t = time.time()

    # ------------------------------------------------------------------ run
    def start(self, control_dt, reward_weights, success=False, header=None, step=0):
        """Begin the run: clear the previous run's snapshots, print the run header, take the first snapshot.

        reward_weights: reward term -> weight (the terms logged each iteration, in this order).
        success: the task counts successes (a success_rate column).
        header: more run header fields, after reward_weights (e.g. the snapshot's fixed command).
        """
        self.t0 = self.checkpoint_t = time.time()
        self.step = int(step)
        self.control_dt = control_dt
        self.weights = dict(reward_weights)
        self.term_names = list(self.weights)
        self.header = dict(header or {})
        self.columns = columns(self.term_names, success)
        if self.dir is not None:
            shutil.rmtree(self.dir / "snapshots", ignore_errors=True)  # a previous run's
            (self.dir / "snapshots").mkdir()
        self._write_manifest("training")
        print_line(RUN_PREFIX, self.run_header())
        self.tick()

    def log(self, step, row, terms):
        """One training iteration.

        row: the BASE_COLUMNS (step and time are filled in), success_rate if counted, and the TRAIN_COLUMNS the
            trainer has (the others are None). Values are rounded here.
        terms: reward term -> its weighted contribution this iteration, in reward per second of simulated time.
        """
        self.step = int(step)
        names = [c for c in self.columns if not c.startswith("term_")]
        values = dict(row, step=self.step, time=time.time() - self.t0)
        line = {}
        for c in names:
            v = values.get(c)
            line[c] = int(v) if c in ("step", "fps") and v is not None else round_sig(v, DIGITS.get(c, 5))
        terms = {k: round_sig(terms.get(k), TERM_DIGITS) for k in self.term_names}
        print_line(LOG_PREFIX, {**line, "terms": terms})
        self.rows.append([line[c] for c in names] + [terms[k] for k in self.term_names])
        if len(self.rows) % self.write_every == 0:
            self._write_metrics()
        self.tick()

    def finish(self, step=None):
        """The end of training: the final snapshot and checkpoint, every row written, status "trained"."""
        if step is not None:
            self.step = int(step)
        self.snapshot(label="Final")
        self.checkpoint(label="Final")
        self._write_metrics()
        self._write_manifest("trained")

    def elapsed(self):
        return time.time() - self.t0

    def out_of_time(self):
        return bool(self.max_seconds) and self.elapsed() >= self.max_seconds

    # ------------------------------------------------------- snapshots, checkpoints
    def progress(self):
        p = 0.0
        if self.total_timesteps:
            p = max(p, self.step / self.total_timesteps)
        if self.max_seconds:
            p = max(p, self.elapsed() / self.max_seconds)
        return p

    def tick(self):
        """Snapshot and checkpoint if one is due."""
        due = bool(self.pending) and self.progress() >= self.pending[0]
        if due:
            while self.pending and self.progress() >= self.pending[0]:
                self.pending.pop(0)
            self.snapshot()
        if due or time.time() - self.checkpoint_t >= CHECKPOINT_EVERY_S:
            self.checkpoint()

    def checkpoint(self, label=None):
        step = self.step
        if self.checkpoint_fn is None or step == 0 or step == self.checkpoint_step:
            return
        self.checkpoint_step, self.checkpoint_t = step, time.time()
        try:
            self.checkpoint_fn(step, len(self.rows), label)
        except Exception as e:  # never stop training over it
            print(f"checkpoint export failed ({e!r}); no more checkpoints", flush=True)
            self.checkpoint_fn = None
            return
        print(f"  (checkpoint took {time.time() - self.checkpoint_t:.1f} s)", flush=True)

    def snapshot(self, label=None):
        if not self.snapshots_enabled or self.rollout is None:
            return
        step = self.step
        if self.snapshots and self.snapshots[-1]["step"] == step:
            if label and not self.snapshots[-1].get("label"):
                self.snapshots[-1]["label"] = label
                self._write_manifest("training")
            return
        t = time.time()
        entry = dict(step=step, time=round_sig(t - self.t0, 6), label=label)
        render = self.render_ok and self.dir is not None
        try:
            frames, stats = self.rollout(render)
        except Exception as e:
            if not render:
                print(f"snapshot failed ({e!r}); disabling snapshots", flush=True)
                self.snapshots_enabled = False
                return
            # Most likely no OpenGL context: keep training, and keep the stats
            print(f"snapshot rendering failed ({e!r}); continuing without GIFs", flush=True)
            self.render_ok = False
            try:
                frames, stats = self.rollout(False)
            except Exception as e2:  # not a rendering problem after all
                print(f"snapshot failed ({e2!r}); disabling snapshots", flush=True)
                self.snapshots_enabled = False
                return
            entry["render_error"] = str(e)[:200]
        entry["stats"] = stats
        if frames and self.dir is not None:
            rel = f"snapshots/step_{step:010d}.gif"
            save_gif(frames, self.dir / rel, stats["fps"])
            entry["gif"] = rel
        self.snapshots.append(entry)
        self._write_manifest("training")
        line = {k: v for k, v in entry.items() if k != "stats"}
        line.update({k: v for k, v in stats.items() if k not in self.log_exclude})
        print_line(SNAPSHOT_PREFIX, line)
        outcome = f": {stats['outcome']}," if stats.get("outcome") else ""
        print(f"  (snapshot{outcome} took {time.time() - t:.1f} s)", flush=True)

    # ------------------------------------------------------------------ files
    def run_header(self):
        h = dict(
            format=FORMAT,
            version=FORMAT_VERSION,
            total_timesteps=self.total_timesteps,
            max_minutes=self.max_seconds / 60.0 if self.max_seconds else None,
            control_dt=self.control_dt,
            reward_weights=self.weights,
            **self.header,
            snapshots_enabled=self.snapshots_enabled,
        )
        if self.task is not None:
            h["task"] = self.task
        h.update(self.run_info)
        return h

    def _write_metrics(self):
        if self.dir is None or self.columns is None:
            return
        write_json(self.dir / "metrics.json", dict(version=FORMAT_VERSION, columns=self.columns, rows=self.rows))

    def _write_manifest(self, status):
        if self.dir is None:
            return
        write_json(self.dir / "manifest.json", dict(
            self.run_header(),
            status=status,
            updated=now_iso(),
            files=dict(metrics="metrics.json", eval="eval.json"),
            snapshots=self.snapshots,
        ))


def write_eval(custom_data_dir, results, seconds, seeds, caption=None):
    """eval.json: the benchmark of the exported policy (one result per row of the task's eval columns), and the
    run marked "finished" in the manifest. seeds: the episodes per row, or their seeds."""
    d = resolve_custom_data_dir(custom_data_dir)
    if d is None:
        return
    n = seeds if isinstance(seeds, int) else len(seeds)
    write_json(d / "eval.json", dict(version=FORMAT_VERSION, seconds=seconds, seeds=n, caption=caption,
                                     results=results))
    manifest_path = d / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "finished"
        manifest["updated"] = now_iso()
        write_json(manifest_path, manifest)
