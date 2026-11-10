# edgeimpulse-robotics

Shared code for [Edge Impulse custom learning blocks](https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-learning-blocks)
that train robot policies. A block keeps what is specific to its robot and task (the environment, the
rewards, the snapshot it plays, its `sim.json` description). This package holds what every block would
otherwise copy: the Studio arguments and output files, the formats the Studio extensions read, and the
trainer integrations.

## The Studio extensions

A block writes files to its custom data directory (`/home/custom_data` in Studio), which Studio keeps with
the learn block. Two Studio extensions read them, while the job trains and after:

- [rl_training_monitor_extension](https://github.com/edgeimpulse/rl_training_monitor_extension) reads the
  **training telemetry** (format `rl-viz`): log lines and files with the learning curves, the reward terms,
  clips of the policy at training milestones, and the benchmark of the exported model.
- [rl_policy_simulator_extension](https://github.com/edgeimpulse/rl_policy_simulator_extension) reads the
  **simulation bundle** (format `ei-rl-sim`): a copy of the simulation the policy trains in, saved in `sim/` as
  three files:
  - `scene.mjb.gz`: the physics model, compiled as the policy trained on it;
  - `visual.bin.gz`: what to draw;
  - `sim.json`: the robot and the task, i.e. what the policy observes, how its actions drive the motors, the
    commands, the start states.

  The extension runs the policy in the browser, in the physics it trained in. It also runs the
  **checkpoints** the block saves in `checkpoints/` while it trains, so you can watch the policy improve before
  the job ends.

Each extension's README specifies its format.

## Layout

| Module | What | Needs |
|---|---|---|
| `ei_robotics.core` | The learning-block contract: arguments (`parser`, `parse_args`), the out and custom data directories, `custom_training_metrics.json`, Studio samples (acquisition files, `info.labels`), CPU and memory limits in a container, running `model.onnx` | numpy |
| `ei_robotics.rl.viz` | The training telemetry (`RLViz`, `write_eval`), whatever the trainer, and when to take snapshots and checkpoints | `[rl]` |
| `ei_robotics.rl.bundle` | The simulation bundle: the physics model without its visual-only geoms (`physics_model`), the decimated visuals (`visual_model`), writing `sim/` (`write_bundle`), and the checkpoints (`save_checkpoint`). The block describes its own robot and task in `sim.json` | `[rl]` |
| `ei_robotics.rl.sb3` | Gymnasium + Stable-Baselines3: `RLVizCallback` and the ONNX export of the actor with its observation normalization | `[sb3]` |

`core` never imports `rl`, so a block that isn't RL can use it alone.

## Install

There is no PyPI release. A block installs a fixed commit, in its requirements file:

```text
ei-robotics[rl,sb3] @ https://github.com/edgeimpulse/edgeimpulse-robotics/archive/<commit>.tar.gz
```

The archive URL needs no git in the image. The package's dependencies are lower bounds only; the block pins
the exact versions it trains with. To change this code alongside a block, install your checkout over the
pinned one: `pip install -e ../edgeimpulse-robotics`.

**Torch on Linux.** The `[sb3]` extra needs torch. On Linux, the torch wheels on PyPI also install NVIDIA's CUDA
libraries (`nvidia-*` packages, several GB, under NVIDIA's proprietary license) and triton. A block that trains on
CPU installs the CPU-only wheels instead, by adding the PyTorch index to its requirements file and pinning a
`+cpu` build:

```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.14.1+cpu
```

## Using it from a block

```python
from ei_robotics.core import custom_data_directory, out_directory, parse_args, parser
from ei_robotics.rl.sb3 import RLVizCallback, export_onnx
from ei_robotics.rl.viz import RLViz

p = parser("My robot, PPO")
p.add_argument("--timesteps-millions", type=float, default=10.0)
args = parse_args(p)
out = out_directory(args)

viz = RLViz(custom_data_directory(args, out), total_timesteps=args.timesteps_millions * 1e6,
            task=my_task_info())                       # what the monitor shows for this task
callback = RLVizCallback(viz, rollout=my_snapshot)     # my_snapshot(policy, render) -> (frames, stats)
model.learn(total_timesteps=int(args.timesteps_millions * 1e6), callback=callback)
export_onnx(model, vec_norm, out / "model.onnx")
```

The env provides `reward_scales` (term -> weight) and `control_dt` as attributes, each reward term's value in
the step info, and `is_failure` / `is_success` in the last info of an episode. Another trainer gets the same
outputs by calling `RLViz.start`, `log` and `finish` itself.

## The formats belong to the extensions

`rl-viz` is owned by the training monitor extension and `ei-rl-sim` by the policy simulator extension; their
READMEs are the specification. This package implements them, and changes only when they do: a new field or
file is agreed with the extension's owner first.

## Building a new block

[skills/ei-build-robotics-rl-blocks](skills/ei-build-robotics-rl-blocks/SKILL.md) is a guide to building an RL block
with this package, from the environment to the push. It is an [Agent Skill](https://agentskills.io): a folder
with a `SKILL.md`, which any agent that supports skills can load. Copy the folder into your agent's skills
directory (for Claude Code, `~/.claude/skills/`), or read it as a plain guide.

## Tests

```bash
python -m unittest discover tests
```

The tests need numpy only; the ones for `rl` run when mujoco and pillow are installed.

## License

[The Clear BSD License](LICENSE). See [NOTICE](NOTICE) for the dependencies' licenses.
