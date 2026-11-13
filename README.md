# ei_rl

A Python package for building [Edge Impulse custom learning blocks](https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-learning-blocks)
that train robot policies with reinforcement learning (RL) in simulation. It isn't a block itself: each RL block
installs it and keeps only what is specific to its robot and task (the environment, the rewards, the robot model,
its `sim.json` description).

## How it fits together

```text
RL block  --(ei_rl)-->  model.onnx + files in custom data  --(Studio)-->  RL extensions in Studio
```

1. **The block** trains the policy. It calls `ei_rl` for everything a Studio learning block must do (read the
   arguments, write `model.onnx` and the metrics) and to write the files the extensions read.
2. **Studio** keeps the block's custom data directory (`/home/custom_data`) with the learn block.
3. **Two Studio extensions** read those files, while the job trains and after: the training monitor shows the
   learning curves and clips of the policy, and the policy simulator runs the policy in the browser.

The block never writes these files itself. `ei_rl` is the one implementation of their formats, so when an
extension changes, the blocks get the change by moving their pin to a newer commit of this package.

## Install

There is no PyPI release. A block installs a fixed commit, in its requirements file:

```text
ei-rl[mujoco,sb3] @ https://github.com/edgeimpulse/learning_block_rl_core/archive/<commit>.tar.gz
```

| Extra | Adds | For |
|---|---|---|
| (none) | numpy, pillow | `ei_rl.core`, `ei_rl.viz` |
| `mujoco` | mujoco | `ei_rl.bundle` |
| `sb3` | Stable-Baselines3, torch, onnx, onnxruntime | `ei_rl.sb3` |
| `mjlab` | mjlab (with rsl_rl, torch, mujoco) | `ei_rl.mjlab`, `ei_rl.bundle` |

The archive URL needs no git in the image. The package's dependencies are lower bounds only; the block pins
the exact versions it trains with. An mjlab block usually gets mjlab from its robot package's lock file and
installs `ei-rl` into that environment, which keeps those versions. To change this code alongside a block, install your checkout over the
pinned one: `pip install -e ../learning_block_rl_core`.

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
from ei_rl.core import custom_data_directory, out_directory, parse_args, parser
from ei_rl.sb3 import RLVizCallback, export_onnx
from ei_rl.viz import RLViz

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
the step info, and `is_failure` / `is_success` in the last info of an episode.

With mjlab and rsl_rl, `RslRlViz` hooks the runner instead; the reward terms and time-outs come from the task's
managers:

```python
from ei_rl.mjlab import RslRlViz, checkpoint

runner = OnPolicyRunner(env, agent_cfg, log_dir, device)
viz = RLViz(custom_data, total_timesteps=iterations * agent_cfg["num_steps_per_env"] * num_envs,
            max_minutes=args.max_minutes, task=my_task_info(), checkpoint=checkpoint(runner, custom_data))
hook = RslRlViz(viz, rollout=my_snapshot)              # my_snapshot(policy, render) -> (frames, stats)
hook.attach(runner, env)
model_pt = hook.learn(iterations, init_at_random_ep_len=True)   # stops at the time budget
hook.finish()
```

Another trainer gets the same outputs by calling `RLViz.start`, `log` and `finish` itself.

## Modules

| Module | What | Extra |
|---|---|---|
| `ei_rl.core` | The learning-block contract: arguments (`parser`, `parse_args`), the out and custom data directories, `custom_training_metrics.json`, Studio samples (acquisition files, `info.labels`), CPU and memory limits in a container, running `model.onnx` | |
| `ei_rl.viz` | The training telemetry for the monitor (`RLViz`, `write_eval`), whatever the trainer, and when to take snapshots and checkpoints | |
| `ei_rl.bundle` | The simulation bundle for the simulator: the physics model without its visual-only geoms (`physics_model`), the decimated visuals (`visual_model`), writing `sim/` (`write_bundle`), and the checkpoints (`save_checkpoint`) | `[mujoco]` |
| `ei_rl.sb3` | Gymnasium + Stable-Baselines3: `RLVizCallback` and the ONNX export of the actor with its observation normalization | `[sb3]` |
| `ei_rl.mjlab` | mjlab + rsl_rl: `RslRlViz` (the runner's logger hook and the time budget) and `checkpoint` (mjlab's ONNX export) | `[mjlab]` |

`ei_rl.core` imports nothing from the other modules, and has no RL code: a block that isn't RL can use it alone.

## What the extensions read

- [rl_training_monitor_extension](https://github.com/edgeimpulse/rl_training_monitor_extension) reads the
  **training telemetry** (format `rl-viz`, written by `ei_rl.viz`): log lines and files with the learning curves,
  the reward terms, clips of the policy at training milestones, and the benchmark of the exported model.
- [rl_policy_simulator_extension](https://github.com/edgeimpulse/rl_policy_simulator_extension) reads the
  **simulation bundle** (format `ei-rl-sim`, written by `ei_rl.bundle`): a copy of the simulation the policy
  trains in, saved in `sim/` as three files:
  - `scene.mjb.gz`: the physics model, compiled as the policy trained on it;
  - `visual.bin.gz`: what to draw;
  - `sim.json`: the robot and the task, written by the block: what the policy observes, how its actions drive
    the motors, the commands, the start states.

  The simulator also runs the **checkpoints** the block saves in `checkpoints/` while it trains, so you can
  watch the policy improve before the job ends.

The formats belong to the extensions: each extension's README is the specification of its format. This package
implements them, and changes only when they do. A new field or file is agreed with the extension's owner first.

## Building a new block

[skills/ei-build-robotics-rl-blocks](skills/ei-build-robotics-rl-blocks/SKILL.md) is a guide to building an RL block
with this package, from the environment to the push. It is an [Agent Skill](https://agentskills.io): a folder
with a `SKILL.md`, which any agent that supports skills can load. Copy the folder into your agent's skills
directory (for Claude Code, `~/.claude/skills/`), or read it as a plain guide.

## Tests

```bash
python -m unittest discover tests
```

The tests need numpy only. They cover `ei_rl.core`, `ei_rl.viz` and the `ei_rl.mjlab` hook (with a stand-in
runner); `ei_rl.bundle` and `ei_rl.sb3` have no tests yet.

## License

[The Clear BSD License](LICENSE). See [NOTICE](NOTICE) for the dependencies' licenses.
