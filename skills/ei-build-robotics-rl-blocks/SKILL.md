---
name: ei-build-robotics-rl-blocks
description: Build an Edge Impulse custom learning block that trains a robot policy with reinforcement learning in simulation (MuJoCo, Gymnasium, Stable-Baselines3 or another trainer), using the shared ei_robotics package. Covers the block layout, the Studio contract for a block without training data, the environment contract, training on a wall-clock budget, the ONNX export of the actor, the training telemetry for the RL training monitor extension (rl-viz, the task descriptor, snapshots), the simulation bundle and live checkpoints for the RL policy simulator extension (ei-rl-sim), the sim-to-sim evaluation, the Dockerfile, pinning, asset licenses, local tests and pushing. Use when asked to create, port, extend or debug an RL block for a new robot, task or trainer, or to add the monitor or simulator outputs to an existing one.
---

Help the user build an Edge Impulse custom learning block that trains a robot policy by reinforcement learning.
The block simulates the robot itself, so it learns from no project data. It writes `model.onnx` for Studio, plus
two outputs for the Studio extensions:

- the **training telemetry** (format `rl-viz`), read by the
  [RL training monitor](https://github.com/edgeimpulse/rl_training_monitor_extension): learning curves, reward
  terms, clips of the policy at training milestones, the benchmark of the exported model;
- the **simulation bundle** (format `ei-rl-sim`), read by the
  [RL policy simulator](https://github.com/edgeimpulse/rl_policy_simulator_extension): a copy of the simulation
  the policy trains in (`sim/scene.mjb.gz`, `sim/visual.bin.gz`, `sim/sim.json`), so the extension runs the policy
  in the browser, plus the **checkpoints** (`checkpoints/step_<n>.onnx`) it saves while it trains.

The shared code is in this repository (`ei_robotics`). A block keeps only what is specific to its robot and task.
For the general learning-block rules (parameters.json, arguments, output files, Docker, push), see the
`ei-build-custom-learning-blocks` skill; this one adds what is specific to RL.

## The formats belong to the extensions

`rl-viz` is specified in the training monitor's README (sections "Data format", "Task descriptor", "Adding a
robot"), `ei-rl-sim` in the policy simulator's README ("Bundle format", "Live checkpoints", "Parity test").
`ei_robotics` implements both. Never invent a field or a file in a block: if a task needs something new (an
observation kind, an actuator, a column), it is agreed with the extension's owner, added to the extension, then to
`ei_robotics`. A block emits the formats only through `ei_robotics`, never with its own copy.

Read those README sections before writing a task descriptor or a `sim.json`.

## Layers

| Layer | Holds | Example |
|---|---|---|
| `ei_robotics` (this repo) | The Studio contract, the two formats, the trainer integrations | `RLViz`, `RLVizCallback`, `export_onnx`, `write_bundle` |
| The robot and task | The MJCF scene, the environment, the rewards, the randomization. Ideally usable without Edge Impulse | `miniauto_soccer_env.py`, a Gymnasium env; or an mjlab task registration |
| The block | The Studio entry point and the task-specific glue for the extensions | `ei_train.py`, `rl_viz.py`, `sim_bundle.py`, `parameters.json`, `Dockerfile` |

A typical block:

| File | Purpose |
|---|---|
| `<robot>_env.py` (or a dependency) | The environment |
| `train.py` | The training loop, also runnable on its own (`python train.py --timesteps ...`) |
| `ei_train.py` | The ENTRYPOINT: arguments, training, export, verification, evaluation, metrics |
| `rl_viz.py` | `task_info()` (the task descriptor) and `rollout_snapshot()` (the test each snapshot plays) |
| `sim_bundle.py` | `describe()` (sim.json), `write_bundle()`, a Python `Engine` that runs the bundle, `--check` |
| `record_samples.py` | Optional: model inputs and outputs as Studio samples |
| `parameters.json`, `Dockerfile`, `requirements-block.txt` | The block |
| `requirements.txt`, `setup.sh` | A local venv with the same pins |

Start from the closest existing block and replace the robot and the task, rather than from scratch.

## The Studio contract for RL

- **No training data.** Declare `"operatesOn": "regression"`, which every Studio accepts. The project then needs
  an impulse with an input block, **Raw data** and the block, plus a few uploaded samples so Studio starts a job;
  the block ignores them. Where Studio supports self-contained blocks (edgeimpulse PR #16421),
  `"operatesOn": "self_contained_freeform_output"` needs neither. Say in the description that the block needs no
  project data.
- **Arguments.** Use `ei_robotics.core.parser` and `parse_args` (they add `--data-directory`, `--out-directory`,
  `--custom-data-directory` and ignore unknown arguments). Every `parameters.json` entry has a matching argument
  with the same default. Values arrive as strings: give `type=` in argparse.
- **Typical parameters:** training timesteps (millions), max training time (minutes), learning rate, the task's
  own settings (episode length, a curriculum), parallel simulations (0 = automatic), seed; experts' settings under
  `"section": "advanced"`.
- **Outputs** in `--out-directory`: `model.onnx`, `custom_training_metrics.json` (the Experiments tab), and any
  file for deployment (e.g. `deploy_config.json`: the control rate, action scaling, default joint positions).
- **Custom data** in `/home/custom_data` (`custom_data_directory(args, out)`, which returns `None` when it can't
  be used): the telemetry, the bundle, the checkpoints. Studio keeps it with the learn block and the extensions
  read it, even while the job runs.
- **No network at runtime.** Everything the block needs is in the image.
- **Container limits.** `os.cpu_count()` is the host's, not the job's: use
  `default_workers(worker_bytes, main_bytes)`, which reads the cgroup CPU quota and memory limit. Measure a
  worker's memory (each subprocess importing torch costs about 0.45 GB).

```python
from ei_robotics.core import (custom_data_directory, default_workers, describe_dataset, out_directory,
                              parse_args, parser, write_training_metrics)

p = parser("My robot, PPO (Edge Impulse custom learning block)")
p.add_argument("--timesteps-millions", type=float, default=10.0)
p.add_argument("--max-minutes", type=float, default=60.0)
p.add_argument("--n-envs", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
args = parse_args(p)
out = out_directory(args)
n_envs = args.n_envs or default_workers(worker_bytes=0.6e9, main_bytes=1.0e9)
describe_dataset(args.data_directory, note="not used: experience comes from the simulation")
custom_data = custom_data_directory(args, out)
```

## The environment

The task lives in the environment; everything downstream reads it. For `RLVizCallback`, the env provides:

- `reward_scales`: reward term -> weight, as an attribute (it orders and weights the monitor's reward terms);
- `control_dt`: the policy period in seconds;
- each term's **unweighted** value in the step `info`, under the same name; with reward
  `sum(scale * term) * control_dt`, the monitor shows `scale * mean(term)`, reward per second of simulated time;
- `is_failure` / `is_success` in the `info` of an episode's last step (or pass `failure=` / `success=` functions
  to the callback; `ended_early` counts every termination that isn't a time-out as a failure).

```python
class MyRobotEnv(gym.Env):
    def __init__(self, control_dt=0.02, randomize=True, ...):
        self.control_dt = control_dt
        self.reward_scales = dict(track_velocity=1.5, upright=0.5, action_rate=-0.01, fall=-10.0)

    def step(self, action):
        ...
        terms = dict(track_velocity=..., upright=..., action_rate=..., fall=float(fell))
        reward = sum(self.reward_scales[k] * v for k, v in terms.items()) * self.control_dt
        info = dict(terms, is_failure=fell, is_success=reached)
        return obs, reward, fell, timed_out, info
```

Design it for sim-to-real from the start, since the bundle has to describe all of it:

- an observation from what the real robot senses, with its noise and delay; a privileged critic may see more
  (the true state), appended after the actor's part, so the export can slice it off;
- actions in [-1, 1], mapped to targets the same way in training, in the bundle and on the robot;
- the actuator model (servo, PD, the firmware's own mixing) and domain randomization (friction, masses,
  motor parameters, latency), with nominal values for the snapshots and ranges for the evaluation;
- a `randomize=False` mode and a seeded `reset(seed=...)`, so a snapshot or a test is the same every time.

## Training

**Stable-Baselines3:** add `RLVizCallback` to `learn`. It logs every rollout, plays the snapshots, schedules the
checkpoints and finishes the run.

```python
from ei_robotics.rl.sb3 import RLVizCallback
from ei_robotics.rl.viz import RLViz

viz = RLViz(custom_data, total_timesteps=args.timesteps_millions * 1e6, max_minutes=args.max_minutes,
            task=task_info(), checkpoint=checkpoint,              # see the bundle section
            run_info=dict(n_envs=n_envs, seed=args.seed))
callback = RLVizCallback(viz, rollout=lambda policy, render: rollout_snapshot(policy, render=render))
model.learn(total_timesteps=int(args.timesteps_millions * 1e6), callback=[callback, WallClockLimit(args.max_minutes)])
```

`rollout(policy, render)` gets `policy(obs) -> action` on raw observations (the callback applies the current
`VecNormalize` statistics) and returns `(frames, stats)`.

**Another trainer** (rsl_rl, skrl, a custom loop): call `RLViz` yourself and set its `rollout`.

```python
viz.rollout = lambda render: rollout_snapshot(current_policy(), render=render)
viz.start(control_dt=env.control_dt, reward_weights=env.reward_scales, success=True)
for it in range(iterations):
    ...  # collect, update
    viz.log(step, dict(ep_rew=..., ep_len=..., fps=..., fall_rate=..., success_rate=..., entropy=..., lr=...),
            terms={k: weighted_mean_per_second[k] for k in env.reward_scales})
    if viz.out_of_time():
        break
viz.finish(step)
```

**The wall-clock budget.** Studio jobs have a time limit. Stop at `--max-minutes` whatever the timesteps, then
export what you have. `RLVizCallback` doesn't stop training: add a callback whose `_on_step` returns `False` past
the deadline (`WallClockLimit` in the miniAuto block's `train.py`), or check `viz.out_of_time()` in your loop. Snapshots follow whichever of timesteps and time is further.
Say in the parameter's help to keep it below the job limit, and in the README what a run gives at a few budgets.

## Exporting the policy

The ONNX contract the extensions and the deployment rely on: input `"input"` `[1, obs_dim]` float32, output
`"output"` `[1, act_dim]` in [-1, 1], observation normalization baked in, the actor only, opset 10, a single file.

```python
from ei_robotics.rl.sb3 import export_onnx, verify_onnx

export_onnx(model, vec_norm, out / "model.onnx", obs_dim=OBS_DIM)    # obs_dim: the actor's part, with a privileged critic
verify_onnx(model, vec_norm, out / "model.onnx", MyRobotEnv(randomize=False), obs_dim=OBS_DIM)  # vs model.predict, 1e-4
```

Another trainer writes its own exporter to the same contract and checks it against its own inference the same
way. Always verify: a wrong normalization or slice gives a model that loads and acts badly.

## The training monitor

The block gives `RLViz` two task-specific things, usually in `rl_viz.py`.

**`task_info()`**, the task descriptor: how the monitor names the task and what it shows. Every field is
optional; see "Task descriptor" in the monitor's README.

```python
def task_info(episode_seconds=5.0):
    return dict(
        name="SO-101 push cube to target",
        summary="Push a 35 mm cube 6–12 cm onto a target within 2 cm",
        episode=f"one episode is {episode_seconds:g} s of pushing",
        success=dict(label="Success rate"),        # what success_rate counts
        failure=dict(label="Cube lost"),           # what fall_rate and a snapshot's fell_at mean here
        snapshot=dict(
            help="The deterministic policy, nominal physics, the same layout every time.",
            stats=[dict(key="final_distance", label="Final distance", unit="cm", scale=100, digits=1),
                   dict(key="success_at", label="On target at", unit="s", digits=1)],
            trace=dict(key="cube_distance", label="Cube to target", unit="cm", scale=100, digits=1,
                       reference_key="success_radius", reference_label="Success radius")),
        eval=dict(name_label="Layout",
                  columns=[dict(key="successes", label="On target", of="episodes", good="high"),
                           dict(key="final_distance", label="Final distance", unit="cm", scale=100, digits=1)]),
    )
```

**`rollout_snapshot(policy, render)`**: the same deterministic test at every snapshot (nominal physics, a fixed
seed, a fixed command or layout), so the clips compare like with like. It returns the frames (a list of RGB
arrays) and the stats: `fps` when there are frames, `outcome` (a few words: "On target after 1.2 s"),
`outcome_kind` (`success`, `failure`, `neutral`), `fell_at` (seconds or `None`), `sim_return`, the trace (a list
per frame) and the descriptor's stat keys.

- Render with `mujoco.Renderer` at about 10 fps and 240 x 320; GIFs stay small.
- Rendering needs OpenGL: `MUJOCO_GL=osmesa` and `libosmesa6` in the image. If it fails, `RLViz` keeps the stats
  and skips the GIF; never let a snapshot stop training.
- Snapshots are at fixed fractions of the run plus "Final"; a `--no-snapshots` flag turns them off.

After training, `write_eval(custom_data, results, seconds, seeds, caption)` writes the benchmark and marks the
run finished.

## The simulation bundle

`sim_bundle.py` writes `sim/` and proves it reproduces the training environment.

1. **Physics:** `bundle.physics_model(full_model, spec, reset=..., control=...)` compiles the scene without its
   visual-only geoms, then checks that it simulates exactly as the full model (random controls from the reset).
   It returns the `.mjb` bytes.
2. **Visuals:** `bundle.visual_model(full_model)` decimates the meshes into `visual.bin` and lists every visual
   geom. A mesh that may not be redistributed goes in `remote_meshes` instead (see Assets).
3. **`describe(env, model)`:** the block's `sim.json`, read from the environment's own objects (addresses,
   gains, noise, delays, ranges), never retyped constants, so the bundle can't drift from the training. Use only
   the kinds the simulator's README lists (observation terms, signals, actions, actuators, commands,
   randomization). A new kind is a change to the extension first.

   ```python
   def describe(env, model):
       return {
           "format": FORMAT, "version": VERSION,
           "robot": {"name": "myrobot", "display_name": "My robot", "root_body_id": ..., ...},
           "task": {"id": "myrobot-walk", "episode_length_s": env.max_steps * env.control_dt, "kind": "perpetual"},
           "mujoco": {"version": mujoco.__version__, "model": "scene.mjb.gz",
                      "physics_dt": float(model.opt.timestep), "decimation": env.n_substeps, ...},
           "joints": [...], "action": {"type": "joint_position", "offset": [...], "scale": [...]},
           "actuator": {"type": "pd", "kp": [...], "kd": [...], ...},
           "observation": {"dim": OBS_DIM, "terms": [...]},
           "commands": {...}, "terminations": {...}, "randomization": {...},
           "initial_states": initial_states(env),     # sampled from the trained reset distribution
           "policy": {"onnx": "policy/model.onnx", "input": "input", "output": "output", "obs_dim": OBS_DIM, ...},
           "camera": {...},
       }
   ```
4. **`write_bundle(out_dir, ...)`** calls `bundle.write_bundle(out_dir, describe(...), mjb, vis_bin, visual)`,
   which adds the `bundle_id` and writes the three files.
5. **`Engine`:** a Python implementation of the simulator's engine that runs the bundle alone (sim.json and the
   `.mjb`), as `engine.js` does in the browser. `--check` steps it next to the environment, from the same states
   under the same actions, and fails unless observations, qpos and terminations match (around 1e-5 and 1e-8).
   Run it in the Dockerfile so a broken bundle fails the build. The simulator's parity test then compares
   `engine.js` to this `Engine`.

**Live checkpoints.** In `ei_train.py`, before training: `clear_policies(custom_data)`, write the bundle, then
`print_live_header(sim_dir)`. RLViz calls the checkpoint function with the snapshots, at least every 15 minutes,
and at the end; after training, `write_policy(custom_data, onnx_path)`.

```python
from ei_robotics.rl.bundle import clear_policies, print_live_header, save_checkpoint, write_policy

live = False
if custom_data is not None:
    clear_policies(custom_data)
    try:                                   # the bundle is optional: never stop a training over it
        print_live_header(sim_bundle.write_bundle(custom_data, env_kwargs))
        live = True
    except Exception as err:
        print(f"WARNING: no simulation bundle for the policy simulator extension: {err!r}")

def checkpoint(step, iteration, label):
    save_checkpoint(custom_data or out, step, iteration, label,
                    lambda path: export_onnx(callback.model, callback.training_env, path, obs_dim=OBS_DIM), live)
```

**One MuJoCo version.** A `.mjb` only loads in the MuJoCo version that compiled it, and the extension ships each
version a block pins. Moving a block to a new MuJoCo means adding that version to the extension first.

## Evaluation and metrics

After the export, run the **ONNX model**, not the trainer's policy, on a fixed benchmark: randomization and noise
on, fixed seeds, the same whatever the block's parameters, so impulses compare. Several conditions (opponents,
commands, layouts) make the rows of `eval.json`; each result has a `name` and the descriptor's eval columns.
Put a few headline numbers in `custom_training_metrics.json` with `write_training_metrics(out, {...})` (success
rate, failure rate, timesteps trained, training time), which Studio shows in the Experiments tab.

**Studio samples** (optional): `write_samples` and `write_io_pairs` turn the benchmark's model inputs and outputs
into acquisition files and an `info.labels`, so the project has samples to start a job with and to test a
deployment against.

## Dockerfile

```dockerfile
FROM public.ecr.aws/g7a8t7v6/ei-custom-ml-block-base:v1.95.10
WORKDIR /app                                   # never /home or /data

# Software OpenGL for the snapshot GIFs (no GPU, no display)
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

COPY requirements-block.txt ./
RUN pip3 install --break-system-packages -r requirements-block.txt

COPY myrobot_env.py train.py ei_train.py rl_viz.py sim_bundle.py assets/ ./

# Fail the build early: the env steps, offscreen rendering works, the bundle reproduces the env
RUN python3 -c "from myrobot_env import MyRobotEnv; e = MyRobotEnv(); e.reset(seed=0); e.step(e.action_space.sample())"
RUN python3 -c "from rl_viz import rollout_snapshot; f, s = rollout_snapshot(lambda o: [0.0] * ACT_DIM, seconds=0.2)"
RUN python3 sim_bundle.py --out /tmp/sim-check --check && rm -rf /tmp/sim-check

ENTRYPOINT ["python3", "-u", "ei_train.py"]
```

## Dependencies

- Pin exact versions in `requirements-block.txt` (torch CPU wheels from `--extra-index-url
  https://download.pytorch.org/whl/cpu`, mujoco, gymnasium, stable-baselines3, numpy, onnx, onnxruntime, pillow,
  pyopengl), and the same ones in the local `requirements.txt`.
- Install `ei_robotics` from a fixed commit, no git needed in the image:

  ```text
  ei-robotics[rl,sb3] @ https://github.com/edgeimpulse/edgeimpulse-robotics/archive/<commit>.tar.gz
  ```

  To change it alongside a block, install a checkout over the pin: `pip install -e ../edgeimpulse-robotics`.
- An external robot package or task library (e.g. an mjlab task) is pinned to a commit and installed at build
  time.

## Assets and licenses

- Check the license of every model, mesh and dependency before it goes in the repo, the image or the bundle.
  The bundle is a distribution: whatever is in `scene.mjb.gz` and `visual.bin.gz` is shipped to every viewer.
- A mesh that may be downloaded from where it is published but not redistributed: train on primitive collision
  shapes, keep the mesh out of the repo and the bundle, and list it in `remote_meshes` so the viewer downloads it
  from its source. `visual_model` should raise if a mesh geom slips in.
- Credit the robot's and the scene's authors in the README.

## Testing locally

- **A short run** in the image, without network, as Studio runs it:

  ```bash
  docker build -t myrobot-block .
  mkdir -p data out
  docker run --rm --network=none -v $PWD/data:/data -v $PWD/out:/out myrobot-block \
      --data-directory /data --out-directory /out --custom-data-directory /out/custom_data \
      --timesteps-millions 0.2 --max-minutes 5
  ```

  Then check `out/model.onnx`, `out/custom_training_metrics.json`, and in `out/custom_data/`: `manifest.json`
  (status `finished`), `metrics.json`, `eval.json`, `snapshots/*.gif`, `sim/`, `checkpoints/`, `policy/model.onnx`.
- **The monitor:** in its checkout, `DEMO_DIR=/path/to/out node server.js`, then open
  `http://localhost:8080/?demo=1`. It answers the Studio API from `out/custom_data/` and the log `out.log` (save
  the run's stdout there). See its "Run locally".
- **The simulator:** in its checkout, `DEMO_DIR=/path/to/out node server.cjs` and `?demo=1`; it needs
  `custom_data/sim/` and `custom_data/policy/model.onnx`. Its parity test replays the block's `Engine` in
  `engine.js` (see "Run locally" and "Parity test").
- **Unit-sized checks** while developing: `python sim_bundle.py --out /tmp/sim --check [--onnx model.onnx]`, and
  `python -m unittest discover tests` in this repo.

**Refactoring a block** (moving code into `ei_robotics`, renaming, reordering): run the old and the new code with
the same seed and a short training on one process, and compare everything except wall-clock fields (`time`,
`fps`, `updated`): every `[rl-...]` log line, `manifest.json`, `metrics.json`, `eval.json`, the GIFs, the
checkpoints, `model.onnx` and `sim/` byte for byte. Run the old code twice first to confirm it is deterministic.

## Pitfalls

- `parse_args` with unknown arguments: Studio passes more than the block declares.
- Worker count from `os.cpu_count()`, or one worker per CPU without checking memory: the job is killed.
- Files the extensions read while the job runs are written to a temporary name, then renamed (`ei_robotics`
  does this); a block's own files there do the same.
- Leftovers of a previous run in the custom data: clear `snapshots/`, `checkpoints/` and `policy/` at the start
  (`RLViz.start` and `clear_policies` do).
- Constants copied into `sim.json` by hand instead of read from the env: the simulator runs a different robot than
  the one trained, and `--check` is the only thing that catches it.
- Evaluating the trainer's policy instead of the exported ONNX: the benchmark then misses an export bug.
- A snapshot or checkpoint failure that raises: catch it, warn, and keep training.
- Stale descriptions: when the observation or action changes, update `parameters.json`, the README and
  `deploy_config.json` along with it.

## Push

```bash
edge-impulse-blocks init    # first time only: pick the organization, writes .ei-block-config
edge-impulse-blocks push    # Edge Impulse builds the image, including the --check step
```

In a project: upload a few samples (unless the block is self-contained), add an input block, **Raw data** and the
block under **Create impulse**, then train. Set the job's CPU, memory and time limits on the block in Studio after
the push; they are not in `parameters.json`.

## Instructions

1. Ask what the robot is (its model, actuators, sensors), the task and how success and failure are defined, the
   trainer (default: Gymnasium + Stable-Baselines3 PPO on CPU), and whether a real robot will run the policy.
   Look for an existing block with a similar robot or task to start from.
2. Build and test the environment on its own first (`train.py`, a short run that learns something), then wrap it
   in `ei_train.py`.
3. Use `ei_robotics` for everything it covers; don't copy its code into the block. If the block needs a change
   in it, make it there, keeping the formats unchanged.
4. Write `task_info()` and `rollout_snapshot()` from the monitor's README, and `describe()` from the simulator's
   "Bundle format". If the robot needs a kind the simulator doesn't have, say so and stop at the telemetry until
   the extension supports it.
5. Make `sim_bundle.py --check` pass and run it in the Dockerfile.
6. Verify the ONNX export, evaluate the ONNX model on a fixed benchmark, and write the headline metrics.
7. Pin every dependency, `ei_robotics` to a commit, and check asset licenses before anything is committed.
8. Run a short training in the image with `--network=none`, open the outputs in both extensions locally, then
   push.
9. Document in the block's README: the task, the observation and action, the rewards, the randomization, what a
   run gives at a few budgets, and the extensions' outputs.
