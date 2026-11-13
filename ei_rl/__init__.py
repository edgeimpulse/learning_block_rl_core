# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Shared code for Edge Impulse custom learning blocks that train robot policies by reinforcement learning.

ei_rl.core     the Studio learning-block contract: arguments, output files, custom data, Studio samples
ei_rl.viz      the training telemetry the RL training monitor extension reads (rl-viz), whatever the trainer
ei_rl.bundle   the simulation bundle the RL policy simulator extension runs (ei-rl-sim), and the checkpoints
ei_rl.sb3      the Gymnasium + Stable-Baselines3 integration: the RLViz callback, the ONNX export

Import the modules directly: bundle needs mujoco, sb3 needs torch and Stable-Baselines3. core imports none of
the others.
"""

__version__ = "0.1.0"
