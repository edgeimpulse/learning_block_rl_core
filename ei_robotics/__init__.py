# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Shared code for Edge Impulse custom learning blocks that train robot policies.

ei_robotics.core   the Studio learning-block contract (arguments, output files, custom data, Studio samples)
ei_robotics.rl     RL in simulation: the training monitor format, the simulation bundle, trainer integrations

core never imports rl.
"""

__version__ = "0.1.0"
