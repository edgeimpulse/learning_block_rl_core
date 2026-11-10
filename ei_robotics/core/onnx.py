# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Running a block's model.onnx: one input, one output, batch of one."""

import numpy as np


def onnx_policy(path, input_name="input"):
    """`f(x) -> y` for a single float32 input vector, on the CPU."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return lambda x: sess.run(None, {input_name: np.asarray(x, np.float32)[None]})[0][0]
