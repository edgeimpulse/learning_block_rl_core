# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Studio samples: model inputs as Edge Impulse acquisition files, ready for edge-impulse-uploader.

Each sample is one row of values, exactly the model input, so a project can show, test and profile the model
on the inputs it was trained on.
"""

import json
import time
from pathlib import Path

import numpy as np


def acquisition_json(values, sensors, device_name, device_type, interval_ms, units="-"):
    """One acquisition file (unsigned). values: one row per time step; sensors: one name per value."""
    rows = np.atleast_2d(np.asarray(values)).tolist()
    return {
        "protected": {"ver": "v1", "alg": "none", "iat": int(time.time())},
        "signature": "0" * 64,
        "payload": {
            "device_name": device_name,
            "device_type": device_type,
            "interval_ms": interval_ms,
            "sensors": [{"name": n, "units": units} for n in sensors],
            "values": rows,
        },
    }


def write_samples(out_dir, samples, sensors, device_name, device_type, interval_ms, test_fraction=0.2, seed=0):
    """Write <out_dir>/<name>.json for each (name, values, label) in `samples`, and info.labels.

    label: a string, or None for an unlabeled sample. Each sample goes to testing with probability
    test_fraction (drawn with `seed`, so the split is the same every time).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    files = []
    for name, values, label in samples:
        path = f"{name}.json"
        (out / path).write_text(json.dumps(acquisition_json(values, sensors, device_name, device_type, interval_ms)))
        category = "testing" if rng.uniform() < test_fraction else "training"
        files.append({"path": path, "category": category,
                      "label": {"type": "unlabeled"} if label is None else {"type": "label", "label": label}})
    (out / "info.labels").write_text(json.dumps({"version": 1, "files": files}, indent=2))
    return files


def write_io_pairs(out_dir, arrays, inputs, outputs, input_names, output_names, columns=None):
    """io_pairs.npz (`arrays`, as is) and io_pairs.csv: one row per step, the `columns` (name -> one value per
    step, e.g. the episode and step), then the named inputs and outputs, with 9 significant digits."""
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "io_pairs.npz", **arrays)
    columns = columns or {}
    with open(out / "io_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(columns) + list(input_names) + list(output_names))
        for i, (x, y) in enumerate(zip(inputs, outputs)):
            w.writerow([c[i] for c in columns.values()] + [f"{v:.9g}" for v in x] + [f"{v:.9g}" for v in y])
