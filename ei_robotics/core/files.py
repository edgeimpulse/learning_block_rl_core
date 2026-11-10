# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Files that Studio, or a Studio extension, may read while the job still writes them."""

import json
import os
from pathlib import Path


def write_json(path, obj, indent=None):
    """Write atomically, so a reader never sees a half-written file. Compact unless `indent` is given."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(obj, indent=indent) if indent is not None else json.dumps(obj, separators=(",", ":"))
    tmp.write_text(text)
    os.replace(tmp, path)


def write_bytes(path, data):
    """Write atomically (see write_json)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
