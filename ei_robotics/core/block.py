"""The Studio learning-block contract: the arguments Studio passes and the directories a block writes.

Studio runs the block's entrypoint with --data-directory, --out-directory, --info-file and one argument per
parameter in parameters.json, and may pass arguments the block doesn't declare. The block writes its model to
the out directory. Files in the custom data directory (/home/custom_data in Studio) stay with the learn
block after training, and Studio extensions can read them through the API while the job still runs.
"""

import argparse
import os
from pathlib import Path

import numpy as np

CUSTOM_DATA_DIRECTORY = "/home/custom_data"


def parser(description):
    """An ArgumentParser with the arguments every block takes. Add the block's parameters to it."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-directory", type=str, default=None)
    p.add_argument("--out-directory", type=str, required=True)
    p.add_argument("--custom-data-directory", type=str, default=CUSTOM_DATA_DIRECTORY,
                   help="files for the Studio extensions; falls back to <out-directory>/custom_data if not writable")
    return p


def parse_args(p, argv=None):
    """Parse, ignoring the arguments the block doesn't declare (--info-file, and any Studio adds later)."""
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print("Ignoring extra arguments:", unknown)
    return args


def out_directory(args):
    """The out directory, created if missing (Studio doesn't guarantee it exists)."""
    out = Path(args.out_directory)
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_custom_data_dir(path):
    """`path` if it can be created, otherwise None."""
    if path is None:
        return None
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)
    except OSError as e:
        print(f"Custom data directory {path} not writable ({e})")
        return None


def custom_data_directory(args, out):
    """--custom-data-directory, else <out>/custom_data (when running locally), else None."""
    return resolve_custom_data_dir(args.custom_data_directory) or resolve_custom_data_dir(Path(out) / "custom_data")


def describe_dataset(data_dir, note="the dataset is not used for training"):
    """Log what's in the data directory. For blocks that make their own data (simulation): never fails."""
    if not data_dir or not os.path.isdir(data_dir):
        print(f"Data directory {data_dir!r} not found; continuing")
        return
    for name in sorted(os.listdir(data_dir)):
        if name.endswith(".npy"):
            # Some files Studio passes are object arrays (pickled); they are not loaded, as unpickling can run code
            try:
                arr = np.load(os.path.join(data_dir, name), mmap_mode="r", allow_pickle=False)
                print(f"  {name}: shape={arr.shape} dtype={arr.dtype}")
            except Exception as e:
                print(f"  {name}: not inspected ({type(e).__name__})")
    if note:
        print(f"Note: {note}.")
