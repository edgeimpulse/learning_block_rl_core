"""custom_training_metrics.json: numbers Studio shows on the learn block page and in Experiments."""

import json
from pathlib import Path

FILE = "custom_training_metrics.json"


def write_training_metrics(out_dir, metrics):
    """metrics: name -> value, in display order. None values are left out, floats rounded to 4 decimals."""
    items = [{"name": k, "value": round(v, 4) if isinstance(v, float) else v}
             for k, v in metrics.items() if v is not None]
    (Path(out_dir) / FILE).write_text(json.dumps({"type": "metrics", "version": 1, "metrics": items}, indent=2))
    print(f"\nTraining metrics ({FILE}):")
    for m in items:
        print(f"  {m['name']}: {m['value']}")
    return items
