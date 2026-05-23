"""Read v2 episodes."""

import json
from pathlib import Path
from typing import Optional
import numpy as np


def load_episode(path: str | Path) -> dict:
    return dict(np.load(path, allow_pickle=True))


def load_metadata(metadata_path: str | Path) -> list[dict]:
    with open(metadata_path) as f:
        return [json.loads(line) for line in f if line.strip()]
