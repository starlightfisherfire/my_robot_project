"""Write v2 episodes to .npz files."""

import json, time, uuid
from pathlib import Path
from typing import Optional
import numpy as np


class V2EpisodeWriter:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.episodes_dir = self.output_dir / "episodes"
        self.metadata_dir = self.output_dir / "metadata"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self._metadata = []
    
    def save_episode(self, payload: dict, metadata: Optional[dict] = None) -> str:
        ep_id = str(uuid.uuid4())[:12]
        ep_path = self.episodes_dir / f"{ep_id}.npz"
        
        # Ensure schema_version
        if "schema_version" not in payload:
            payload["schema_version"] = np.array(["0.1"], dtype=object)
        
        np.savez(ep_path, **payload)
        
        meta = metadata.copy() if metadata else {}
        meta["episode_id"] = ep_id
        meta["episode_path"] = str(ep_path)
        meta["saved_at"] = time.time()
        self._metadata.append(meta)
        
        # Append to episodes.jsonl
        with open(self.metadata_dir / "episodes.jsonl", "a") as f:
            f.write(json.dumps(meta) + "\n")
        
        return ep_id
