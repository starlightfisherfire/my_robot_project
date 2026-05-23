#!/usr/bin/env python3
"""Validate a v2 episode npz against the schema."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.state_schemas.validation import validate_v2_npz
from src.state_schemas.schema_registry import load_schema

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", required=True)
    p.add_argument("--schema", default="configs/state_schema/visual_structured_state_v2.yaml")
    args = p.parse_args()
    schema = load_schema(args.schema)
    r = validate_v2_npz(args.episode, schema)
    print(f"Valid: {r['valid']}")
    for e in r['errors']: print(f"  ❌ {e}")
    for w in r['warnings']: print(f"  ⚠ {w}")
    for k,v in r.get('shapes',{}).items(): print(f"  {k}: {v}")
    sys.exit(0 if r['valid'] else 1)
if __name__ == "__main__": main()
