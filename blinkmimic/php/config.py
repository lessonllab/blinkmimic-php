"""Resolve original OBB bundles and explicitly indexed PHP mesh collections."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np


def configure(robot, directory=None, config=None, skills=None):
    root=Path(directory or robot.motion_dir).expanduser().resolve()
    path=root/'collection_manifest.json'
    collection=json.loads(path.read_text()) if path.exists() else None
    config=config or {}
    names=tuple(skills or config.get('skills') or (collection['skills'] if collection else robot.motion_skills))
    if collection:
        slots=1
        for row in collection['active']:
            if row['skill'] not in names: continue
            with np.load(root/row['terrain'],allow_pickle=False) as src:
                slots=max(slots,len(src['vertices']))
        return replace(robot,motion_skills=names,terrain_slots=slots,terrain_format='vertices')
    return replace(robot,motion_skills=names)
