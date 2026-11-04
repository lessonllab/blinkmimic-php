"""Named PHP NPZ references and skill-weighted adaptive motion sampling."""

from pathlib import Path
import math
import warnings
import json
import hashlib

import numpy as np
import torch

from .dataset import SUPERMIMIC_BUNDLES as BUNDLES, MotionLibrary, motion_paths
from .robot import G1_JOINT_NAMES, default_joint_pos_list
from .rotations import mat_to_6d, quat_to_mat, yaw_mat

CONTROL_DT = 0.02
DEFAULT_SKILLS = ("Low-Step", "High-Step", "Low-Climb", "High-Climb", "Ultra-Climb")
# Edit these weights to change skill proportions; selected skills are normalized.
# Locomotion is supported for explicit legacy selections, not loaded by default.
SKILL_SAMPLING_WEIGHTS = {
    "Locomotion": 0.15,
    "Low-Step": 0.15,
    "High-Step": 0.15,
    "Low-Climb": 0.15,
    "High-Climb": 0.15,
    "Ultra-Climb": 0.40,
}


class MotionLib:
    @classmethod
    def for_playback(cls, directory, device):
        """Load all clips in the supplied bundles, independently of training size."""
        root = Path(directory).expanduser().resolve()
        skills = tuple(
            name for name, folder in BUNDLES.items()
            if any((root / folder).glob("*_motion.npz"))
        )
        if not skills:
            raise FileNotFoundError(
                f"No *_motion.npz files in PHP skill bundles under {root}"
            )
        return cls(root, device, skills=skills)

    def __init__(
        self, directory, device, max_clips_per_skill=0, *, skills=DEFAULT_SKILLS
    ):
        if max_clips_per_skill < 0:
            raise ValueError("max_clips_per_skill must be nonnegative")
        root = Path(directory).expanduser().resolve()
        collection_path = root/'collection_manifest.json'
        collection = json.loads(collection_path.read_text()) if collection_path.exists() else None
        self.skill_names = tuple(skills if skills is not None else collection['skills'] if collection else DEFAULT_SKILLS)
        bundles = BUNDLES if collection is None else {s:s for s in collection['skills']}
        if not self.skill_names or len(set(self.skill_names)) != len(self.skill_names):
            raise ValueError("Skills must be a nonempty list without duplicates")
        if any(name not in bundles for name in self.skill_names):
            raise ValueError(f"Unknown PHP skills: {self.skill_names}")
        weights = [SKILL_SAMPLING_WEIGHTS[name] if collection is None else 1. for name in self.skill_names]
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Skill sampling weights must be finite and nonnegative")
        total_weight = sum(weights)
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("Selected skills must have a positive finite total weight")
        self.skill_probabilities = tuple(w / total_weight for w in weights)
        paths, skill_ids, terrain_files = [], [], []
        self.collection_manifest = collection
        self.entry_fraction = .25 if collection else 0.
        self.collection_rows = []
        if collection:
            if collection.get('format') != 'php_reference_collection_v1':
                raise ValueError('Unsupported PHP collection contract')
            rows = [r for r in collection['active'] if r['skill'] in self.skill_names]
            for row in rows:
                if row['cohort'] not in ('original550','repaired69'):
                    raise ValueError('Quarantined/non-active data is not trainable')
                path, terrain = root/row['path'], root/row['terrain']
                if not path.resolve().is_relative_to(root) or not terrain.resolve().is_relative_to(root):
                    raise ValueError('Collection path escapes dataset root')
                for file, expected in ((path,row['sha256']),(terrain,row['terrain_sha256'])):
                    with file.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
                    if actual != expected: raise ValueError(f'Collection checksum mismatch: {file}')
                paths.append(path);terrain_files.append(terrain)
                skill_ids.append(self.skill_names.index(row['skill']))
            if len(rows)!=len(set(r['source_motion_id'] for r in rows)):
                raise ValueError('Duplicate collection source motion IDs')
            self.collection_rows = rows
        else:
            for i, name in enumerate(self.skill_names):
                selected = motion_paths(str(root / bundles[name]))
                if max_clips_per_skill: selected = selected[:max_clips_per_skill]
                paths.extend(selected)
                skill_ids.extend([i] * len(selected))
        self.library = m = MotionLibrary(paths, G1_JOINT_NAMES, device=device,terrain_files=terrain_files if collection else None)
        if collection and m.time_step_total!=sum(r['frames'] for r in self.collection_rows):
            raise ValueError('Collection frame count differs from the indexed per-clip contract')
        self.terrain_format = m.terrain_format
        if not np.isclose(m.fps * CONTROL_DT, 1.0):
            raise ValueError(f"Expected PHP motions at 50 Hz, got {m.fps}")
        self.paths = m.paths
        self.device = device
        self.clip_skill = torch.tensor(skill_ids, device=device)
        self.clip_len = m.motion_lengths
        self.clip_start = m.motion_start_idx
        # SuperMimic uses inclusive ends; PHP's loader uses exclusive ends.
        self.clip_end = m.motion_end_idx - 1
        self.clip_id = torch.repeat_interleave(
            torch.arange(len(paths), device=device), self.clip_len
        )
        self.num_frames = m.time_step_total
        self.root_p = m.body_pos_w[:, 0]
        self.root_R = quat_to_mat(m.body_quat_w[:, 0])
        self.root_v = m.body_lin_vel_w[:, 0]
        self.root_w = m.body_ang_vel_w[:, 0]
        self.q, self.qd = m.joint_pos, m.joint_vel
        self.body_p, self.body_R = m.body_pos_w, quat_to_mat(m.body_quat_w)
        self.body_v, self.body_w = m.body_lin_vel_w, m.body_ang_vel_w
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        # Exclude final frames: every sampled start must have a real next frame.
        valid = m.bin_starts < self.clip_end[m.bin_clips]
        self.bin_start, self.bin_clip = m.bin_starts[valid], m.bin_clips[valid]
        self.bin_size = torch.minimum(
            m.bin_sizes[valid], self.clip_end[self.bin_clip] - self.bin_start
        )
        self.bin_skill = self.clip_skill[self.bin_clip]
        self.num_bins = len(self.bin_start)
        self.frame_bin = (
            torch.searchsorted(
                self.bin_start, torch.arange(self.num_frames, device=device), right=True
            )
            - 1
        )
        self.max_boxes = m.max_boxes
        self.terrain = m.terrain
        self.terrain_files = m.terrain_files
        self.variant_counts = m.variant_counts
        self.box_counts = m.box_counts
        self.context_dim = 33 + 10 * self.max_boxes
        self.terrain_dim = 13 * self.max_boxes
        print(
            f"[PHP SuperMimic] {len(paths)} clips, {self.num_frames} frames, "
            f"{len(self.skill_names)} skills {self.skill_names}, {self.max_boxes} box slots",
            flush=True,
        )

    def sampling_probabilities(self, returns, history, max_return):
        """Configured mass per skill; difficulty sampling only within each skill."""
        if not math.isfinite(max_return) or max_return <= 0:
            raise ValueError("max_return must be finite and positive")
        bad_returns = ~torch.isfinite(returns)
        bad_history = ~torch.isfinite(history) | (history < 0)
        if bool(bad_returns.any() | bad_history.any()):
            warnings.warn(
                "Repairing invalid motion sampling statistics: "
                f"{bad_returns.sum().item()} non-finite returns, "
                f"{bad_history.sum().item()} invalid history weights. "
                "Check simulator rewards or the resumed checkpoint.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Reset unknown returns to the initialization prior. Repair in place
            # so a damaged checkpoint cannot poison every subsequent reset.
            returns[bad_returns] = max_return
            history[bad_history] = 0
        probs = torch.zeros_like(returns)
        for skill in range(len(self.skill_names)):
            mask = self.bin_skill == skill
            values = returns[mask].double()
            if values.numel() == 0:
                raise ValueError(
                    f"No sampleable motion bins for skill {self.skill_names[skill]}"
                )
            uniform = torch.full_like(values, 1.0 / values.numel())
            # Multiply in float64: large finite negative float32 returns can
            # otherwise overflow to +inf before softmax subtracts its maximum.
            logits = torch.nan_to_num(-25 * (values / max_return))
            adaptive = torch.softmax(logits, 0)
            memory = history[mask].double()
            tiny = torch.finfo(memory.dtype).tiny
            memory = memory / memory.max().clamp_min(tiny)
            total = memory.sum()
            memory = torch.where(total > 0, memory / total.clamp_min(tiny), uniform)
            recent = 0.99 * memory + 0.01 * adaptive
            recent = recent / recent.sum()
            history[mask] = recent.to(history.dtype)
            probs[mask] = (
                (0.5 * uniform + 0.4 * adaptive + 0.1 * recent)
                * self.skill_probabilities[skill]
            ).to(probs.dtype)
        return probs

    def future_frames(self, f, count):
        future = f.unsqueeze(-1) + torch.arange(1, count + 1, device=f.device)
        return torch.minimum(future, self.clip_end[self.clip_id[f]].unsqueeze(-1))

    def goal(self, f, count):
        """Upstream 68D motion features, current frame through k-1, clip-clamped."""
        frames = torch.minimum(
            f[:, None] + torch.arange(count, device=f.device),
            self.clip_end[self.clip_id[f]][:, None],
        )
        R = self.root_R[frames]
        gravity = -R[..., 2, :]
        velocity_h = (
            yaw_mat(R).transpose(-1, -2) @ self.root_v[frames].unsqueeze(-1)
        ).squeeze(-1)
        default = torch.tensor(default_joint_pos_list(), device=f.device)
        return torch.cat(
            (
                self.q[frames] - default,
                self.qd[frames],
                gravity,
                self.root_w_local[frames],
                velocity_h,
                self.root_p[frames, 2:3],
            ),
            -1,
        ).flatten(1)

    def legacy_goal(self, f, count):
        future = self.future_frames(f, count)
        Ht = yaw_mat(self.root_R[f]).transpose(-1, -2).unsqueeze(1)
        dp = (
            Ht @ (self.root_p[future] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)
        ).squeeze(-1)
        features = torch.cat(
            (
                dp,
                mat_to_6d(Ht @ self.root_R[future]),
                self.root_v_local[future],
                self.root_w_local[future],
                self.q[future],
                0.1 * self.qd[future],
            ),
            -1,
        )
        return features.flatten(1)

    def ref_state(self, f):
        return (
            self.root_p[f],
            self.root_R[f],
            self.root_v[f],
            self.root_w[f],
            self.q[f],
            self.qd[f],
        )
