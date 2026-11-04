"""Data and network contract for the PHP BlinkMimic checkpoint format."""
import hashlib
import json
from pathlib import Path

FORMAT = 'php_blinkmimic_v1'
MESH_FORMAT = 'php_blinkmimic_mesh_v2'


def motion_manifest(motion):
    entries = []
    for name, path, terrain in zip(motion.names, motion.paths, motion.terrain_files):
        def digest(p):
            with p.open('rb') as stream:
                return hashlib.file_digest(stream, 'sha256').hexdigest()
        entries.append({'name': name, 'motion_sha256': digest(path), 'terrain_sha256': digest(terrain)})
    return entries


def add_metadata(args, robot, motion, mdp, layout, goal_dim):
    from importlib.metadata import version
    import torch
    source = json.loads((Path(__file__).resolve().parents[1] / 'UPSTREAM.json').read_text())
    args.blinkmimic_source = source
    args.runtime = {name: version(name) for name in ('torch', 'mjlab', 'mujoco', 'mujoco-warp', 'warp-lang')}
    args.runtime['gpu'] = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
    args.iteration_unit = f'update_group ({args.wm_batches} world-model + 1 policy + critic; {1/args.utd:g} vector steps at utd={args.utd:g})'
    if robot.terrain_slots:
        args.php_format = MESH_FORMAT if robot.terrain_format=='vertices' else FORMAT
        args.terrain_format=robot.terrain_format
        if robot.terrain_format=='vertices':
            args.collection_rows=motion.collection_rows
            args.collection_quality={'repaired_verified':sum(r['cohort']=='repaired69' for r in motion.collection_rows),'original_pending':sum(r['cohort']=='original550' for r in motion.collection_rows)}
            args.terrain_collision='compound tetrahedra preserving all source triangles'
            args.entry_fraction=motion.entry_fraction
        args.motion_manifest = motion_manifest(motion)
        args.obs_dim, args.goal_dim, args.xi_dim = mdp.obs_dim, goal_dim, layout.dim
        args.num_clips, args.num_frames = len(motion.names), motion.num_frames
        args.terrain_slots = robot.terrain_slots
        args.joint_names = list(robot.joint_names)
        args.tracked_bodies = list(robot.tracked_bodies)


def validate(config, robot, motion, mdp, goal_dim):
    if not robot.terrain_slots:
        return
    expected_format=MESH_FORMAT if robot.terrain_format=='vertices' else FORMAT
    if config.get('php_format') != expected_format:
        raise ValueError('Expected a PHP BlinkMimic checkpoint; old PHP teacher weights use a different architecture')
    expected = {'obs_dim': mdp.obs_dim, 'goal_dim': goal_dim, 'xi_dim': mdp.xi_dim,
                'terrain_slots': robot.terrain_slots, 'joint_names': list(robot.joint_names),
                'tracked_bodies': list(robot.tracked_bodies)}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Checkpoint {key} mismatch: saved={config.get(key)!r}, current={value!r}')
    if config.get('motion_manifest') != motion_manifest(motion):
        raise ValueError('Motion/terrain contents differ from this checkpoint; use its recorded PHP dataset and skills')
