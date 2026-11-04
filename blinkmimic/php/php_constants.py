from pathlib import Path

from blinkmimic.robot import RobotSpec
from blinkmimic.dr import DomainRandCfg
from supermimic_parkour.robot import G1_JOINT_NAMES, get_robot_cfg
from supermimic_parkour.robot_config import G1_SPHERE_HAND_XML
from supermimic_parkour.dataset import BODY_NAMES


def unused_loader(path):
    raise RuntimeError('PHP uses the named NPZ/terrain loader, not LAFAN1 CSV conversion')


SPEC = RobotSpec(
    name='php', mjcf_path=str(G1_SPHERE_HAND_XML), joint_names=tuple(G1_JOINT_NAMES),
    tracked_bodies=tuple(BODY_NAMES), default_joint_pos=get_robot_cfg().init_state.joint_pos,
    action_scale={name: 1.0 for name in G1_JOINT_NAMES}, com_body='torso_link',
    com_range=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)), robot_cfg=get_robot_cfg,
    motion_dir=str(Path(__file__).resolve().parents[2] / 'data/php_motion'),
    motion_glob='*/*_motion.npz', load_motion=unused_loader, terrain_slots=7,
    motion_skills=('Low-Step', 'High-Step', 'Low-Climb', 'High-Climb'),
    njmax=600, nconmax=128,
    dr=DomainRandCfg(friction_range=(0.4, 1.3),
                     friction_geoms=r'(left|right)_foot[1-7]_collision',
                     joint_offset={'.*': 0.01, '.*ankle.*': 0.06}),
)
