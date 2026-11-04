"""PHP-specific data, terrain conditioning, contact reward and task criteria.

The trainer/optimizers/EMA/residual policy/mirroring/graphs/probes remain the
upstream BlinkMimic implementation. PHP changes the robot, dataset and physical
task, including torso/hand/foot tracking criteria and real obstacle contacts.
"""
import torch
from pathlib import Path

from blinkmimic.motion import MotionLib, CONTROL_DT
from blinkmimic.mdp import MDP, flatten_state
from blinkmimic.rotations import quat_to_mat, mat_to_6d, mm3, mv3, yaw_mat, rot_angle_sq


def context_features(S, xi, slots, terrain_format='obb'):
    if terrain_format=='vertices':
        vertices=xi[..., -(24*slots+1):-1].unflatten(-1,(slots,8,3))
        Ht=yaw_mat(S[1]).transpose(-1,-2)[...,None,None,:,:]
        active=vertices[...,2].mean(-1)>-20
        relative=mv3(Ht,vertices-S[0][...,None,None,:]).flatten(-2)
        features=torch.cat((torch.where(active[...,None],relative,0.),active[...,None].float()),-1).flatten(-2)
        return torch.cat((xi[...,:-(24*slots+1)],features),-1)
    boxes = xi[..., -10 * slots:].unflatten(-1, (slots, 10))
    Ht = yaw_mat(S[1]).transpose(-1, -2).unsqueeze(-3)
    active = boxes[..., 2] > -20
    features = torch.cat((
        mv3(Ht, boxes[..., :3] - S[0].unsqueeze(-2)),
        mat_to_6d(mm3(Ht, quat_to_mat(boxes[..., 3:7]))), boxes[..., 7:],
        active.unsqueeze(-1).float(),
    ), -1)
    features = torch.where(active[..., None], features, 0.0).flatten(-2)
    return torch.cat((xi[..., :-10 * slots], features), -1)


def support_height(points, boxes):
    """Top hit of a vertical ray on OBB terrain, with ground plane fallback."""
    origin = torch.cat((points[..., :2], torch.full_like(points[..., 2:3], 10.0)), -1)
    R = quat_to_mat(boxes[..., 3:7])
    local = mv3(R.transpose(-1, -2).unsqueeze(-4), origin.unsqueeze(-2) - boxes[..., None, :, :3])
    direction = -R[..., 2, :].unsqueeze(-3)
    parallel = direction.abs() < 1e-7
    inv = torch.where(parallel, 1.0, direction).reciprocal()
    half = boxes[..., None, :, 7:] * 0.5
    t1, t2 = (-half - local) * inv, (half - local) * inv
    lo = torch.where(parallel, -torch.inf, torch.minimum(t1, t2)).amax(-1)
    hi = torch.where(parallel, torch.inf, torch.maximum(t1, t2)).amin(-1)
    hit = (hi >= lo.clamp(min=0)) & ~(parallel & (local.abs() > half)).any(-1)
    hit &= boxes[..., None, :, 2] > -20
    heights = torch.where(hit, 10.0 - lo, 0.0)
    return heights.amax(-1).clamp(min=0.0)


class PhpMotionLib(MotionLib):
    def __init__(self, directory, robot, kin, device):
        from supermimic_parkour.motion import MotionLib as PhpSource
        source = PhpSource(directory, device, skills=robot.motion_skills)
        self.__dict__.update(source.__dict__)
        self.terrain_slots = robot.terrain_slots
        if source.max_boxes > self.terrain_slots:
            raise ValueError('PHP terrain exceeds configured obstacle capacity')
        if source.max_boxes < self.terrain_slots:
            pad = torch.zeros(*self.terrain.shape[:-2], self.terrain_slots-source.max_boxes, 10, device=device)
            pad[...,2], pad[...,3], pad[...,7:] = -50, 1, .02
            self.terrain = torch.cat((self.terrain, pad), -2)
        self.max_boxes = self.terrain_slots
        root=Path(directory).expanduser().resolve()
        self.names = [str(p.resolve().relative_to(root)).replace('_motion.npz', '') for p in source.paths]
        self.bin_clip = source.bin_clip
        self.bin_start = source.bin_start
        self.skill_weights = torch.tensor(source.skill_probabilities, device=device)
        counts = torch.bincount(self.bin_skill, minlength=len(self.skill_names))
        self.base_bin_probs = self.skill_weights[self.bin_skill] / counts[self.bin_skill]
        clip_counts=torch.bincount(self.clip_skill,minlength=len(self.skill_names))
        self.base_clip_probs=self.skill_weights[self.clip_skill]/clip_counts[self.clip_skill]
        if self.terrain_format=='vertices':
            from .mesh import MeshBank
            self.mesh_bank=MeshBank(self)
            self.mesh_ray_coeff=torch.as_tensor(self.mesh_bank.coefficients.reshape(-1,self.max_boxes,12,9),device=device)
            self.mesh_ray_valid=torch.as_tensor(self.mesh_bank.valid.reshape(-1,self.max_boxes,12),device=device)
            print(f'[PHP mesh] {self.max_boxes} slots, {self.mesh_bank.num_assets} tetrahedral assets; source surfaces retained',flush=True)
        self.ref_contact = None
        self.ref_contact_floor = None

    def goal_dim(self, K):
        return (15 + 2 * self.q.shape[-1]) * K

    goal = MotionLib.goal
    future_frames = MotionLib.future_frames
    ref_state = MotionLib.ref_state

    def set_ref_contact(self, feet, margin=0.015, vmax=0.5, q=0.05):
        # Unlike LAFAN1's floor, PHP feet also stand on raised platforms.
        floor = torch.zeros(self.num_frames, len(feet), device=self.device)
        ref_contact = torch.zeros_like(floor)
        for c in range(len(self.names)):
            start, end = int(self.clip_start[c]), int(self.clip_end[c]) + 1
            positions = self.body_p[start:end, feet]
            if self.terrain_format=='vertices':
                from .mesh import support_height as mesh_height
                gid=c*self.terrain.shape[1]
                height=mesh_height(positions,self.mesh_ray_coeff[gid],self.mesh_ray_valid[gid])
            else:
                boxes = self.terrain[c, 0].expand(end - start, -1, -1)
                height=support_height(positions,boxes)
            clearance = positions[..., 2] - height
            threshold = clearance.quantile(q, dim=0)
            floor[start:end] = threshold
            ref_contact[start:end] = ((clearance < threshold + margin) &
                                      (self.body_v[start:end, feet, :2].norm(dim=-1) < vmax)).float()
        self.ref_contact, self.ref_contact_floor = ref_contact, floor


class PhpMDP(MDP):
    def __init__(self, kin, motion, default_q, xi_dim, sensing):
        super().__init__(kin, motion, default_q, xi_dim, sensing)
        from supermimic_parkour.dataset import BODY_NAMES
        self.anchor = BODY_NAMES.index('torso_link')
        self.ee = [BODY_NAMES.index(n) for n in ('left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link')]
        self.critic_extra += motion.terrain_slots-1 if motion.terrain_format=='vertices' else 3 * motion.terrain_slots

    def _tracking(self, S, f):
        bp, bR, bv, bw = self.kin.forward(*S)
        ap, aR = bp[..., self.anchor, :], bR[..., self.anchor, :, :]
        rp, rR = self.motion.body_p[f, self.anchor], self.motion.body_R[f, self.anchor]
        dR = yaw_mat(mm3(aR, rR.transpose(-1, -2)))
        anchor = torch.cat((ap[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + mv3(dR.unsqueeze(-3), self.motion.body_p[f] - rp.unsqueeze(-2))
        ref_bR = mm3(dR.unsqueeze(-3), self.motion.body_R[f])
        return bp, bR, bv, bw, ap, aR, rp, rR, ref_bp, ref_bR

    def reward(self, S, f, a, a_prev, terms=False, w_contact=0.0, xi=None):
        bp, bR, bv, bw, ap, aR, rp, rR, ref_bp, ref_bR = self._tracking(S, f)
        c, m = self.cfg, self.motion
        e_pos = (ref_bp - bp).square().sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = (m.body_v[f] - bv).square().sum(-1).mean(-1)
        e_av = (m.body_w[f] - bw).square().sum(-1).mean(-1)
        e_rp, e_ro = (rp - ap).square().sum(-1), rot_angle_sq(rR, aR)
        track = (torch.exp(-e_pos / c.body_pos_std**2) + torch.exp(-e_ori / c.body_ori_std**2)
                 + torch.exp(-e_lv / c.body_linvel_std**2) + torch.exp(-e_av / c.body_angvel_std**2)
                 + .5 * torch.exp(-e_rp / c.root_pos_std**2) + .5 * torch.exp(-e_ro / c.root_ori_std**2))
        rate = (a - a_prev).square().sum(-1)
        limit = (torch.relu(S[4] - self.soft_hi) + torch.relu(self.soft_lo - S[4])).sum(-1)
        r = (track + c.w_action_rate * rate + c.w_joint_limit * limit) * CONTROL_DT
        if w_contact:
            feet = bp[..., self.feet, :]
            if m.terrain_format=='vertices':
                from .mesh import support_height as mesh_height
                gid=xi[...,-1].long() if xi is not None else m.clip_id[f]*m.terrain.shape[1]
                height=mesh_height(feet,m.mesh_ray_coeff[gid],m.mesh_ray_valid[gid])
            else:
                boxes = (xi[..., -10*m.terrain_slots:].unflatten(-1, (m.terrain_slots, 10)) if xi is not None else m.terrain[m.clip_id[f], 0])
                height=support_height(feet,boxes)
            z = feet[..., 2] - height
            threshold = m.ref_contact_floor[f] + .015
            contact = torch.sigmoid((threshold - z) / .005)
            ref = m.ref_contact[f]
            slip = bv[..., self.feet, :2].square().sum(-1)
            r += w_contact * ((1 - (contact - ref).square()) + ref * torch.exp(-slip / .3**2)).mean(-1) * CONTROL_DT
        if terms:
            return r, {'track': track*CONTROL_DT, 'e_body_pos':e_pos,'e_body_ori':e_ori,'e_root_pos':e_rp,
                       'action_rate':rate,'joint_limit':limit}
        return r

    def termination_errors(self, S, f):
        bp, _, _, _, ap, aR, rp, rR, ref_bp, _ = self._tracking(S, f)
        return ((rp-ap).norm(dim=-1), (rR[...,2,2]-aR[...,2,2]).abs(),
                (ref_bp[...,self.ee,:]-bp[...,self.ee,:]).norm(dim=-1).amax(-1))

    def termination(self, S, f):
        pos, tilt, ee = self.termination_errors(S, f)
        return (pos > .5) | (tilt > .8) | (ee > .5) | ~torch.isfinite(flatten_state(S)).all(-1)

    def soft_termination(self, S, f):
        pos, tilt, ee = self.termination_errors(S, f)
        return 1 - torch.sigmoid((.5-pos)/.02)*torch.sigmoid((.8-tilt)/.04)*torch.sigmoid((.5-ee)/.02)

    def critic_features(self, S, f, xi):
        features = super().critic_features(S, f, xi)
        return torch.cat((features[..., :-self.xi_dim], context_features(S, xi, self.motion.terrain_slots,self.motion.terrain_format)), -1)
