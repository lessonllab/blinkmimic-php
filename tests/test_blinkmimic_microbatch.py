"""Execute the real trainer functions without its import-time simulator/CLI.

Check the memory adaptation against an unchunked BPTT update, including actor,
critic, normalizer and EMA state. Noise is disabled to compare the same samples.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from blinkmimic.dr import XiLayout
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import flatten_state, unflatten_state, inv_dim, make_mdp
from blinkmimic.motion import MotionLib
from blinkmimic.networks import Policy, WorldModel, Critic
from blinkmimic.objective import actor_objective
from blinkmimic.robot import get_robot


class MicrobatchEquivalence(unittest.TestCase):
    def test_real_bptt_update_matches_full_batch(self):
        torch.manual_seed(31)
        robot = get_robot('php')
        kin = Kinematics(robot, 'cpu')
        layout = XiLayout(robot, 'cpu')
        B, K, H, HZ, F = 8, 2, 2, 4, 50
        q0 = torch.tensor(robot.default_joint_pos_list())
        # A consistent synthetic reference uses the actual PHP robot FK.
        motion = MotionLib.__new__(MotionLib)
        motion.terrain_slots = 7
        motion.terrain_format = 'obb'
        motion.root_p = torch.tensor([0., 0., .8]).repeat(F, 1)
        motion.root_R = torch.eye(3).repeat(F, 1, 1)
        motion.root_v = motion.root_w = torch.zeros(F, 3)
        motion.root_v_local = motion.root_w_local = torch.zeros(F, 3)
        motion.q, motion.qd = q0.repeat(F, 1), torch.zeros(F, 29)
        motion.clip_id, motion.clip_end = torch.zeros(F, dtype=torch.long), torch.tensor([F-1])
        motion.body_p, motion.body_R, motion.body_v, motion.body_w = kin.forward(*motion.ref_state(torch.arange(F)))
        mdp = make_mdp(kin, motion, q0, layout.dim, robot.sensing)
        mdp.add_noise = lambda value: value
        goal_dim = motion.goal_dim(K)
        pi = Policy(mdp.obs_dim, goal_dim, 29, hist=H, enc_hidden=(32, 16, 8), dec_hidden=(32, 16, 8))
        torch.nn.init.normal_(pi.dec[-1].weight, std=.01)
        wm = WorldModel(29, layout.dim, hidden=(32, 32, 32), structured=True, terrain_slots=7)
        torch.nn.init.normal_(wm.net[-1].weight, std=.001)
        critic = Critic(inv_dim(29)+mdp.critic_extra+goal_dim)
        S = (motion.root_p[:B].clone(), motion.root_R[:B].clone(), torch.randn(B,3)*.05,
             torch.randn(B,3)*.05, q0+torch.randn(B,29)*.03, torch.randn(B,29)*.1)
        xi = torch.zeros(B, layout.dim)
        boxes = xi[:, -70:].reshape(B,7,10)
        boxes[...,2], boxes[...,3], boxes[...,7:] = -50, 1, .02
        s = {'s':flatten_state(S), 'xi':xi, 'f':torch.arange(B)+5, 'a_prev':torch.randn(B,29)*.03}
        h = {key:value[:,None].repeat(1,H,*([1]*(value.ndim-1))) for key,value in s.items()}

        names = {'_policy_chunk','policy_update','_imag_step','_reward_term','_critic_input',
                 '_critic_loss','_gae_targets','set_requires_grad','ema_update'}
        tree = ast.parse(Path('blinkmimic/train.py').read_text())
        functions = ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names], type_ignores=[])
        code = compile(functions, 'blinkmimic/train.py', 'exec')
        results = []
        for chunk_size in (0, 2):
            p, w, v = copy.deepcopy(pi), copy.deepcopy(wm), copy.deepcopy(critic)
            opt_pi, opt_v = torch.optim.Adam(p.parameters(),lr=1e-4), torch.optim.Adam(v.parameters(),lr=1e-4)
            avg = torch.optim.swa_utils.AveragedModel(p, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(.995), use_buffers=True)
            ns = dict(torch=torch,nn=torch.nn,mdp=mdp,pi=p,wm=w,critic=v,motion=motion,
                pipe=SimpleNamespace(stateful=False), normalize_xi=layout.normalize,
                flatten_state=flatten_state,unflatten_state=unflatten_state,actor_objective=actor_objective,
                K=K,Hh=H,HZ=HZ,GRAPH=False,dev='cpu',opt_pi=opt_pi,opt_v=opt_v,pi_avg=avg,
                args=SimpleNamespace(pi_batch=B,pi_microbatch=chunk_size,contact_reward=0.,gamma=.99,
                    lam=.95,no_sync=False,pi_clip_value=.1,pi_ema=.995,compile_more=False,
                    fused_adam=False,critic_minibatches=2,critic_epochs=1),
                sample_valid_starts=lambda _: (s,h))
            exec(code,ns)
            ns['imag_step'],ns['reward_term'],ns['critic_input_c'] = ns['_imag_step'],ns['_reward_term'],ns['_critic_input']
            torch.manual_seed(42)
            metrics = ns['policy_update']()
            self.assertEqual(int(avg.n_averaged),1)
            self.assertEqual(int(v.norm.count),(HZ+1)*B)
            self.assertTrue(all(int(state['step'])==1 for state in opt_pi.state.values()))
            self.assertTrue(all(int(state['step'])==2 for state in opt_v.state.values()))
            results.append((p,v,avg,metrics))
        for left,right in zip(results[0][:3],results[1][:3]):
            for key,value in left.state_dict().items():
                torch.testing.assert_close(value,right.state_dict()[key],atol=2e-6,rtol=2e-5,msg=key)
        for left,right in zip(results[0][3],results[1][3]):
            torch.testing.assert_close(left,right,atol=2e-6,rtol=2e-5)


if __name__=='__main__': unittest.main()
