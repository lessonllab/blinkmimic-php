import re
import shlex
import unittest
from pathlib import Path

import torch

from blinkmimic.php.recipe import UPSTREAM_MAIN
from blinkmimic.php.adapter import support_height, context_features
from blinkmimic.robot import get_robot
from blinkmimic.mirror import Mirror
from blinkmimic.dr import XiLayout
from blinkmimic.networks import Policy, WorldModel, policy_kwargs
from blinkmimic.rotations import quat_to_mat, mat_to_quat


class BlinkPhpTests(unittest.TestCase):
    def test_physical_obstacle_can_contact_php_hand(self):
        import mujoco
        from mjlab.scene import Scene
        from blinkmimic.env import make_scene_cfg
        scene = Scene(make_scene_cfg(get_robot('php'),1),device='cpu')
        model = scene.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        hand, box = model.geom('robot/left_hand_collision').id, model.geom('php_box_0/box').id
        mid = model.body_mocapid[model.geom_bodyid[box]]
        model.geom_size[box] = [.05,.05,.05]
        data.mocap_pos[mid] = data.geom_xpos[hand] + [0.,0.,-.06]
        mujoco.mj_forward(model,data)
        pairs = {frozenset(c.geom) for c in data.contact}
        self.assertIn(frozenset((hand,box)),pairs)

    def test_recipe_is_the_pinned_upstream_main(self):
        text = Path('scripts/blinkmimic-presets.sh').read_text()
        source = re.search(r'\[main\]="([^"]+)"', text).group(1)
        self.assertEqual(UPSTREAM_MAIN, shlex.split(source))

    def test_contact_height_tracks_platform_and_ground(self):
        boxes = torch.tensor([[[0.,0.,.5,1.,0.,0.,0.,2.,2.,1.],
                               [0.,0.,-50.,1.,0.,0.,0.,.02,.02,.02]]])
        points = torch.tensor([[[0.,0.,1.06],[3.,0.,.06]]])
        torch.testing.assert_close(support_height(points, boxes), torch.tensor([[1.,0.]]))

    def test_terrain_mirror_is_involution(self):
        robot=get_robot('php');layout=XiLayout(robot,'cpu');mirror=Mirror(robot,layout.slices,'cpu')
        xi=torch.randn(4,layout.dim);boxes=xi[:,-70:].reshape(4,7,10)
        boxes[...,3:7]=torch.nn.functional.normalize(boxes[...,3:7],dim=-1)
        torch.testing.assert_close(mirror.xi(mirror.xi(xi)),xi)
        S=(torch.randn(4,3),quat_to_mat(torch.nn.functional.normalize(torch.randn(4,4),dim=-1)),
           torch.randn(4,3),torch.randn(4,3),torch.randn(4,29),torch.randn(4,29))
        for expected,actual in zip(S,mirror.state(mirror.state(S))):
            torch.testing.assert_close(actual,expected)

    def test_terrain_context_is_translation_invariant(self):
        xi=torch.zeros(2,103);boxes=xi[:,-70:].reshape(2,7,10)
        boxes[...,3]=1.;boxes[...,7:]=.3;boxes[...,2]=.5
        S=(torch.randn(2,3),torch.eye(3).repeat(2,1,1),torch.zeros(2,3),torch.zeros(2,3),torch.zeros(2,29),torch.zeros(2,29))
        expected=context_features(S,xi,7)
        delta=torch.tensor([2.,-3.,0.]);other=xi.clone();other[:,-70:].reshape(2,7,10)[...,:3]+=delta
        actual=context_features((S[0]+delta,*S[1:]),other,7)
        torch.testing.assert_close(actual,expected)

    def test_reference_residual_is_exact_with_zero_decoder(self):
        robot=get_robot('php')
        policy=Policy(99,730,29,enc_hidden=(16,),dec_hidden=(16,),ref_skip=(15,robot.default_joint_pos_list(),robot.action_scale_list()))
        torch.nn.init.zeros_(policy.dec[-1].weight);torch.nn.init.zeros_(policy.dec[-1].bias)
        goal=torch.zeros(3,730);target=torch.randn(3,29);goal[:,15:44]=target
        action=policy(torch.zeros(3,10,99),goal)
        q0=torch.tensor(robot.default_joint_pos_list());alpha=torch.tensor(robot.action_scale_list())
        torch.testing.assert_close(q0+alpha*action,target)

    def test_structured_model_and_terrain_input_have_finite_gradients(self):
        wm=WorldModel(29,103,hidden=(32,32),structured=True,terrain_slots=7)
        S=(torch.zeros(3,3),torch.eye(3).repeat(3,1,1),torch.ones(3,3),torch.zeros(3,3),torch.zeros(3,29),torch.zeros(3,29))
        xi=torch.zeros(3,103);xi[:,-70:].reshape(3,7,10)[...,3]=1.
        action=torch.zeros(3,29,requires_grad=True)
        out=wm.step(S,action,action,xi)
        self.assertTrue(all(bool(torch.isfinite(v).all()) for v in out))
        sum(v.square().sum() for v in out).backward()
        self.assertTrue(bool(torch.isfinite(action.grad).all()))


if __name__=='__main__': unittest.main()
