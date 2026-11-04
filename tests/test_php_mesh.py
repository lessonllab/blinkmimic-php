import unittest
from dataclasses import replace
import numpy as np
import torch
import mujoco
from blinkmimic.php.mesh import partition,ray_coefficients,support_height,TET_FACES
from blinkmimic.php.adapter import context_features
from blinkmimic.robot import get_robot
from blinkmimic.dr import XiLayout
from blinkmimic.mirror import Mirror


def dented_box():
    v=np.array([[-1,-1,0],[-1,-1,1],[-1,1,0],[-1,1,1],[1,-1,0],[1,-1,1],[1,1,0],[1,1,.5]],float)
    f=np.array([[0,6,4],[0,2,6],[1,5,7],[1,7,3],[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6]],int)
    return v,f


class MeshTerrainTests(unittest.TestCase):
    def test_decomposition_keeps_a_nonconvex_source_surface(self):
        v,f=dented_box();center,tets=partition(v,f)
        spec=mujoco.MjSpec()
        for i,tet in enumerate(tets):
            faces=TET_FACES if np.linalg.det(tet[1:]-tet[0])>0 else TET_FACES[:,::-1]
            spec.add_mesh(name=f'm{i}',uservert=tet.flatten(),userface=faces.flatten())
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,meshname=f'm{i}',pos=center)
        model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        hit=np.array([-1],np.int32)
        distance=mujoco.mj_ray(model,data,np.array([0.,0.,10.]),np.array([0.,0.,-1.]),None,1,-1,hit)
        self.assertAlmostEqual(10-distance,.75,places=6)
        coeff,valid=ray_coefficients(v,f)
        points=torch.tensor([[[0.,0.,1.],[3.,0.,1.]]],requires_grad=True)
        heights=support_height(points,torch.tensor(coeff[None,None]),torch.tensor(valid[None,None]))
        torch.testing.assert_close(heights,torch.tensor([[.75,0.]]))
        heights.sum().backward();self.assertTrue(bool(torch.isfinite(points.grad).all()))

    def test_mesh_context_and_mirror_keep_geometry_id_out_of_network_features(self):
        robot=replace(get_robot('php'),terrain_slots=1,terrain_format='vertices')
        layout=XiLayout(robot,'cpu');mirror=Mirror(robot,layout.slices,'cpu')
        xi=torch.zeros(2,layout.dim);v,_=dented_box();xi[:,33:-1]=torch.tensor(v,dtype=torch.float32).flatten()
        xi[:,-1]=torch.tensor([1.,2.])
        S=(torch.zeros(2,3),torch.eye(3).repeat(2,1,1),torch.zeros(2,3),torch.zeros(2,3),torch.zeros(2,29),torch.zeros(2,29))
        expected=context_features(S,xi,1,'vertices');other=xi.clone();other[:,-1]+=1000
        torch.testing.assert_close(context_features(S,other,1,'vertices'),expected)
        torch.testing.assert_close(mirror.xi(mirror.xi(xi)),xi)
        delta=torch.tensor([2.,-3.,0.]);other[:,33:-1]= (other[:,33:-1].reshape(2,1,8,3)+delta).flatten(1)
        torch.testing.assert_close(context_features((S[0]+delta,*S[1:]),other,1,'vertices'),expected)


if __name__=='__main__':unittest.main()
