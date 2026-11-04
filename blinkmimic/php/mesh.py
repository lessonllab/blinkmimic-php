"""Exact surface geometry for PHP's star-shaped eight-vertex obstacles.

MuJoCo collides convex meshes. Partition each closed obstacle into twelve
tetrahedra, one per source triangle, rather than filling its concavities with
one convex hull or approximating the source with an oriented box.
"""
from collections import Counter
import hashlib

import mujoco
import numpy as np
import torch

TET_FACES = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32)
DUMMY = np.array([[0,0,0],[.02,0,0],[0,.02,0],[0,0,.02]],np.float32)


def partition(vertices, faces):
    center = vertices.mean(0)
    tri = vertices[faces]
    signed = np.einsum('fi,fi->f',np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),tri[:,0]-center)
    edges = Counter(tuple(sorted((int(a),int(b)))) for f in faces for a,b in zip(f,np.roll(f,-1)))
    if len(edges)!=18 or any(n!=2 for n in edges.values()) or not (np.all(signed>1e-10) or np.all(signed < -1e-10)):
        raise ValueError('Terrain is not a closed star-shaped obstacle; cannot preserve its surface with tetrahedra')
    return center, np.concatenate((np.broadcast_to(center,(12,1,3)),tri),axis=1)-center


def ray_coefficients(vertices, faces):
    """Affine barycentric coordinates and z on each source triangle."""
    tri = vertices[faces]
    edge = (tri[:,1:,:2]-tri[:,0,None,:2]).transpose(0,2,1)
    det = np.linalg.det(edge)
    valid = np.abs(det)>1e-10
    inv = np.zeros_like(edge)
    inv[valid] = np.linalg.inv(edge[valid])
    uv = np.concatenate((inv,-np.einsum('fij,fj->fi',inv,tri[:,0,:2])[...,None]),-1)
    dz = tri[:,1:,2]-tri[:,0,None,2]
    z = np.einsum('fi,fij->fj',dz,uv)
    z[:,2] += tri[:,0,2]
    return np.concatenate((uv[:,0],uv[:,1],z),-1).astype(np.float32),valid


def support_height(points, coefficients, valid):
    # points (..., feet, 3); coefficients (..., slots, triangles, 9)
    xy1 = torch.cat((points[...,:2],torch.ones_like(points[...,2:3])),-1)
    weights = (xy1[..., :,None,None,None,:] * coefficients[...,None,:,:,:].unflatten(-1,(3,3))).sum(-1)
    u,v,z = weights.unbind(-1)
    hit = (u>=-1e-5)&(v>=-1e-5)&(u+v<=1+1e-5)&valid[...,None,:,:]
    return torch.where(hit,z,0.).amax(dim=(-1,-2)).clamp(min=0.)


class MeshBank:
    def __init__(self, motion):
        self.vertices = motion.library.terrain_vertices_np
        self.faces = motion.library.terrain_faces_np
        C,V,S = self.vertices.shape[:3]
        self.centers = self.vertices.mean(-2).astype(np.float32)
        self.keys = np.empty((C,V,S,12),object)
        self.assets = [dict(dummy=DUMMY) for _ in range(S)]
        self.coefficients = np.zeros((C,V,S,12,9),np.float32)
        self.valid = np.zeros((C,V,S,12),bool)
        for c in range(C):
            for v in range(V):
                for slot in range(S):
                    if v>=int(motion.variant_counts[c]) or slot>=int(motion.box_counts[c]):
                        self.keys[c,v,slot] = 'dummy'
                        self.centers[c,v,slot] = (0,0,-50)
                        continue
                    vertices,faces = self.vertices[c,v,slot],self.faces[c,v,slot]
                    center,tets = partition(vertices.astype('float64'),faces)
                    self.centers[c,v,slot] = center
                    coeff,valid = ray_coefficients(vertices.astype('float64'),faces)
                    self.coefficients[c,v,slot],self.valid[c,v,slot] = coeff,valid
                    for part,tet in enumerate(tets.astype(np.float32)):
                        key = 'm'+hashlib.sha256(tet.tobytes()).hexdigest()[:24]
                        self.keys[c,v,slot,part] = key
                        self.assets[slot][key] = tet
        self.num_variants = V
        self.num_assets = sum(map(len,self.assets))

    def spec(self, slot):
        spec = mujoco.MjSpec()
        for name,vertices in self.assets[slot].items():
            faces=TET_FACES if np.linalg.det(vertices[1:]-vertices[0])>0 else TET_FACES[:,::-1]
            spec.add_mesh(name=name,uservert=vertices.flatten(),userface=faces.flatten())
        body = spec.worldbody.add_body(name='box',mocap=True,pos=(0,0,-50))
        for part in range(12):
            body.add_geom(name=f'part_{part}',type=mujoco.mjtGeom.mjGEOM_MESH,meshname='dummy',
                          contype=2,conaffinity=0,condim=3,group=0,rgba=(.35,.59,.78,1),
                          friction=(.9,.005,.0001),density=0.)
        return spec

    def bind(self, env):
        model,dev = env.sim.mj_model,env.device
        C,V,S = self.keys.shape[:3]
        dataids = np.empty((C,V,S,12),np.int32)
        for slot in range(S):
            ids = {name:model.mesh(f'php_box_{slot}/{name}').id for name in self.assets[slot]}
            for c in range(C):
                for v in range(V):
                    dataids[c,v,slot] = [ids[k] for k in self.keys[c,v,slot]]
        size = np.zeros((model.nmesh,3),np.float32)
        aabb = np.zeros((model.nmesh,2,3),np.float32)
        rbound = np.zeros(model.nmesh,np.float32)
        for mesh in np.unique(dataids):
            start,count = model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
            verts = model.mesh_vert[start:start+count]
            lo,hi = verts.min(0),verts.max(0)
            aabb[mesh,0],aabb[mesh,1] = (lo+hi)*.5,(hi-lo)*.5
            size[mesh] = np.maximum(abs(lo),abs(hi))
            rbound[mesh] = np.linalg.norm(verts,axis=-1).max()
        env.terrain_mesh_bound = {k:torch.as_tensor(v,device=dev) for k,v in
            dict(dataids=dataids,centers=self.centers,size=size,aabb=aabb,rbound=rbound,
                 pos=model.mesh_pos.astype(np.float32),quat=model.mesh_quat.astype(np.float32)).items()}
        env.terrain_mesh_gids=torch.cat([e.indexing.geom_ids for e in env.terrain_entities])
        env.terrain_mesh_mids=torch.tensor(env.terrain_mids,device=dev)

    def place(self, env, mask, clips, variants):
        bank=env.terrain_mesh_bound
        centers,ids = bank['centers'][clips,variants],bank['dataids'][clips,variants].long()
        gids,mids=env.terrain_mesh_gids,env.terrain_mesh_mids
        mesh=ids.flatten(1)
        current=env.sim.model.geom_dataid[:,gids]
        env.sim.model.geom_dataid[:,gids]=torch.where(mask[:,None],mesh.to(current.dtype),current)
        for name,key in [('geom_pos','pos'),('geom_quat','quat'),('geom_size','size'),('geom_aabb','aabb'),('geom_rbound','rbound')]:
            field=getattr(env.sim.model,name)
            value=bank[key][mesh]
            field[:,gids]=torch.where(mask.view(-1,*([1]*(value.ndim-1))),value,field[:,gids])
        current=env.sim.data.mocap_pos[:,mids]
        env.sim.data.mocap_pos[:,mids]=torch.where(mask[:,None,None],centers+env.scene.env_origins[:,None],current)
        current=env.sim.data.mocap_quat[:,mids]
        quat=torch.zeros_like(current);quat[...,0]=1.
        env.sim.data.mocap_quat[:,mids]=torch.where(mask[:,None,None],quat,current)
