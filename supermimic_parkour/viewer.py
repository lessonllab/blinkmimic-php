"""Ordinary MJLab rendering with a synchronized PHP reference ghost."""

from copy import deepcopy

import torch
from mjlab.viewer.viser.scene import MjlabViserScene


class ComparisonScene:
    def __init__(self, server, env):
        self.server, self.env = server, env
        # Keep renderer-only visibility edits out of the simulation model.
        model = deepcopy(env.sim.mj_model)
        terrain_geoms = [int(g) for entity in env.terrain_entities for g in entity.indexing.geom_ids]
        model.geom_rgba[terrain_geoms, 3] = 0
        server.scene.set_up_direction("+z")
        self.scene = MjlabViserScene(server, model, num_envs=1)
        self.scene.debug_visualization_enabled = True
        self.scene.camera_tracking_enabled = True

        # Match PhpViserPlayViewer: compile-time meshes do not reflect resized
        # terrain. Parent the replacement boxes to the camera-tracked world.
        world = self.scene.fixed_bodies_frame.name
        self.world=world
        self.mesh_mode=getattr(env.motion,'terrain_format','obb')=='vertices'
        self.meshes=[];self.mesh_key=None
        self.boxes = [] if self.mesh_mode else [
            server.scene.add_box(f"{world}/php/box_{i}", color=(90, 150, 200))
            for i in range(env.motion.max_boxes)
        ]

        self.ghost_model = deepcopy(model)
        color = env.command.cfg.viz.ghost_color if hasattr(env, 'command') else (0.2, 0.9, 0.4, 0.5)
        self.ghost_model.geom_rgba[:] = color
        self.ghost_model.geom_rgba[terrain_geoms, 3] = 0
        self.free_q = env.robot.indexing.free_joint_q_adr.cpu().numpy()
        self.joint_q = env.robot.indexing.joint_q_adr.cpu().numpy()

    def reference_qpos(self):
        """Use the exact current reference frame, including its world position."""
        env, motion = self.env, self.env.motion
        frame = int(env.f[0])
        pose = torch.cat(
            (
                motion.root_p[frame] + env.scene.env_origins[0],
                motion.library.body_quat_w[frame, 0],
                motion.q[frame],
            )
        ).cpu().numpy()
        qpos = self.ghost_model.qpos0.copy()
        qpos[self.free_q] = pose[:7]
        qpos[self.joint_q] = pose[7:]
        return qpos

    def sync(self, *, show_reference=True, opacity=0.5):
        env = self.env
        self.scene.clear()
        if show_reference:
            self.scene.add_ghost_mesh(
                self.reference_qpos(), model=self.ghost_model,
                alpha=opacity, label="PHP reference",
            )
        else:
            self.scene.clear_debug_all()
        origin = env.scene.env_origins[0].cpu().numpy()
        count = int(env.motion.box_counts[int(env.motion_ids[0])])
        with self.server.atomic():
            self.scene.update(env.sim.data)
            if self.mesh_mode:
                key=(int(env.motion_ids[0]),int(env.variant_ids[0]))
                if key!=self.mesh_key:
                    for handle in self.meshes: handle.remove()
                    self.meshes=[]
                    vertices=env.xi[0,env.layout.slices['terrain']].reshape(-1,8,3).cpu().numpy()
                    faces=env.motion.library.terrain_faces_np[key[0],key[1]]
                    for i in range(count):
                        self.meshes.append(self.server.scene.add_mesh_simple(f'{self.world}/php/mesh_{i}',
                            vertices=vertices[i]+origin,faces=faces[i],color=(90,150,200),side='double'))
                    self.mesh_key=key
                for handle in self.meshes: handle.visible=self.scene.geom_groups_visible[0]
                return
            boxes = env.xi[0, 33:].reshape(-1, 10).cpu().numpy()
            for i, (box, handle) in enumerate(zip(boxes, self.boxes)):
                handle.visible = i < count and self.scene.geom_groups_visible[0]
                if handle.visible:
                    handle.position = box[:3] + origin
                    handle.wxyz = box[3:7]
                    handle.dimensions = tuple(box[7:])
