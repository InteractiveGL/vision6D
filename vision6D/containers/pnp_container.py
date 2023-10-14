'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: pnp_container.py
@time: 2023-07-03 20:27
@desc: create container for PnP related algorithms in application
'''

import math

import trimesh
import numpy as np
import matplotlib.pyplot as plt

from ..tools import utils
from ..components import CameraStore
from ..components import MaskStore
from ..components import MeshStore

class PnPContainer:
    def __init__(self, plotter, export_mesh_render, output_text):
        self.plotter = plotter
        self.export_mesh_render = export_mesh_render
        self.output_text = output_text
        
        self.camera_store = CameraStore()
        self.mask_store = MaskStore()
        self.mesh_store = MeshStore()

    def nocs_epnp(self, color_mask, mesh):
        vertices = mesh.vertices
        pts3d, pts2d = utils.create_2d_3d_pairs(color_mask, vertices)
        pts2d = pts2d.astype('float32')
        pts3d = pts3d.astype('float32')
        camera_intrinsics = self.camera_store.camera_intrinsics.astype('float32')
        focal_length = (1080 / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
        camera_intrinsics[0, 0] = focal_length
        camera_intrinsics[1, 1] = focal_length
        predicted_pose = utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics)
        self.output_text.append(f"-> Focal length is {focal_length}: ")
        return predicted_pose

    def latlon_epnp(self, color_mask, mesh):
        binary_mask = utils.color2binary_mask(color_mask)
        idx = np.where(binary_mask == 1)
        # swap the points for opencv, maybe because they handle RGB image differently (RGB -> BGR in opencv)
        idx = idx[:2][::-1]
        pts2d = np.stack((idx[0], idx[1]), axis=1)
        pts3d = []
        
        # Obtain the rg color
        color = color_mask[pts2d[:,1], pts2d[:,0]][..., :2]
        if np.max(color) > 1: color = color / 255
        gx = color[:, 0]
        gy = color[:, 1]

        lat = np.array(self.mesh_store.latlon[..., 0])
        lon = np.array(self.mesh_store.latlon[..., 1])
        lonf = lon[mesh.faces]
        msk = (np.sum(lonf>=0, axis=1)==3) & (np.sum(lat[mesh.faces]>=0, axis=1)==3)
        for i in range(len(pts2d)):
            pt = utils.latLon2xyz(mesh, lat, lonf, msk, gx[i], gy[i])
            pts3d.append(pt)
       
        pts3d = np.array(pts3d).reshape((len(pts3d), 3))

        pts2d = pts2d.astype('float32')
        pts3d = pts3d.astype('float32')
        camera_intrinsics = self.camera_store.camera_intrinsics.astype('float32')
        focal_length = (1080 / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
        camera_intrinsics[0, 0] = focal_length
        camera_intrinsics[1, 1] = focal_length
        predicted_pose = utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics)
        self.output_text.append(f"-> Focal length is {focal_length}: ")
        return predicted_pose

    def epnp_mesh(self):
        if self.mesh_store.reference:
            mesh_data = self.mesh_store.meshes[self.mesh_store.reference]
            colors = utils.get_mesh_actor_scalars(mesh_data.actor)
            if colors is not None and (not np.all(colors == colors[0])):
                color_mask = self.export_mesh_render(save_render=False)
                gt_pose = mesh_data.actor.user_matrix
                if mesh_data.mirror_x: gt_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                if mesh_data.mirror_y: gt_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose

                if color_mask is not None and np.sum(color_mask):
                    if mesh_data.color == 'nocs':
                        vertices, faces = utils.get_mesh_actor_vertices_faces(mesh_data.actor)
                        mesh = trimesh.Trimesh(vertices, faces, process=False)
                        predicted_pose = self.nocs_epnp(color_mask, mesh)
                        if mesh_data.mirror_x: predicted_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        if mesh_data.mirror_y: predicted_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        angular_distance = utils.angler_distance(predicted_pose[:3, :3], gt_pose[:3, :3])
                        translation_error = np.linalg.norm(predicted_pose[:3, 3] - gt_pose[:3, 3])
                        self.output_text.append(f"Predicted pose with NOCS color: ")
                        predicted_pose_text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                        predicted_pose[0, 0], predicted_pose[0, 1], predicted_pose[0, 2], predicted_pose[0, 3], 
                        predicted_pose[1, 0], predicted_pose[1, 1], predicted_pose[1, 2], predicted_pose[1, 3], 
                        predicted_pose[2, 0], predicted_pose[2, 1], predicted_pose[2, 2], predicted_pose[2, 3],
                        predicted_pose[3, 0], predicted_pose[3, 1], predicted_pose[3, 2], predicted_pose[3, 3])
                        self.output_text.append(predicted_pose_text)
                        self.output_text.append(f"GT Pose: ")
                        gt_pose_text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                        gt_pose[0, 0], gt_pose[0, 1], gt_pose[0, 2], gt_pose[0, 3], 
                        gt_pose[1, 0], gt_pose[1, 1], gt_pose[1, 2], gt_pose[1, 3], 
                        gt_pose[2, 0], gt_pose[2, 1], gt_pose[2, 2], gt_pose[2, 3],
                        gt_pose[3, 0], gt_pose[3, 1], gt_pose[3, 2], gt_pose[3, 3])
                        self.output_text.append(gt_pose_text)
                        self.output_text.append(f"Angular Error (in degree): {angular_distance}")
                        self.output_text.append(f"Translation Error: {translation_error}\n")
                    else: utils.display_warning("Only works using EPnP with latlon mask")
                else: utils.display_warning("The color mask is blank (maybe set the reference mesh wrong)")
            else: utils.display_warning("The mesh need to be colored, with gradient color")
        else: utils.display_warning("A mesh need to be loaded/mesh reference need to be set")

    def epnp_mask_handle_binary_mask(self, mask_data):
        if self.mesh_store.reference:
            mesh_data = self.mesh_store.meshes[self.mesh_store.reference]
            colors = utils.get_mesh_actor_scalars(mesh_data.actor)
            if colors is not None and (not np.all(colors == colors[0])):
                color_mask = self.export_mesh_render(save_render=False)
                nocs_color = (mesh_data.color == 'nocs')
                gt_pose = mesh_data.actor.user_matrix
                if mesh_data.mirror_x: gt_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                if mesh_data.mirror_y: gt_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                vertices, faces = utils.get_mesh_actor_vertices_faces(mesh_data.actor)
                mesh = trimesh.Trimesh(vertices, faces, process=False)
            else: utils.display_warning("The mesh need to be colored, with gradient color")
        else: utils.display_warning("A mesh need to be loaded/mesh reference need to be set")
        
        if color_mask is not None: color_mask = (color_mask * mask_data).astype(np.uint8)
        else: utils.display_warning("Color mask is None")
        
        return mesh_data, color_mask, nocs_color, gt_pose, mesh
    
    def epnp_mask_nocs_theme(self, mesh_data, color_mask, mesh):
        color_theme = 'NOCS'
        predicted_pose = self.nocs_epnp(color_mask, mesh)
        if mesh_data.mirror_x: predicted_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        if mesh_data.mirror_y: predicted_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return color_theme, predicted_pose
    
    def epnp_mask_latlon_theme(self, mesh_data, color_mask, mesh):
        color_theme = 'LATLON'
        if mesh_data.mirror_x: color_mask = color_mask[:, ::-1, :]
        if mesh_data.mirror_y: color_mask = color_mask[::-1, :, :]
        predicted_pose = self.latlon_epnp(color_mask, mesh)
        return color_theme, predicted_pose
    
    def epnp_mask(self, nocs_method):
        if self.mask_store.mask_actor:
            mask_data = self.mask_store.render_mask(camera=self.plotter.camera.copy())
            if np.max(mask_data) > 1: mask_data = mask_data / 255
            if np.all(np.logical_or(mask_data == 0, mask_data == 1)):
                mesh_data, color_mask, nocs_color, gt_pose, mesh = self.epnp_mask_handle_binary_mask(mask_data)
            if np.sum(color_mask):
                if nocs_method == nocs_color:
                    if nocs_method: color_theme, predicted_pose = self.epnp_mask_nocs_theme(mesh_data, color_mask, mesh)
                    else: color_theme, predicted_pose = self.epnp_mask_latlon_theme(mesh_data, color_mask, mesh)
                    angular_distance = utils.angler_distance(predicted_pose[:3, :3], gt_pose[:3, :3])
                    translation_error = np.linalg.norm(predicted_pose[:3, 3] - gt_pose[:3, 3])
                    self.output_text.append(f"Predicted pose with {color_theme} color (masked): ")
                    predicted_pose_text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                    predicted_pose[0, 0], predicted_pose[0, 1], predicted_pose[0, 2], predicted_pose[0, 3], 
                    predicted_pose[1, 0], predicted_pose[1, 1], predicted_pose[1, 2], predicted_pose[1, 3], 
                    predicted_pose[2, 0], predicted_pose[2, 1], predicted_pose[2, 2], predicted_pose[2, 3],
                    predicted_pose[3, 0], predicted_pose[3, 1], predicted_pose[3, 2], predicted_pose[3, 3])
                    self.output_text.append(predicted_pose_text)
                    self.output_text.append(f"GT Pose: ")
                    gt_pose_text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                    gt_pose[0, 0], gt_pose[0, 1], gt_pose[0, 2], gt_pose[0, 3], 
                    gt_pose[1, 0], gt_pose[1, 1], gt_pose[1, 2], gt_pose[1, 3], 
                    gt_pose[2, 0], gt_pose[2, 1], gt_pose[2, 2], gt_pose[2, 3],
                    gt_pose[3, 0], gt_pose[3, 1], gt_pose[3, 2], gt_pose[3, 3])
                    self.output_text.append(gt_pose_text)
                    self.output_text.append(f"Angular Error (in degree): {angular_distance}")
                    self.output_text.append(f"Translation Error: {translation_error}\n")
                else: utils.display_warning("Clicked the wrong method")
            else: utils.display_warning("The color mask is blank (maybe set the reference mesh wrong)")
        else: utils.display_warning("please load a mask first")
