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

from PyQt5 import QtWidgets

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
        predicted_pose = utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics, self.plotter.camera.position)
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
        predicted_pose = utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics, self.plotter.camera.position)
        self.output_text.append(f"-> Focal length is {focal_length}: ")
        return predicted_pose

    def epnp_mesh(self):
        if len(self.mesh_store.mesh_actors) == 1: self.mesh_store.reference = list(self.mesh_store.mesh_actors.keys())[0]
        if self.mesh_store.reference:
            colors = utils.get_mesh_actor_scalars(self.mesh_store.mesh_actors[self.mesh_store.reference])
            if colors is not None and (not np.all(colors == colors[0])):
                color_mask = self.export_mesh_render(save_render=False)
                gt_pose = self.mesh_store.mesh_actors[self.mesh_store.reference].user_matrix
                if self.mesh_store.mirror_x: gt_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                if self.mesh_store.mirror_y: gt_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose

                if color_mask is not None and np.sum(color_mask):
                    if self.mesh_store.mesh_colors[self.mesh_store.reference] == 'nocs':
                        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_store.mesh_actors[self.mesh_store.reference])
                        mesh = trimesh.Trimesh(vertices, faces, process=False)
                        predicted_pose = self.nocs_epnp(color_mask, mesh)
                        if self.mesh_store.mirror_x: predicted_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        if self.mesh_store.mirror_y: predicted_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        angular_distance = utils.angler_distance(predicted_pose[:3, :3], gt_pose[:3, :3])
                        translation_error = np.linalg.norm(predicted_pose[:3, 3] - gt_pose[:3, 3])
                        self.output_text.append(f"Predicted pose with <span style='background-color:yellow; color:black;'>NOCS color</span>: ")
                        self.output_text.append(f"{predicted_pose}")
                        self.output_text.append(f"GT Pose: ")
                        self.output_text.append(f"{gt_pose}")
                        self.output_text.append(f"Angular Error (in degree): {angular_distance}")
                        self.output_text.append(f"Translation Error: {translation_error}")
                        self.output_text.append(f"\n************************************************************\n")
                    else:
                        QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Only works using EPnP with latlon mask", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                else:
                    QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "The color mask is blank (maybe set the reference mesh wrong)", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            else:
                QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "The mesh need to be colored, with gradient color", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "A mesh need to be loaded/mesh reference need to be set", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def epnp_mask(self, nocs_method):
        if self.mask_store.mask_actor:
            mask_data = self.mask_store.render_mask(camera=self.plotter.camera.copy())
            if np.max(mask_data) > 1: mask_data = mask_data / 255

            # current shown mask is binary mask
            if np.all(np.logical_or(mask_data == 0, mask_data == 1)):
                if len(self.mesh_store.mesh_actors) == 1: 
                    self.mesh_store.reference = list(self.mesh_store.mesh_actors.keys())[0]
                if self.mesh_store.reference:
                    colors = utils.get_mesh_actor_scalars(self.mesh_store.mesh_actors[self.mesh_store.reference])
                    if colors is not None and (not np.all(colors == colors[0])):
                        color_mask = self.export_mesh_render(save_render=False)
                        nocs_color = (self.mesh_store.mesh_colors[self.mesh_store.reference] == 'nocs')
                        gt_pose = self.mesh_store.mesh_actors[self.mesh_store.reference].user_matrix
                        if self.mesh_store.mirror_x: gt_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                        if self.mesh_store.mirror_y: gt_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_store.mesh_actors[self.mesh_store.reference])
                        mesh = trimesh.Trimesh(vertices, faces, process=False)
                    else:
                        QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "The mesh need to be colored, with gradient color", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                        return 0
                else: 
                    QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "A mesh need to be loaded/mesh reference need to be set", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    return 0
                
                if color_mask is not None: color_mask = (color_mask * mask_data).astype(np.uint8)
                else: 
                    QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Color mask is None", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    return 0
            
            if np.sum(color_mask):
                if nocs_method == nocs_color:
                    if nocs_method: 
                        color_theme = 'NOCS'
                        predicted_pose = self.nocs_epnp(color_mask, mesh)
                        if self.mesh_store.mirror_x: predicted_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        if self.mesh_store.mirror_y: predicted_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                    else: 
                        color_theme = 'LATLON'
                        if self.mesh_store.mirror_x: color_mask = color_mask[:, ::-1, :]
                        if self.mesh_store.mirror_y: color_mask = color_mask[::-1, :, :]
                        predicted_pose = self.latlon_epnp(color_mask, mesh)
                    angular_distance = utils.angler_distance(predicted_pose[:3, :3], gt_pose[:3, :3])
                    translation_error = np.linalg.norm(predicted_pose[:3, 3] - gt_pose[:3, 3])
                    self.output_text.append(f"Predicted pose with <span style='background-color:yellow; color:black;'>{color_theme} color (masked)</span>: ")
                    self.output_text.append(f"{predicted_pose}")
                    self.output_text.append(f"GT Pose: ")
                    self.output_text.append(f"{gt_pose}")
                    self.output_text.append(f"Angular Error (in degree): {angular_distance}")
                    self.output_text.append(f"Translation Error: {translation_error}")
                    self.output_text.append(f"\n************************************************************\n")
                else:
                    QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(),"vision6D", "Clicked the wrong method")
            else:
                QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "The color mask is blank (maybe set the reference mesh wrong)", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(),"vision6D", "please load a mask first")
