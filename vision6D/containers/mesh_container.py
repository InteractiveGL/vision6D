'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: mesh_container.py
@time: 2023-07-03 20:27
@desc: create container for mesh related actions in application
'''

import ast
import copy
import pathlib

import trimesh
import PIL.Image
import numpy as np
import pyvista as pv

from PyQt5 import QtWidgets

from ..tools import utils
from ..components import CameraStore
from ..components import MaskStore
from ..components import MeshStore
from ..widgets import GetTextDialog

class MeshContainer:
    def __init__(self, 
                color_button, 
                plotter, 
                hintLabel, 
                track_actors_names, 
                add_button_actor_name, 
                button_group_actors_names,
                check_button,
                opacity_spinbox, 
                opacity_value_change,
                reset_camera,
                toggle_register,
                load_mask,
                output_text):
        
        self.ignore_opacity_change = False
        self.toggle_hide_meshes_flag = False

        self.color_button = color_button
        self.plotter = plotter
        self.hintLabel = hintLabel
        self.track_actors_names = track_actors_names
        self.add_button_actor_name = add_button_actor_name
        self.button_group_actors_names = button_group_actors_names
        self.check_button = check_button
        self.opacity_spinbox = opacity_spinbox
        self.opacity_value_change = opacity_value_change
        self.reset_camera = reset_camera
        self.toggle_register = toggle_register
        self.load_mask = load_mask
        self.output_text = output_text
        
        self.camera_store = CameraStore()
        self.mask_store = MaskStore()
        self.mesh_store = MeshStore()

    def add_mesh_file(self, mesh_path='', prompt=False):
        if prompt: 
            mesh_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.mesh *.ply *.stl *.obj *.off *.dae *.fbx *.3ds *.x3d)") 
        if mesh_path:
            self.hintLabel.hide()
            mesh_data = self.mesh_store.add_mesh(mesh_source=mesh_path)
            if mesh_data: self.add_mesh(mesh_data, np.eye(4))
            else: QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "The mesh format is not supported!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def mirror_mesh(self, name, direction):
        if self.mesh_store.toggle_anchor_mesh: name = self.mesh_store.reference
        if direction == 'x': self.mesh_store.meshes[name].mirror_x = not self.mesh_store.meshes[name].mirror_x
        elif direction == 'y': self.mesh_store.meshes[name].mirror_y = not self.mesh_store.meshes[name].mirror_y
        if self.mesh_store.meshes[name].initial_pose is None: self.mesh_store.meshes[name].initial_pose = self.mesh_store.meshes[name].actor.user_matrix
        transformation_matrix = self.mesh_store.meshes[name].initial_pose
        if self.mesh_store.meshes[name].mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        if self.mesh_store.meshes[name].mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        self.mesh_store.meshes[name].actor.user_matrix = transformation_matrix
        self.check_button(actor_name=name, output_text=False) 
        self.output_text.append(f"-> Mirrored transformation matrix is: \n{transformation_matrix}")

    def add_mesh(self, mesh_data, transformation_matrix):
        """ add a mesh to the pyqt frame """
        mesh = self.plotter.add_mesh(mesh_data.pv_mesh, color=mesh_data.color, opacity=mesh_data.opacity, name=mesh_data.name)
        mesh.user_matrix = transformation_matrix
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=mesh_data.name)
        mesh_data.actor = actor
        self.color_button.setText(mesh_data.color)

        # add remove current mesh to removeMenu
        if mesh_data.name not in self.track_actors_names:
            self.track_actors_names.append(mesh_data.name)
            self.add_button_actor_name(mesh_data.name)
        #* very important for mirroring
        self.check_button(actor_name=mesh_data.name, output_text=False) 
        
    def anchor_mesh(self):
        self.mesh_store.toggle_anchor_mesh = not self.mesh_store.toggle_anchor_mesh
                
    def set_spacing(self):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button:
            actor_name = checked_button.text()
            if actor_name in self.mesh_store.meshes:
                spacing, ok = QtWidgets.QInputDialog().getText(QtWidgets.QMainWindow(), 'Input', "Set Spacing", text=str(self.mesh_store.meshes[actor_name].spacing))
                if ok:
                    try: self.mesh_store.meshes[actor_name].spacing = ast.literal_eval(spacing)
                    except: QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    vertices = self.mesh_store.meshes[actor_name].source_mesh.vertices * self.mesh_store.meshes[actor_name].spacing
                    self.mesh_store.meshes[actor_name].pv_mesh.points = vertices
            else:
                QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Need to select a mesh object instead", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Need to select a mesh actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
      
    def set_color(self, color, actor_name):
        if color in self.mesh_store.colors:
            actor = self.plotter.add_mesh(self.mesh_store.meshes[actor_name].pv_mesh, color=color, opacity=self.mesh_store.meshes[actor_name].opacity, name=actor_name)
        else: 
            scalars = utils.color_mesh(self.mesh_store.meshes[actor_name].pv_mesh.points, color=color)
            try: actor = self.plotter.add_mesh(self.mesh_store.meshes[actor_name].pv_mesh, scalars=scalars, rgb=True, opacity=self.mesh_store.meshes[actor_name].opacity, name=actor_name)
            except ValueError: 
                QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Cannot set the selected color", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return 0
        actor.user_matrix = self.mesh_store.meshes[actor_name].actor.user_matrix
        self.mesh_store.meshes[actor_name].actor = actor
                    
    def set_mesh_opacity(self, name: str, surface_opacity: float):
        self.mesh_store.meshes[name].opacity = surface_opacity
        self.mesh_store.meshes[name].actor.user_matrix = pv.array_from_vtkmatrix(self.mesh_store.meshes[name].actor.GetMatrix())
        self.mesh_store.meshes[name].actor.GetProperty().opacity = surface_opacity

    def toggle_surface_opacity(self, up):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button:
            if checked_button.text() in self.mesh_store.meshes: 
                change = 0.05
                if not up: change *= -1
                current_opacity = self.opacity_spinbox.value()
                current_opacity += change
                current_opacity = np.clip(current_opacity, 0, 1)
                self.opacity_spinbox.setValue(current_opacity)
            
    def toggle_hide_meshes_button(self):
        self.toggle_hide_meshes_flag = not self.toggle_hide_meshes_flag
        if self.toggle_hide_meshes_flag:

            checked_button = self.button_group_actors_names.checkedButton()
            if checked_button: self.checked_button_name = checked_button.text()
            else: self.checked_button_name = None

            for button in self.button_group_actors_names.buttons():
                actor_name = button.text()
                if actor_name in self.mesh_store.meshes:
                    if len(self.mesh_store.meshes) != 1 and actor_name == self.checked_button_name: 
                        continue
                    self.ignore_opacity_change = True
                    self.opacity_spinbox.setValue(0)
                    self.ignore_opacity_change = False
                    self.mesh_store.meshes[actor_name].previous_opacity = copy.deepcopy(self.mesh_store.meshes[actor_name].opacity)
                    self.mesh_store.meshes[actor_name].opacity = 0
                    self.set_mesh_opacity(actor_name, self.mesh_store.meshes[actor_name].opacity)
        else:
            for button in self.button_group_actors_names.buttons():
                actor_name = button.text()
                if actor_name in self.mesh_store.meshes:
                    if len(self.mesh_store.meshes) != 1 and actor_name == self.checked_button_name: 
                        continue
                    self.ignore_opacity_change = True
                    self.opacity_spinbox.setValue(self.mesh_store.meshes[actor_name].previous_opacity)
                    self.ignore_opacity_change = False
                    self.set_mesh_opacity(actor_name, self.mesh_store.meshes[actor_name].previous_opacity)
                    self.mesh_store.meshes[actor_name].previous_opacity = copy.deepcopy(self.mesh_store.meshes[actor_name].opacity)
                            
    def add_pose_file(self, pose_path):
        if pose_path:
            self.hintLabel.hide()
            transformation_matrix = np.load(pose_path)
            if self.mesh_store.meshes[self.mesh_store.reference].mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            if self.mesh_store.meshes[self.mesh_store.reference].mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            self.add_pose(matrix=transformation_matrix)

    def add_pose(self, matrix:np.ndarray=None, rot:np.ndarray=None, trans:np.ndarray=None):
        if matrix is None and (rot is not None and trans is not None): matrix = np.vstack((np.hstack((rot, trans)), [0, 0, 0, 1]))
        if self.mesh_store.toggle_anchor_mesh:
            for mesh_data in self.mesh_store.meshes.values(): mesh_data.initial_pose = matrix
        else: self.mesh_store.meshes[self.mesh_store.reference].initial_pose = matrix
        self.reset_gt_pose()
        
    def set_pose(self):
        # get the gt pose
        get_text_dialog = GetTextDialog()
        res = get_text_dialog.exec_()
        if res == QtWidgets.QDialog.Accepted:
            try:
                if "," not in get_text_dialog.user_text:
                    get_text_dialog.user_text = get_text_dialog.user_text.replace(" ", ",")
                    get_text_dialog.user_text = get_text_dialog.user_text.strip().replace("[,", "[")
                gt_pose = ast.literal_eval(get_text_dialog.user_text)
                gt_pose = np.array(gt_pose)
                if gt_pose.shape == (4, 4):
                    self.hintLabel.hide()
                    transformation_matrix = gt_pose
                    if self.mesh_store.meshes[self.mesh_store.reference].mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                    if self.mesh_store.meshes[self.mesh_store.reference].mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                    self.add_pose(matrix=transformation_matrix)
                else:
                    QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "It needs to be a 4 by 4 matrix", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok) 
            except: 
                QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
    # todo: fix the reset gt_pose for not anchored situation
    def reset_gt_pose(self):
        if self.mesh_store.reference:
            if self.mesh_store.meshes[self.mesh_store.reference].initial_pose is not None:
                self.output_text.append(f"-> Reset the GT pose to: \n{self.mesh_store.meshes[self.mesh_store.reference].initial_pose}")
                self.toggle_register(self.mesh_store.meshes[self.mesh_store.reference].initial_pose)
                self.reset_camera()
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Need to set a reference mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def update_gt_pose(self):
        if self.mesh_store.reference:
            if self.mesh_store.meshes[self.mesh_store.reference].initial_pose is not None:
                self.mesh_store.meshes[self.mesh_store.reference].initial_pose = self.mesh_store.meshes[self.mesh_store.reference].actor.user_matrix
                self.mesh_store.reference_pose()
                self.toggle_register(self.mesh_store.meshes[self.mesh_store.reference].actor.user_matrix)
                self.output_text.append(f"-> Update the {self.mesh_store.reference} GT pose to: \n{self.mesh_store.meshes[self.mesh_store.reference].initial_pose}")
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Need to set a reference mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def undo_pose(self):
        if self.button_group_actors_names.checkedButton():
            actor_name = self.button_group_actors_names.checkedButton().text()
            if actor_name in self.mesh_store.meshes:
                if self.mesh_store.meshes[actor_name].undo_poses and len(self.mesh_store.meshes[actor_name].undo_poses) != 0: 
                    self.mesh_store.undo_pose(actor_name)
                    # register the rest meshes' pose to current undoed pose
                    self.check_button(actor_name=actor_name)
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Choose a mesh actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def export_pose(self):
        if self.mesh_store.reference:
            original_vertices = self.mesh_store.meshes[self.mesh_store.reference].source_mesh.vertices
            verts, faces = utils.get_mesh_actor_vertices_faces(self.mesh_store.meshes[self.mesh_store.reference].actor)
            current_vertices = utils.transform_vertices(verts, self.mesh_store.meshes[self.mesh_store.reference].actor.user_matrix)
            original_mesh = trimesh.Trimesh(original_vertices, faces, process=False)
            current_mesh = trimesh.Trimesh(current_vertices, faces, process=False)
            pose, _ = trimesh.registration.mesh_other(original_mesh, current_mesh)
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Pose Files (*.npy)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.npy')
                self.update_gt_pose()
                np.save(output_path, pose)
                self.output_text.append(f"-> Saved:\n{pose}\nExport to:\n {output_path}")
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
  
    def export_mesh_render(self, save_render=True):
        image = None
        if self.mesh_store.reference:
            image = self.mesh_store.render_mesh(camera=self.plotter.camera.copy())
            if save_render:
                output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Mesh Files (*.png)")
                if output_path:
                    if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.png')
                    rendered_image = PIL.Image.fromarray(image)
                    rendered_image.save(output_path)
                    self.output_text.append(f"-> Export mesh render to:\n {output_path}")
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Need to load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        return image

    def export_segmesh_render(self):
        if self.mesh_store.reference and self.mask_store.mask_actor:
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "SegMesh Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.png')
                mask_surface = self.mask_store.update_mask()
                self.load_mask(mask_surface)
                segmask = self.mask_store.render_mask(camera=self.plotter.camera.copy())
                if np.max(segmask) > 1: segmask = segmask / 255
                image = self.mesh_store.render_mesh(camera=self.plotter.camera.copy())
                image = (image * segmask).astype(np.uint8)
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.output_text.append(f"-> Export segmask render:\n to {output_path}")
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Need to load a mesh or mask first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
