'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: mesh_container.py
@time: 2023-07-03 20:27
@desc: create container for mesh related actions in application
'''
import os
import pathlib

import trimesh
import PIL.Image
import matplotlib
import numpy as np
import pyvista as pv

from PyQt5 import QtWidgets

from ..tools import utils
from ..tools import exception
from ..components import MaskStore
from ..components import ImageStore
from ..components import MeshStore
from ..widgets import GetPoseDialog

from ..path import PKG_ROOT

class MeshContainer:
    def __init__(self,
                plotter, 
                hintLabel,
                track_actors_names, 
                add_button_actor_name, 
                button_group_actors_names,
                check_button,
                reset_camera,
                toggle_register,
                load_mask,
                output_text):

        self.plotter = plotter
        self.hintLabel = hintLabel
        self.track_actors_names = track_actors_names
        self.add_button_actor_name = add_button_actor_name
        self.button_group_actors_names = button_group_actors_names
        self.check_button = check_button
        self.reset_camera = reset_camera
        self.toggle_register = toggle_register
        self.load_mask = load_mask
        self.output_text = output_text
        
        self.toggle_hide_meshes_flag = False

        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.mesh_store = MeshStore()

    def add_mesh_file(self, mesh_path='', prompt=False):
        if prompt: 
            mesh_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.mesh *.ply *.stl *.obj *.off *.dae *.fbx *.3ds *.x3d)") 
        if mesh_path:
            self.hintLabel.hide()
            mesh_data = self.mesh_store.add_mesh(mesh_source=mesh_path)
            if mesh_data: 
                if self.mesh_store.reference is not None:
                    name = self.mesh_store.reference
                    reference_matrix = self.mesh_store.meshes[name].actor.user_matrix
                    self.add_mesh(mesh_data, reference_matrix)
                else:
                    self.add_mesh(mesh_data, np.array([[1, 0, 0, 0], 
                                                    [0, 1, 0, 0], 
                                                    [0, 0, 1, 1e+3], 
                                                    [0, 0, 0, 1]])) # set the initial pose, r_x, r_y, t_z includes the scaling too
            else: utils.display_warning("The mesh format is not supported!")

    def mirror_mesh(self, name, direction):
        mesh_data = self.mesh_store.meshes[name]
        if direction == 'x': mesh_data.mirror_x = not mesh_data.mirror_x
        elif direction == 'y': mesh_data.mirror_y = not mesh_data.mirror_y
        if (mesh_data.initial_pose != np.eye(4)).all(): mesh_data.initial_pose = mesh_data.actor.user_matrix
        transformation_matrix = mesh_data.actor.user_matrix
        if mesh_data.mirror_x: 
            transformation_matrix[0,0] *= -1
            mesh_data.mirror_x = False # very important
        if mesh_data.mirror_y: 
            transformation_matrix[2,2] *= -1
            mesh_data.mirror_y = False # very important
        mesh_data.actor.user_matrix = transformation_matrix
        self.check_button(name=name, output_text=False)
        
    def add_mesh(self, mesh_data, transformation_matrix):
        """ add a mesh to the pyqt frame """
        mesh = self.plotter.add_mesh(mesh_data.pv_mesh, color=mesh_data.color, opacity=mesh_data.opacity, name=mesh_data.name)
        mesh.user_matrix = transformation_matrix
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=mesh_data.name)
        mesh_data.actor = actor

        # add remove current mesh to removeMenu
        if mesh_data.name not in self.track_actors_names:
            self.track_actors_names.append(mesh_data.name)
            self.add_button_actor_name(mesh_data.name)
        #* very important for mirroring
        self.check_button(name=mesh_data.name, output_text=False) 
        self.reset_camera()
        
    def set_spacing(self):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button:
            name = checked_button.text()
            if name in self.mesh_store.meshes:
                mesh_data = self.mesh_store.meshes[name]
                spacing, ok = QtWidgets.QInputDialog().getText(QtWidgets.QMainWindow(), 'Input', "Set Spacing", text=str(mesh_data.spacing))
                if ok:
                    mesh_data.spacing = exception.set_spacing(spacing)
                    # Calculate the centroid
                    centroid = np.mean(mesh_data.source_mesh.vertices, axis=0)
                    offset = mesh_data.source_mesh.vertices - centroid
                    scaled_offset = offset * mesh_data.spacing
                    vertices = centroid + scaled_offset
                    mesh_data.pv_mesh.points = vertices
            else: utils.display_warning("Need to select a mesh object instead")
        else: utils.display_warning("Need to select a mesh actor first")
        
    def set_color(self, color, name):
        scalars = None
        mesh_data = self.mesh_store.meshes[name]
        if color == "nocs": scalars = utils.color_mesh_nocs(mesh_data.pv_mesh.points)
        elif color == "texture":
            texture_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.npy)")
            if texture_path: 
                mesh_data.texture_path = texture_path
                scalars = np.load(texture_path) / 255 # make sure the color range is from 0 to 1

        if scalars is not None: mesh = self.plotter.add_mesh(mesh_data.pv_mesh, scalars=scalars, rgb=True, opacity=mesh_data.opacity, name=name)
        else:
            mesh = self.plotter.add_mesh(mesh_data.pv_mesh, opacity=mesh_data.opacity, name=name)
            mesh.GetMapper().SetScalarVisibility(0)
            if color in self.mesh_store.colors: mesh.GetProperty().SetColor(matplotlib.colors.to_rgb(color))
            else:
                mesh_data.color = "wheat"
                mesh.GetProperty().SetColor(matplotlib.colors.to_rgb(mesh_data.color))

        mesh.user_matrix = mesh_data.actor.user_matrix
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=name)
        mesh_data.actor = actor #^ very import to change the actor too!
        if scalars is None and color == 'texture': color = mesh_data.color
        return color
        """
        if color in self.mesh_store.colors:
            mesh_data.actor.GetMapper().SetScalarVisibility(0)
            mesh_data.actor.GetProperty().SetColor(matplotlib.colors.to_rgb(color))
        else:
            if color == "nocs": scalars = utils.color_mesh_nocs(mesh_data.pv_mesh.points)
            else: 
                texture_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.npy)")
                if texture_path:
                    mesh_data.texture_path = texture_path
                    scalars = np.load(texture_path) / 255 # make sure the color range is from 0 to 1
            if scalars is not None: 
                mesh = self.plotter.add_mesh(mesh_data.pv_mesh, scalars=scalars, rgb=True, opacity=mesh_data.opacity, name=name)
            else: 
                mesh = self.plotter.add_mesh(mesh_data.pv_mesh, opacity=mesh_data.opacity, name=name)
                mesh.GetMapper().SetScalarVisibility(0)
                mesh.GetProperty().SetColor(matplotlib.colors.to_rgb(mesh_data.color))
            mesh.user_matrix = mesh_data.actor.user_matrix
            actor, _ = self.plotter.add_actor(mesh, pickable=True, name=name)
            mesh_data.actor = actor #^ very import to change the actor too!
            if scalars is None and color == 'texture': color = mesh_data.color
        return color
        """
            
    def set_mesh_opacity(self, name: str, mesh_opacity: float):
        mesh_data = self.mesh_store.meshes[name]
        mesh_data.previous_opacity = mesh_data.opacity
        mesh_data.opacity = mesh_opacity
        mesh_data.actor.user_matrix = pv.array_from_vtkmatrix(mesh_data.actor.GetMatrix())
        mesh_data.actor.GetProperty().opacity = mesh_opacity

    def toggle_surface_opacity(self, up):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button:
            name = checked_button.text()
            if name in self.mesh_store.meshes: 
                change = 0.05
                if not up: change *= -1
                current_opacity = self.mesh_store.meshes[name].opacity_spinbox.value()
                current_opacity += change
                current_opacity = np.clip(current_opacity, 0, 1)
                self.mesh_store.meshes[name].opacity_spinbox.setValue(current_opacity)
                
    def handle_hide_meshes_opacity(self, flag):
        checked_button = self.button_group_actors_names.checkedButton()
        checked_name = checked_button.text() if checked_button else None
        for button in self.button_group_actors_names.buttons():
            name = button.text()
            if name not in self.mesh_store.meshes: continue
            if len(self.mesh_store.meshes) != 1 and name == checked_name: continue
            mesh_data = self.mesh_store.meshes[name]
            if flag: self.set_mesh_opacity(name, 0)
            else: self.set_mesh_opacity(name, mesh_data.previous_opacity)
            
    def toggle_hide_meshes_button(self):
        self.toggle_hide_meshes_flag = not self.toggle_hide_meshes_flag
        self.handle_hide_meshes_opacity(self.toggle_hide_meshes_flag)
                            
    def add_pose_file(self, pose_path):
        if pose_path:
            self.hintLabel.hide()
            if isinstance(pose_path, list): transformation_matrix = np.array(pose_path)
            else: transformation_matrix = np.load(pose_path)
            mesh_data = self.mesh_store.meshes[self.mesh_store.reference]
            if mesh_data.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            if mesh_data.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            self.add_pose(matrix=transformation_matrix)

    def add_pose(self, matrix:np.ndarray=None, rot:np.ndarray=None, trans:np.ndarray=None):
        if matrix is None and (rot is not None and trans is not None): matrix = np.vstack((np.hstack((rot, trans)), [0, 0, 0, 1]))
        self.mesh_store.meshes[self.mesh_store.reference].initial_pose = matrix
        self.reset_gt_pose()
        
    def set_pose(self):
        if self.mesh_store.reference:
            mesh_data = self.mesh_store.meshes[self.mesh_store.reference]
            get_pose_dialog = GetPoseDialog(mesh_data.actor.user_matrix)
            res = get_pose_dialog.exec_()
            if res == QtWidgets.QDialog.Accepted:
                user_text = get_pose_dialog.get_text()
                if "," not in user_text:
                    user_text = user_text.replace(" ", ",")
                    user_text =user_text.strip().replace("[,", "[")
                gt_pose = exception.set_data_format(user_text, mesh_data.actor.user_matrix)
                if gt_pose.shape == (4, 4):
                    self.hintLabel.hide()
                    transformation_matrix = gt_pose
                    if mesh_data.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                    if mesh_data.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                    self.add_pose(matrix=transformation_matrix)
                else: utils.display_warning("It needs to be a 4 by 4 matrix")
        else: utils.display_warning("Needs to select a mesh first")
    
    def reset_gt_pose(self):
        if self.mesh_store.reference:
            mesh_data = self.mesh_store.meshes[self.mesh_store.reference]
            # if mesh_data.initial_pose is not None:
            text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
            mesh_data.initial_pose[0, 0], mesh_data.initial_pose[0, 1], mesh_data.initial_pose[0, 2], mesh_data.initial_pose[0, 3], 
            mesh_data.initial_pose[1, 0], mesh_data.initial_pose[1, 1], mesh_data.initial_pose[1, 2], mesh_data.initial_pose[1, 3], 
            mesh_data.initial_pose[2, 0], mesh_data.initial_pose[2, 1], mesh_data.initial_pose[2, 2], mesh_data.initial_pose[2, 3],
            mesh_data.initial_pose[3, 0], mesh_data.initial_pose[3, 1], mesh_data.initial_pose[3, 2], mesh_data.initial_pose[3, 3])
            self.output_text.append("-> Reset the GT pose to:")
            self.output_text.append(text)
            self.toggle_register(mesh_data.initial_pose)
            self.reset_camera()
        else: utils.display_warning("Need to set a reference mesh first")

    def update_gt_pose(self):
        if self.mesh_store.reference:
            mesh_data = self.mesh_store.meshes[self.mesh_store.reference]
            # if mesh_data.initial_pose is not None:
            mesh_data.initial_pose = mesh_data.actor.user_matrix
            self.toggle_register(mesh_data.actor.user_matrix)
            text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
            mesh_data.initial_pose[0, 0], mesh_data.initial_pose[0, 1], mesh_data.initial_pose[0, 2], mesh_data.initial_pose[0, 3], 
            mesh_data.initial_pose[1, 0], mesh_data.initial_pose[1, 1], mesh_data.initial_pose[1, 2], mesh_data.initial_pose[1, 3], 
            mesh_data.initial_pose[2, 0], mesh_data.initial_pose[2, 1], mesh_data.initial_pose[2, 2], mesh_data.initial_pose[2, 3],
            mesh_data.initial_pose[3, 0], mesh_data.initial_pose[3, 1], mesh_data.initial_pose[3, 2], mesh_data.initial_pose[3, 3])
            self.output_text.append(f"-> Update the {self.mesh_store.reference} GT pose to:")
            self.output_text.append(text)
        else: utils.display_warning("Need to set a reference mesh first")

    def undo_actor_pose(self):
        if self.button_group_actors_names.checkedButton():
            name = self.button_group_actors_names.checkedButton().text()
            if name in self.mesh_store.meshes:
                self.mesh_store.undo_actor_pose(name)
                self.check_button(name=name) # very important, donnot change this line to "toggle_register"
        else: utils.display_warning("Choose a mesh actor first")

    def render_mesh(self, camera):
        render = utils.create_render(self.image_store.width, self.image_store.height); render.clear()
        mesh_data = self.mesh_store.meshes[self.mesh_store.reference]
        vertices, faces = utils.get_mesh_actor_vertices_faces(mesh_data.actor)
        pv_mesh = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        colors = utils.get_mesh_actor_scalars(mesh_data.actor)
        if colors is not None: mesh = render.add_mesh(pv_mesh, scalars=colors, rgb=True, style='surface', opacity=1, name=self.mesh_store.reference)
        else: mesh = render.add_mesh(pv_mesh, color=mesh_data.color, style='surface', opacity=1, name=self.mesh_store.reference)
        mesh.user_matrix = mesh_data.actor.user_matrix
        # set the light source to add the textures
        light = pv.Light(light_type='headlight')
        render.add_light(light)
        render.camera = camera
        render.disable()
        render.show(auto_close=False)
        image = render.last_image
        return image

    def export_mesh_pose(self):
        for mesh_data in self.mesh_store.meshes.values():
            verts, faces = utils.get_mesh_actor_vertices_faces(mesh_data.actor)
            vertices = utils.transform_vertices(verts, mesh_data.actor.user_matrix)
            os.makedirs(PKG_ROOT.parent / "output" / "export_mesh", exist_ok=True)
            output_path = PKG_ROOT.parent / "output" / "export_mesh" / (mesh_data.name + '.ply')
            mesh = trimesh.Trimesh(vertices, faces, process=False)
            ply_file = trimesh.exchange.ply.export_ply(mesh)
            with open(output_path, "wb") as fid: fid.write(ply_file)
            self.output_text.append(f"Export {mesh_data.name} mesh to:\n {output_path}")
                
    def export_mesh_render(self, save_render=True):
        image = None
        if self.mesh_store.reference:
            image = self.render_mesh(camera=self.plotter.camera.copy())
            if save_render:
                output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Mesh Files (*.png)")
                if output_path:
                    if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.png')
                    rendered_image = PIL.Image.fromarray(image)
                    rendered_image.save(output_path)
                    self.output_text.append(f"-> Export mesh render to:\n {output_path}")
        else: utils.display_warning("Need to load a mesh first")
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
                image = self.render_mesh(camera=self.plotter.camera.copy())
                image = (image * segmask).astype(np.uint8)
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.output_text.append(f"-> Export segmask render:\n to {output_path}")
        else: utils.display_warning("Need to load a mesh or mask first")
