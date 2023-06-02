import pathlib
import logging
import numpy as np
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import json
import PIL
import vtk
import ast

# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"

from PyQt5 import QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow
import vision6D as vis
from .GUI import MyMainWindow

np.set_printoptions(suppress=True)

class Interface_GUI(MyMainWindow):
    def __init__(self):
        super().__init__()

        # initialize
        self.reference = None
        self.transformation_matrix = np.eye(4)
        self.initial_pose = self.transformation_matrix

        self.mirror_x = False
        self.mirror_y = False

        self.image_actor = None
        self.mask_actor = None
        self.mesh_actors = {}
        
        self.undo_poses = {}
        self.latlon = vis.utils.load_latitude_longitude()

        self.colors = ["cyan", "magenta", "yellow", "lime", "deepskyblue", "salmon", "silver", "aquamarine", "plum", "blueviolet"]
        self.used_colors = []
        self.mesh_colors = {}

        self.mesh_opacity = {}
        self.store_mesh_opacity = {}
        self.image_opacity = 0.99
        self.mask_opacity = 0.5
        self.surface_opacity = 0.8
        
        self.image_spacing = [0.01, 0.01, 1]
        self.mask_spacing = [0.01, 0.01, 1]
        self.mesh_spacing = [1, 1, 1]
        
    def button_actor_name_clicked(self, text):
        if text in self.mesh_actors:
            # set the current mesh color
            self.color_button.setText(self.mesh_colors[text])
            # set mesh reference
            self.reference = text
            curr_opacity = self.mesh_actors[self.reference].GetProperty().opacity
            self.opacity_slider.setValue(curr_opacity * 100)
        else:
            self.color_button.setText("Color")
            if text == 'image': curr_opacity = self.image_opacity
            elif text == 'mask': curr_opacity = self.mask_opacity
            self.opacity_slider.setValue(curr_opacity * 100)
            self.reference = None
        
        output = f"-> Actor {text}, and its opacity is {curr_opacity}"
        if output not in self.output_text.toPlainText(): self.output_text.append(output)
                                            
    def set_image_opacity(self, image_opacity: float):
        assert image_opacity>=0 and image_opacity<=1, "image opacity should range from 0 to 1!"
        self.image_opacity = image_opacity
        self.image_actor.GetProperty().opacity = image_opacity
        self.plotter.add_actor(self.image_actor, pickable=False, name='image')

    def set_mask_opacity(self, mask_opacity: float):
        assert mask_opacity>=0 and mask_opacity<=1, "image opacity should range from 0 to 1!"
        self.mask_opacity = mask_opacity
        self.mask_actor.GetProperty().opacity = mask_opacity
        self.plotter.add_actor(self.mask_actor, pickable=False, name='mask')

    def set_mesh_opacity(self, name: str, surface_opacity: float):
        assert surface_opacity>=0 and surface_opacity<=1, "mesh opacity should range from 0 to 1!"
        self.mesh_opacity[name] = surface_opacity
        self.mesh_actors[name].user_matrix = pv.array_from_vtkmatrix(self.mesh_actors[name].GetMatrix())
        self.mesh_actors[name].GetProperty().opacity = surface_opacity
        self.plotter.add_actor(self.mesh_actors[name], pickable=True, name=name)

    def add_image(self, image_source):

        if isinstance(image_source, pathlib.WindowsPath) or isinstance(image_source, str):
            image_source = np.array(PIL.Image.open(image_source), dtype='uint8')
        if len(image_source.shape) == 2: 
            image_source = image_source[..., None]

        if self.mirror_x: image_source = image_source[:, ::-1, :]
        if self.mirror_y: image_source = image_source[::-1, :, :]

        dim = image_source.shape
        h, w, channel = dim[0], dim[1], dim[2]

        # Create the render based on the image size
        self.render = pv.Plotter(window_size=[w, h], lighting=None, off_screen=True) 
        self.render.set_background('black'); assert self.render.background_color == "black", "render's background need to be black"

        image = pv.UniformGrid(dimensions=(w, h, 1), spacing=self.image_spacing, origin=(0.0, 0.0, 0.0))
        image.point_data["values"] = image_source.reshape((w * h, channel)) # order = 'C
        image = image.translate(-1 * np.array(image.center), inplace=False)

        # Then add it to the plotter
        image = self.plotter.add_mesh(image, cmap='gray', opacity=self.image_opacity, name='image') if channel == 1 else self.plotter.add_mesh(image, rgb=True, opacity=self.image_opacity, name='image')
        actor, _ = self.plotter.add_actor(image, pickable=False, name='image')
        # Save actor for later
        self.image_actor = actor
        
        # get the image scalar
        image_data = vis.utils.get_image_mask_actor_scalars(self.image_actor)
        assert (image_data == image_source).all() or (image_data*255 == image_source).all(), "image_data and image_source should be equal"

        # add remove current image to removeMenu
        if 'image' not in self.track_actors_names:
            self.track_actors_names.append('image')
            self.add_button_actor_name('image')

        self.check_button('image')

    def add_mask(self, mask_source):

        if isinstance(mask_source, pathlib.WindowsPath) or isinstance(mask_source, str):
            mask_source = np.array(PIL.Image.open(mask_source), dtype='uint8')
        if len(mask_source.shape) == 2: 
            mask_source = mask_source[..., None]

        if self.mirror_x: mask_source = mask_source[:, ::-1, :]
        if self.mirror_y: mask_source = mask_source[::-1, :, :]

        dim = mask_source.shape
        h, w, channel = dim[0], dim[1], dim[2]
        
        mask = pv.UniformGrid(dimensions=(w, h, 1), spacing=self.mask_spacing, origin=(0.0, 0.0, 0.0))
        mask.point_data["values"] = mask_source.reshape((w * h, channel)) # order = 'C
        mask = mask.translate(-1 * np.array(mask.center), inplace=False)

        # Then add it to the plotter
        mask = self.plotter.add_mesh(mask, cmap='gray', opacity=self.mask_opacity, name='mask') if channel == 1 else self.plotter.add_mesh(mask, rgb=True, opacity=self.mask_opacity, name='mask')
        actor, _ = self.plotter.add_actor(mask, pickable=False, name='mask')
        # Save actor for later
        self.mask_actor = actor

        # get the image scalar
        mask_data = vis.utils.get_image_mask_actor_scalars(self.mask_actor)
        assert (mask_data == mask_source).all() or (mask_data*255 == mask_source).all(), "mask_data and mask_source should be equal"

        # add remove current image to removeMenu
        if 'mask' not in self.track_actors_names:
            self.track_actors_names.append('mask')
            self.add_button_actor_name('mask')

        self.check_button('mask')

    def add_pose(self, matrix:np.ndarray=None, rot:np.ndarray=None, trans:np.ndarray=None):
        if matrix is None: matrix = np.vstack((np.hstack((rot, trans)), [0, 0, 0, 1])) #if (rot is not None and trans is not None) else None
        self.initial_pose = matrix #if matrix is not None else self.transformation_matrix
        self.reset_gt_pose()
        self.reset_camera()

    def add_mesh(self, mesh_name, mesh_source, transformation_matrix=None):
        """ add a mesh to the pyqt frame """

        flag = False
                              
        if isinstance(mesh_source, pathlib.WindowsPath) or isinstance(mesh_source, str):
            # Load the '.mesh' file
            if pathlib.Path(mesh_source).suffix == '.mesh': mesh_source = vis.utils.load_trimesh(mesh_source)
            # Load the '.ply' file
            else: mesh_source = pv.read(mesh_source)

        if isinstance(mesh_source, trimesh.Trimesh):
            assert (mesh_source.vertices.shape[1] == 3 and mesh_source.faces.shape[1] == 3), "it should be N by 3 matrix"
            mesh_data = pv.wrap(mesh_source)
            source_verts = mesh_source.vertices * self.mesh_spacing
            source_faces = mesh_source.faces
            flag = True

        if isinstance(mesh_source, pv.PolyData):
            mesh_data = mesh_source
            source_verts = mesh_source.points * self.mesh_spacing
            source_faces = mesh_source.faces.reshape((-1, 4))[:, 1:]
            flag = True

        if not flag:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "The mesh format is not supported!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

        # consider the mesh verts spacing
        mesh_data.points = mesh_data.points * self.mesh_spacing

        # assign a color to every mesh
        if len(self.colors) != 0: mesh_color = self.colors.pop(0)
        else:
            self.colors = self.used_colors
            mesh_color = self.colors.pop(0)
            self.used_colors = []

        self.used_colors.append(mesh_color)
        self.mesh_colors[mesh_name] = mesh_color
        self.color_button.setText(self.mesh_colors[mesh_name])
        mesh = self.plotter.add_mesh(mesh_data, color=mesh_color, opacity=self.mesh_opacity[mesh_name], name=mesh_name)

        mesh.user_matrix = self.transformation_matrix if transformation_matrix is None else transformation_matrix
        self.initial_pose = mesh.user_matrix
                
        # Add and save the actor
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=mesh_name)

        actor_vertices, actor_faces = vis.utils.get_mesh_actor_vertices_faces(actor)
        assert (actor_vertices == source_verts).all(), "vertices should be the same"
        assert (actor_faces == source_faces).all(), "faces should be the same"
        assert actor.name == mesh_name, "actor's name should equal to mesh_name"
        
        self.mesh_actors[mesh_name] = actor

        # add remove current mesh to removeMenu
        if mesh_name not in self.track_actors_names:
            self.track_actors_names.append(mesh_name)
            self.add_button_actor_name(mesh_name)

        self.check_button(mesh_name)

    def reset_camera(self, *args):
        self.plotter.camera = self.camera.copy()

    def zoom_in(self, *args):
        self.plotter.camera.zoom(2)

    def zoom_out(self, *args):
        self.plotter.camera.zoom(0.5)

    def check_button(self, actor_name):
        for button in self.button_group_actors_names.buttons():
            if button.text() == actor_name: 
                button.setChecked(True)
                self.button_actor_name_clicked(actor_name)
                break

    def pick_callback(self, obj, *args):
        x, y = obj.GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.Pick(x, y, 0, self.plotter.renderer)
        picked_actor = picker.GetActor()
        if picked_actor is not None:
            actor_name = picked_actor.name
            if actor_name in self.mesh_actors:        
                if actor_name not in self.undo_poses: self.undo_poses[actor_name] = []
                self.undo_poses[actor_name].append(self.mesh_actors[actor_name].user_matrix)
                if len(self.undo_poses[actor_name]) > 20: self.undo_poses[actor_name].pop(0)
                # check the picked button
                self.check_button(actor_name)

    def toggle_image_opacity(self, *args, up):
        if up:
            self.image_opacity += 0.05
            if self.image_opacity > 1: self.image_opacity = 1
        else:
            self.image_opacity -= 0.05
            if self.image_opacity < 0: self.image_opacity = 0
        self.image_actor.GetProperty().opacity = self.image_opacity
        self.plotter.add_actor(self.image_actor, pickable=False, name="image")
        self.ignore_slider_value_change = True
        self.opacity_slider.setValue(self.image_opacity * 100)
        self.ignore_slider_value_change = False

    def toggle_mask_opacity(self, *args, up):
        if up:
            self.mask_opacity += 0.05
            if self.mask_opacity > 1: self.mask_opacity = 1
        else:
            self.mask_opacity -= 0.05
            if self.mask_opacity < 0: self.mask_opacity = 0
        self.mask_actor.GetProperty().opacity = self.mask_opacity
        self.plotter.add_actor(self.mask_actor, pickable=False, name="mask")
        self.ignore_slider_value_change = True
        self.opacity_slider.setValue(self.mask_opacity * 100)
        self.ignore_slider_value_change = False

    def toggle_mesh_opacity(self, *args, up):
        current_opacity = self.opacity_slider.value() / 100
        if up:
            current_opacity += 0.05
            if current_opacity > 1: current_opacity = 1
        else:
            current_opacity -= 0.05
            if current_opacity < 0: current_opacity = 0
        self.opacity_slider.setValue(current_opacity * 100)

    def reset_gt_pose(self, *args):
        self.output_text.append(f"-> Reset the GT pose to: \n{self.initial_pose}")
        for actor_name, actor in self.mesh_actors.items():
            actor.user_matrix = self.initial_pose
            self.plotter.add_actor(actor, pickable=True, name=actor_name)

    def update_gt_pose(self, *args):
        if self.reference is not None:
            self.transformation_matrix = self.mesh_actors[self.reference].user_matrix
            self.initial_pose = self.transformation_matrix
            self.output_text.append(f"-> Current reference mesh is: <span style='background-color:yellow; color:black;'>{self.reference}</span>")
            self.output_text.append(f"Update the GT pose to: \n{self.initial_pose}")
            for actor_name, actor in self.mesh_actors.items():
                actor.user_matrix = self.initial_pose
                self.plotter.add_actor(actor, pickable=True, name=actor_name)

    def current_pose(self, *args):
        if self.reference is not None:
            transformation_matrix = self.mesh_actors[self.reference].user_matrix
            self.output_text.append(f"-> Current reference mesh is: <span style='background-color:yellow; color:black;'>{self.reference}</span>")
            self.output_text.append(f"Current pose is: \n{transformation_matrix}")
            for actor_name, actor in self.mesh_actors.items():
                actor.user_matrix = transformation_matrix
                self.plotter.add_actor(actor, pickable=True, name=actor_name)

    def undo_pose(self, *args):
        if self.button_group_actors_names.checkedButton() is not None:
            actor_name = self.button_group_actors_names.checkedButton().text()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Choose a mesh actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        if len(self.undo_poses[actor_name]) != 0: 
            transformation_matrix = self.undo_poses[actor_name].pop()
            if (transformation_matrix == self.mesh_actors[actor_name].user_matrix).all():
                if len(self.undo_poses[actor_name]) != 0: 
                    transformation_matrix = self.undo_poses[actor_name].pop()

            self.output_text.append(f"-> Current reference mesh is: <span style='background-color:yellow; color:black;'>{actor_name}</span>")
            self.output_text.append(f"Undo pose to: \n{transformation_matrix}")
                
            self.mesh_actors[actor_name].user_matrix = transformation_matrix
            self.plotter.add_actor(self.mesh_actors[actor_name], pickable=True, name=actor_name)

    def set_scalar(self, nocs, actor_name):
        vertices, faces = vis.utils.get_mesh_actor_vertices_faces(self.mesh_actors[actor_name])
        vertices_color = vertices
        if self.mirror_x: vertices_color = vis.utils.transform_vertices(vertices_color, np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        if self.mirror_y: vertices_color = vis.utils.transform_vertices(vertices_color, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        # get the corresponding color
        colors = vis.utils.color_mesh(vertices_color, nocs=nocs)
        if colors.shape != vertices.shape: 
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Cannot set the selected color", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        assert colors.shape == vertices.shape, "colors shape should be the same as vertices shape"
        # color the mesh and actor
        mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        mesh = self.plotter.add_mesh(mesh_data, scalars=colors, rgb=True, opacity=self.mesh_opacity[actor_name], name=actor_name)
        transformation_matrix = pv.array_from_vtkmatrix(self.mesh_actors[actor_name].GetMatrix())
        mesh.user_matrix = transformation_matrix
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=actor_name)
        actor_colors = vis.utils.get_mesh_actor_scalars(actor)
        assert (actor_colors == colors).all(), "actor_colors should be the same as colors"
        assert actor.name == actor_name, "actor's name should equal to actor_name"
        self.mesh_actors[actor_name] = actor

    def set_color(self, color, actor_name):
        vertices, faces = vis.utils.get_mesh_actor_vertices_faces(self.mesh_actors[actor_name])
        mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        mesh = self.plotter.add_mesh(mesh_data, color=color, opacity=self.mesh_opacity[actor_name], name=actor_name)
        transformation_matrix = pv.array_from_vtkmatrix(self.mesh_actors[actor_name].GetMatrix())
        mesh.user_matrix = transformation_matrix
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=actor_name)
        assert actor.name == actor_name, "actor's name should equal to actor_name"
        self.mesh_actors[actor_name] = actor
        
    def nocs_epnp(self, color_mask, mesh):
        vertices = mesh.vertices
        pts3d, pts2d = vis.utils.create_2d_3d_pairs(color_mask, vertices)
        pts2d = pts2d.astype('float32')
        pts3d = pts3d.astype('float32')
        camera_intrinsics = self.camera_intrinsics.astype('float32')
        predicted_pose = vis.utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics, self.camera.position)
        return predicted_pose

    def latlon_epnp(self, color_mask, mesh):
        binary_mask = vis.utils.color2binary_mask(color_mask)
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

        lat = np.array(self.latlon[..., 0])
        lon = np.array(self.latlon[..., 1])
        lonf = lon[mesh.faces]
        msk = (np.sum(lonf>=0, axis=1)==3) & (np.sum(lat[mesh.faces]>=0, axis=1)==3)
        for i in range(len(pts2d)):
            pt = vis.utils.latLon2xyz(mesh, lat, lonf, msk, gx[i], gy[i])
            pts3d.append(pt)
       
        pts3d = np.array(pts3d).reshape((len(pts3d), 3))

        pts2d = pts2d.astype('float32')
        pts3d = pts3d.astype('float32')
        camera_intrinsics = self.camera_intrinsics.astype('float32')
        
        predicted_pose = vis.utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics, self.camera.position)

        return predicted_pose

    def epnp_mesh(self):
        if self.reference is not None:
            colors = vis.utils.get_mesh_actor_scalars(self.mesh_actors[self.reference])
            if colors is None or (np.all(colors == colors[0])):
                QtWidgets.QMessageBox.warning(self, 'vision6D', "The mesh need to be colored with nocs or latlon with gradient color", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return 0
            color_mask = self.export_mesh_plot(QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Yes, save_render=False)
            gt_pose = self.mesh_actors[self.reference].user_matrix
            if self.mirror_x: gt_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
            if self.mirror_y: gt_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose

            if np.sum(color_mask) == 0:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "The color mask is blank (maybe set the reference mesh wrong)", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return 0
                
            if self.mesh_colors[self.reference] == 'nocs':
                vertices, faces = vis.utils.get_mesh_actor_vertices_faces(self.mesh_actors[self.reference])
                mesh = trimesh.Trimesh(vertices, faces, process=False)
                predicted_pose = self.nocs_epnp(color_mask, mesh)
                if self.mirror_x: predicted_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                if self.mirror_y: predicted_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                error = np.sum(np.abs(predicted_pose - gt_pose))
                self.output_text.append(f"-> PREDICTED POSE WITH <span style='background-color:yellow; color:black;'>NOCS COLOR</span>: ")
                self.output_text.append(f"{predicted_pose}\nGT POSE: \n{gt_pose}\nERROR: \n{error}")

            else:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Only works using EPnP with latlon mask", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "A mesh need to be loaded/mesh reference need to be set", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def epnp_mask(self, nocs_method):
        if self.mask_actor is not None:
            mask_data = vis.utils.get_image_mask_actor_scalars(self.mask_actor)
            if np.max(mask_data) > 1: mask_data = mask_data / 255

            # binary mask
            if np.all(np.logical_or(mask_data == 0, mask_data == 1)):
                if self.reference is not None:
                    colors = vis.utils.get_mesh_actor_scalars(self.mesh_actors[self.reference])
                    if colors is None or (np.all(colors == colors[0])):
                        QtWidgets.QMessageBox.warning(self, 'vision6D', "The mesh need to be colored with nocs or latlon with gradient color", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                        return 0
                    color_mask = self.export_mesh_plot(QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Yes, save_render=False)
                    nocs_color = (self.mesh_colors[self.reference] == 'nocs')
                    gt_pose = self.mesh_actors[self.reference].user_matrix
                    if self.mirror_x: gt_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                    if self.mirror_y: gt_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                    vertices, faces = vis.utils.get_mesh_actor_vertices_faces(self.mesh_actors[self.reference])
                    mesh = trimesh.Trimesh(vertices, faces, process=False)
                else: 
                    QtWidgets.QMessageBox.warning(self, 'vision6D', "A mesh need to be loaded/mesh reference need to be set", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    return 0
                color_mask = (color_mask * mask_data).astype(np.uint8)
            # color mask
            else:
                color_mask = mask_data
                if np.sum(color_mask) != 0:
                    unique, counts = np.unique(color_mask, return_counts=True)
                    digit_counts = dict(zip(unique, counts))
                    if digit_counts[0] == np.max(counts): 
                        nocs_color = False if np.sum(color_mask[..., 2]) == 0 else True

                        # Set the pose information if the format is correct
                        res = self.set_pose()
                        if res is None: return 0
                        
                        gt_pose = self.transformation_matrix
                                                
                        # add the mesh object file
                        QtWidgets.QMessageBox.information(self, "Information", "Please load the corresponding mesh")
                        
                        self.add_mesh_file(prompt=True)
                        if self.mesh_path != '':
                            checked_button = self.button_group_actors_names.checkedButton()
                            vertices, faces = vis.utils.get_mesh_actor_vertices_faces(self.mesh_actors[checked_button.text()])
                            mesh = trimesh.Trimesh(vertices, faces, process=False)
                        else: return 0
                    else:
                        QtWidgets.QMessageBox.warning(self, 'vision6D', "A color mask need to be loaded", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                        return 0
        
            if np.sum(color_mask) == 0:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "The color mask is blank (maybe set the reference mesh wrong)", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return 0
                
            if nocs_method == nocs_color:
                if nocs_method: 
                    predicted_pose = self.nocs_epnp(color_mask, mesh)
                    if self.mirror_x: predicted_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                    if self.mirror_y: predicted_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                    color_theme = 'NOCS'
                else: 
                    if self.mirror_x: color_mask = color_mask[:, ::-1, :]
                    if self.mirror_y: color_mask = color_mask[::-1, :, :]
                    predicted_pose = self.latlon_epnp(color_mask, mesh)
                    color_theme = 'LATLON'
                error = np.sum(np.abs(predicted_pose - gt_pose))
                self.output_text.append(f"-> PREDICTED POSE WITH <span style='background-color:yellow; color:black;'>{color_theme} COLOR (MASKED)</span>: ")
                self.output_text.append(f"{predicted_pose}\nGT POSE: \n{gt_pose}\nERROR: \n{error}")
            else:
                QtWidgets.QMessageBox.warning(self,"vision6D", "Clicked the wrong method")
        else:
            QtWidgets.QMessageBox.warning(self,"vision6D", "please load a mask first")
