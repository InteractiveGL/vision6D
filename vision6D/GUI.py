import os
import numpy as np
import pyvista as pv
import functools
import trimesh
import pathlib
import PIL
import ast
import json
import math
import copy
import cv2
import vtk
import pyvista as pv
import vision6D as vis

# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QPoint
from .mainwindow import MyMainWindow

np.set_printoptions(suppress=True)

class Interface(MyMainWindow):
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
        self.meshdict = {}
        
        self.undo_poses = {}
        self.latlon = vis.utils.load_latitude_longitude()

        self.colors = ["cyan", "magenta", "yellow", "lime", "deepskyblue", "salmon", "silver", "aquamarine", "plum", "blueviolet"]
        self.used_colors = []
        self.mesh_colors = {}

        self.mesh_opacity = {}
        self.store_mesh_opacity = {}
        self.image_opacity = 0.8
        self.mask_opacity = 0.1
        self.surface_opacity = self.opacity_spinbox.value()
        
        self.image_spacing = [0.01, 0.01, 1]
        self.mask_spacing = [0.01, 0.01, 1]
        self.mesh_spacing = [1, 1, 1]
        
        # Set the camera
        self.camera = pv.Camera()
        self.fx = 50000
        self.fy = 50000
        self.cx = 960
        self.cy = 540
        self.cam_viewup = (0, -1, 0)
        self.cam_position = -500
        self.set_camera_props()

        self.track_actors_names = []
        self.button_group_actors_names = QtWidgets.QButtonGroup(self)
        self.toggle_hide_meshes_flag = False

        # Shortcut key bindings
        # camera related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("c"), self).activated.connect(self.reset_camera)
        QtWidgets.QShortcut(QtGui.QKeySequence("z"), self).activated.connect(self.zoom_out)
        QtWidgets.QShortcut(QtGui.QKeySequence("x"), self).activated.connect(self.zoom_in)

        # registration related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("k"), self).activated.connect(self.reset_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("l"), self).activated.connect(self.update_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("t"), self).activated.connect(self.current_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("s"), self).activated.connect(self.undo_pose)

        # change image opacity key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("b"), self).activated.connect(lambda up=True: self.toggle_image_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("n"), self).activated.connect(lambda up=False: self.toggle_image_opacity(up))

        # change mask opacity key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("g"), self).activated.connect(lambda up=True: self.toggle_mask_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("h"), self).activated.connect(lambda up=False: self.toggle_mask_opacity(up))

        # change mesh opacity key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("y"), self).activated.connect(lambda up=True: self.toggle_surface_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("u"), self).activated.connect(lambda up=False: self.toggle_surface_opacity(up))

        # Add shortcut to the right, left, space buttons
        QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self).activated.connect(self.next_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self).activated.connect(self.prev_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self).activated.connect(self.play_video)

    def button_actor_name_clicked(self, text):
        if text in self.mesh_actors:
            # set the current mesh color
            self.color_button.setText(self.mesh_colors[text])
            # set mesh reference
            self.reference = text
            curr_opacity = self.mesh_actors[self.reference].GetProperty().opacity
            self.opacity_spinbox.setValue(curr_opacity)
        else:
            self.color_button.setText("Color")
            if text == 'image': curr_opacity = self.image_opacity
            elif text == 'mask': curr_opacity = self.mask_opacity
            else: curr_opacity = self.opacity_spinbox.value()
            self.opacity_spinbox.setValue(curr_opacity)
            # self.reference = None
        
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
        self.initial_pose = matrix
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

    def reset_camera(self):
        self.plotter.camera = self.camera.copy()

    def zoom_in(self):
        self.plotter.camera.zoom(2)

    def zoom_out(self):
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

    def toggle_image_opacity(self, up):
        if up:
            self.image_opacity += 0.05
            if self.image_opacity > 1: self.image_opacity = 1
        else:
            self.image_opacity -= 0.05
            if self.image_opacity < 0: self.image_opacity = 0
        self.image_actor.GetProperty().opacity = self.image_opacity
        self.plotter.add_actor(self.image_actor, pickable=False, name="image")
        self.ignore_spinbox_value_change = True
        self.opacity_spinbox.setValue(self.image_opacity)
        self.ignore_spinbox_value_change = False

    def toggle_mask_opacity(self, up):
        if up:
            self.mask_opacity += 0.05
            if self.mask_opacity > 1: self.mask_opacity = 1
        else:
            self.mask_opacity -= 0.05
            if self.mask_opacity < 0: self.mask_opacity = 0
        self.mask_actor.GetProperty().opacity = self.mask_opacity
        self.plotter.add_actor(self.mask_actor, pickable=False, name="mask")
        self.ignore_spinbox_value_change = True
        self.opacity_spinbox.setValue(self.mask_opacity)
        self.ignore_spinbox_value_change = False

    def toggle_surface_opacity(self, up):
        current_opacity = self.opacity_spinbox.value()
        if up:
            current_opacity += 0.05
            if current_opacity > 1: current_opacity = 1
        else:
            current_opacity -= 0.05
            if current_opacity < 0: current_opacity = 0
        self.opacity_spinbox.setValue(current_opacity)

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
        if len(self.mesh_actors) == 1: self.reference = list(self.mesh_actors.keys())[0]
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
                if len(self.mesh_actors) == 1: self.reference = list(self.mesh_actors.keys())[0]
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
                    color_theme = 'NOCS'
                    predicted_pose = self.nocs_epnp(color_mask, mesh)
                    if self.mirror_x: predicted_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                    if self.mirror_y: predicted_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                else: 
                    color_theme = 'LATLON'
                    if self.mirror_x: color_mask = color_mask[:, ::-1, :]
                    if self.mirror_y: color_mask = color_mask[::-1, :, :]
                    predicted_pose = self.latlon_epnp(color_mask, mesh)
                error = np.sum(np.abs(predicted_pose - gt_pose))
                self.output_text.append(f"-> PREDICTED POSE WITH <span style='background-color:yellow; color:black;'>{color_theme} COLOR (MASKED)</span>: ")
                self.output_text.append(f"{predicted_pose}\nGT POSE: \n{gt_pose}\nERROR: \n{error}")
            else:
                QtWidgets.QMessageBox.warning(self,"vision6D", "Clicked the wrong method")
        else:
            QtWidgets.QMessageBox.warning(self,"vision6D", "please load a mask first")

    def set_camera_extrinsics(self):
        self.camera.SetPosition((0,0,self.cam_position))
        self.camera.SetFocalPoint((*self.camera.GetWindowCenter(),0)) # Get the camera window center
        self.camera.SetViewUp(self.cam_viewup)
    
    def set_camera_intrinsics(self):
        
        # Set camera intrinsic attribute
        self.camera_intrinsics = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
                
        # convert the principal point to window center (normalized coordinate system) and set it
        wcx = -2*(self.cx - float(self.window_size[0])/2) / self.window_size[0]
        wcy =  2*(self.cy - float(self.window_size[1])/2) / self.window_size[1]
        self.camera.SetWindowCenter(wcx, wcy) # (0,0)
        
        # Setting the view angle in degrees
        view_angle = (180 / math.pi) * (2.0 * math.atan2(self.window_size[1]/2.0, self.fx)) # or view_angle = np.degrees(2.0 * math.atan2(height/2.0, f)) # focal_length = (1080 / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
        self.camera.SetViewAngle(view_angle) # view angle should be in degrees
 
    def set_camera_props(self):
        self.set_camera_intrinsics()
        self.set_camera_extrinsics()
        self.plotter.camera = self.camera.copy()

    def camera_calibrate(self):
        if self.image_path != '' and self.image_path is not None:
            original_image = np.array(PIL.Image.open(self.image_path), dtype='uint8')
            # make the the original image shape is [h, w, 3] to match with the rendered calibrated_image
            original_image = original_image[..., :3]
            if len(original_image.shape) == 2: original_image = original_image[..., None]
            if original_image.shape[-1] == 1: original_image = np.dstack((original_image, original_image, original_image))
            calibrated_image = np.array(self.render_image(self.image_actor, self.plotter.camera.copy()), dtype='uint8')
            if original_image.shape != calibrated_image.shape:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Original image shape is not equal to calibrated image shape!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return 0
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
        calibrate_pop = vis.CalibrationPopWindow(calibrated_image, original_image)
        calibrate_pop.exec_()

    def set_camera(self):
        dialog = vis.CameraPropsInputDialog(
            line1=("Fx", self.fx), 
            line2=("Fy", self.fy), 
            line3=("Cx", self.cx), 
            line4=("Cy", self.cy), 
            line5=("View Up", self.cam_viewup), 
            line6=("Cam Position", self.cam_position))
        if dialog.exec():
            fx, fy, cx, cy, cam_viewup, cam_position = dialog.getInputs()
            pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup, pre_cam_position = self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position
            if not (fx == '' or fy == '' or cx == '' or cy == '' or cam_viewup == '' or cam_position == ''):
                try:
                    self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position = ast.literal_eval(fx), ast.literal_eval(fy), ast.literal_eval(cx), ast.literal_eval(cy), ast.literal_eval(cam_viewup), ast.literal_eval(cam_position)
                    self.set_camera_props()
                except:
                    self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position = pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup, pre_cam_position
                    QtWidgets.QMessageBox.warning(self, 'vision6D', "Error occured, check the format of the input values", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def set_pose(self):
        # get the gt pose
        res = self.get_text_dialog.exec_()

        if res == QtWidgets.QDialog.Accepted:
            try:
                gt_pose = ast.literal_eval(self.get_text_dialog.user_text)
                gt_pose = np.array(gt_pose)
                if gt_pose.shape != (4, 4): 
                    QtWidgets.QMessageBox.warning(self, 'vision6D', "It needs to be a 4 by 4 matrix", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok) 
                    return None
                else:
                    self.hintLabel.hide()
                    transformation_matrix = gt_pose
                    self.transformation_matrix = transformation_matrix
                    if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                    if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                    self.add_pose(matrix=transformation_matrix)
                    return 0
            except: 
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return None
        else: 
            return None

    def set_spacing(self):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button is not None:
            actor_name = checked_button.text()
            if actor_name in self.mesh_actors:
                spacing, ok = self.input_dialog.getText(self, 'Input', "Set Spacing", text=str(self.mesh_spacing))
                if ok:
                    try: self.mesh_spacing = ast.literal_eval(spacing)
                    except: QtWidgets.QMessageBox.warning(self, 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    self.add_mesh(actor_name, self.meshdict[actor_name])
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select a mesh actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def add_workspace(self):
        workspace_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.json)")
        if workspace_path != '':
            # Clear the plot automatically if loading a new workspace
            self.clear_plot()
            self.hintLabel.hide()
            with open(str(workspace_path), 'r') as f: 
                workspace = json.load(f)

            self.image_path = workspace['image_path']
            self.mask_path = workspace['mask_path']
            self.pose_path = workspace['pose_path']
            mesh_paths = workspace['mesh_path']

            self.add_image_file(prompt=False)
            self.add_mask_file(prompt=False)
            self.add_pose_file()

            for mesh_path in mesh_paths:
                self.mesh_path = mesh_path
                self.add_mesh_file(prompt=False)

            # reset camera
            self.reset_camera()

    def load_per_frame_info(self, save=False):
        if self.video_path is not None and self.video_path != '':
            self.video_player.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.video_player.cap.read()
            if ret: 
                video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.add_image(video_frame)
                if save:
                    os.makedirs(pathlib.Path(self.video_path).parent / f"{pathlib.Path(self.video_path).stem}_vision6D", exist_ok=True)
                    os.makedirs(pathlib.Path(self.video_path).parent / f"{pathlib.Path(self.video_path).stem}_vision6D" / "frames", exist_ok=True)
                    output_frame_path = pathlib.Path(self.video_path).parent / f"{pathlib.Path(self.video_path).stem}_vision6D" / "frames" / f"frame_{self.current_frame}.png"
                    save_frame = PIL.Image.fromarray(video_frame)
                    
                    # save each frame
                    save_frame.save(output_frame_path)
                    self.output_text.append(f"-> Save frame {self.current_frame}: ({self.current_frame}/{self.video_player.frame_count})")
                    self.image_path = str(output_frame_path)

                    # save gt_pose for each frame
                    self.current_pose()
                    os.makedirs(pathlib.Path(self.video_path).parent / f"{pathlib.Path(self.video_path).stem}_vision6D" / "gt_poses", exist_ok=True)
                    output_pose_path = pathlib.Path(self.video_path).parent / f"{pathlib.Path(self.video_path).stem}_vision6D" / "gt_poses" / f"pose_{self.current_frame}.npy"
                    np.save(output_pose_path, self.transformation_matrix)
                    self.output_text.append(f"-> Save frame {self.current_frame} pose: \n{self.transformation_matrix}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
    
    def sample_video(self):
        if self.video_path != '' and self.video_path is not None:
            self.video_sampler = vis.VideoSampler(self.video_player, self.fps)
            self.video_sampler.exec_()
            self.fps = round(self.video_sampler.fps)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
    def add_video_file(self, prompt=True):
        if prompt:
            if self.video_path == None or self.video_path == '':
                self.video_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.avi *.mp4 *.mkv *.mov *.fly *.wmv *.mpeg *.asf *.webm)")
            else:
                self.video_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(pathlib.Path(self.image_path).parent), "Files (*.avi *.mp4 *.mkv *.mov *.fly *.wmv *.mpeg *.asf *.webm)")

        if self.video_path != '' and self.video_path is not None:
            self.hintLabel.hide()
            self.current_frame = 0
            self.video_player = vis.VideoPlayer(self.video_path, self.current_frame)
            self.play_video_button.setText("Play Video")
            self.load_per_frame_info(True)
            self.output_text.append(f"-> Load video {self.video_path} into vision6D")
            self.output_text.append(f"-> Current frame is ({self.current_frame}/{self.video_player.frame_count})")
            self.fps = round(self.video_player.fps)
            self.sample_video()
            
    def add_image_file(self, prompt=True):
        if prompt:
            if self.image_path == None or self.image_path == '':
                self.image_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)")
            else:
                self.image_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(pathlib.Path(self.image_path).parent), "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)")

        if self.image_path != '' and self.image_path is not None:
            self.hintLabel.hide()
            image_source = np.array(PIL.Image.open(self.image_path), dtype='uint8')
            if len(image_source.shape) == 2: image_source = image_source[..., None]
            self.add_image(image_source)
            
    def add_mask_file(self, prompt=True):
        if prompt:
            if self.mask_path == None or self.mask_path == '':
                self.mask_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)")
            else:
                self.mask_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(pathlib.Path(self.mask_path).parent), "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)")
        
        if self.mask_path != '' and self.mask_path is not None:
            self.hintLabel.hide()
            mask_source = np.array(PIL.Image.open(self.mask_path), dtype='uint8')
            if len(mask_source.shape) == 2: mask_source = mask_source[..., None]
            self.add_mask(mask_source)

    def add_mesh_file(self, prompt=True):
        if prompt:
            if self.mesh_path == None or self.mesh_path == '':
                self.mesh_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.mesh *.ply *.stl *.obj *.off *.dae *.fbx *.3ds *.x3d)")
            else:
                self.mesh_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(pathlib.Path(self.mesh_path).parent), "Files (*.mesh *.ply)")
        
        if self.mesh_path != '' and self.mesh_path is not None:
            self.hintLabel.hide()
            mesh_name = pathlib.Path(self.mesh_path).stem
            self.meshdict[mesh_name] = self.mesh_path
            self.mesh_opacity[mesh_name] = self.surface_opacity
            transformation_matrix = self.transformation_matrix
            if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix                   
            self.add_mesh(mesh_name, self.mesh_path, transformation_matrix)
                      
    def add_pose_file(self):
        if self.pose_path != '' and self.pose_path is not None:
            self.hintLabel.hide()
            transformation_matrix = np.load(self.pose_path)
            self.transformation_matrix = transformation_matrix
            if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            self.add_pose(matrix=transformation_matrix)
    
    def register_pose(self, pose):
        for actor_name, actor in self.mesh_actors.items():
            actor.user_matrix = pose
            self.plotter.add_actor(actor, pickable=True, name=actor_name)

    def reset_gt_pose(self):
        self.output_text.append(f"-> Reset the GT pose to: \n{self.initial_pose}")
        self.register_pose(self.initial_pose)

    def update_gt_pose(self):
        self.initial_pose = self.transformation_matrix
        self.current_pose()
        self.output_text.append(f"Update the GT pose to: \n{self.initial_pose}")
            
    def current_pose(self):
        if len(self.mesh_actors) == 1: self.reference = list(self.mesh_actors.keys())[0]
        if self.reference is not None:
            self.transformation_matrix = self.mesh_actors[self.reference].user_matrix
            self.output_text.append(f"-> Current reference mesh is: <span style='background-color:yellow; color:black;'>{self.reference}</span>")
            self.output_text.append(f"Current pose is: \n{self.transformation_matrix}")
            self.register_pose(self.transformation_matrix)

    def undo_pose(self):
        if self.button_group_actors_names.checkedButton() is not None:
            actor_name = self.button_group_actors_names.checkedButton().text()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Choose a mesh actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        if len(self.undo_poses[actor_name]) != 0: 
            self.transformation_matrix = self.undo_poses[actor_name].pop()
            if (self.transformation_matrix == self.mesh_actors[actor_name].user_matrix).all():
                if len(self.undo_poses[actor_name]) != 0: 
                    self.transformation_matrix = self.undo_poses[actor_name].pop()

            self.output_text.append(f"-> Current reference mesh is: <span style='background-color:yellow; color:black;'>{actor_name}</span>")
            self.output_text.append(f"Undo pose to: \n{self.transformation_matrix}")
                
            self.mesh_actors[actor_name].user_matrix = self.transformation_matrix
            self.plotter.add_actor(self.mesh_actors[actor_name], pickable=True, name=actor_name)

    def mirror_actors(self, direction):

        if direction == 'x': self.mirror_x = not self.mirror_x
        elif direction == 'y': self.mirror_y = not self.mirror_y

        #^ mirror the image actor
        if self.image_actor is not None:
            original_image_data = np.array(PIL.Image.open(self.image_path), dtype='uint8')
            if len(original_image_data.shape) == 2: original_image_data = original_image_data[..., None]
            self.add_image(original_image_data)

        #^ mirror the mask actor
        if self.mask_actor is not None:
            original_mask_data = np.array(PIL.Image.open(self.mask_path), dtype='uint8')
            if len(original_mask_data.shape) == 2: original_mask_data = original_mask_data[..., None]
            self.add_mask(original_mask_data)

        #^ mirror the mesh actors
        if len(self.mesh_actors) != 0:
            for actor_name, _ in self.mesh_actors.items():
                transformation_matrix = self.transformation_matrix
                if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                self.add_mesh(actor_name, self.meshdict[actor_name], transformation_matrix)
            
            # Output the mirrored transformation matrix
            self.output_text.append(f"-> Mirrored transformation matrix is: \n{transformation_matrix}")
             
    def remove_actor(self, button):
        name = button.text()
        if name == 'image': 
            actor = self.image_actor
            self.image_actor = None
            self.image_path = None
        elif name == 'mask':
            actor = self.mask_actor
            self.mask_actor = None
            self.mask_path = None
        elif name in self.mesh_actors: 
            actor = self.mesh_actors[name]
            del self.mesh_actors[name] # remove the item from the mesh dictionary
            del self.mesh_colors[name]
            del self.mesh_opacity[name]
            del self.meshdict[name]
            self.reference = None
            self.color_button.setText("Color")
            self.mesh_spacing = [1, 1, 1]

        self.plotter.remove_actor(actor)
        self.track_actors_names.remove(name)
        self.output_text.append(f"-> Remove actor: {name}")
        # remove the button from the button group
        self.button_group_actors_names.removeButton(button)
        # remove the button from the self.button_layout widget
        self.button_layout.removeWidget(button)
        # offically delete the button
        button.deleteLater()

        # clear out the plot if there is no actor
        if self.image_actor is None and self.mask_actor is None and len(self.mesh_actors) == 0: self.clear_plot()
   
    def clear_plot(self):
        
        # Clear out everything in the remove menu
        for button in self.button_group_actors_names.buttons():
            name = button.text()
            if name == 'image': actor = self.image_actor
            elif name == 'mask': actor = self.mask_actor
            elif name in self.mesh_actors: actor = self.mesh_actors[name]
            self.plotter.remove_actor(actor)
            # remove the button from the button group
            self.button_group_actors_names.removeButton(button)
            # remove the button from the self.button_layout widget
            self.button_layout.removeWidget(button)
            # offically delete the button
            button.deleteLater()

        self.hintLabel.show()

        # Re-initial the dictionaries
        self.video_path = None
        self.play_video_button.setText("Play Video")
        self.image_path = None
        self.mask_path = None
        self.mesh_path = None
        self.pose_path = None
        self.meshdict = {}
        self.mesh_colors = {}

        # reset everything to original actor opacity
        self.mesh_opacity = {}
        self.store_mesh_opacity = {}
        self.image_opacity = 0.8
        self.mask_opacity = 0.1
        self.surface_opacity = self.opacity_spinbox.value()

        self.image_spacing = [0.01, 0.01, 1]
        self.mask_spacing = [0.01, 0.01, 1]
        self.mesh_spacing = [1, 1, 1]

        self.mirror_x = False
        self.mirror_y = False

        self.reference = None
        self.transformation_matrix = np.eye(4)
        self.image_actor = None
        self.mask_actor = None
        self.mesh_actors = {}
        self.undo_poses = {}
        self.track_actors_names = []

        self.colors = ["cyan", "magenta", "yellow", "lime", "deepskyblue", "salmon", "silver", "aquamarine", "plum", "blueviolet"]
        self.used_colors = []
        self.color_button.setText("Color")

        self.clear_output_text()

    def render_image(self, actor, camera):
        self.render.clear()
        render_actor = actor.copy(deep=True)
        render_actor.GetProperty().opacity = 1
        self.render.add_actor(render_actor, pickable=False)
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        image = self.render.last_image
        return image
    
    def render_mesh(self, render_all_meshes, camera, point_clouds):
        self.render.clear()
        for mesh_name, mesh_actor in self.mesh_actors.items():
            if not render_all_meshes:
                if mesh_name != self.reference: continue
            vertices, faces = vis.utils.get_mesh_actor_vertices_faces(mesh_actor)
            mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
            colors = vis.utils.get_mesh_actor_scalars(mesh_actor)
            if colors is not None: 
                assert colors.shape == vertices.shape, "colors shape should be the same as vertices shape"
                mesh = self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=1, name=mesh_name) if not point_clouds else self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='points', point_size=1, render_points_as_spheres=False, opacity=1, name=mesh_name)
            else:
                mesh = self.render.add_mesh(mesh_data, color=self.mesh_colors[mesh_name], style='surface', opacity=1, name=mesh_name) if not point_clouds else self.render.add_mesh(mesh_data, color=self.mesh_colors[mesh_name], style='points', point_size=1, render_points_as_spheres=False, opacity=1, name=mesh_name)
            mesh.user_matrix = self.mesh_actors[self.reference].user_matrix
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        image = self.render.last_image
        return image

    def export_image_plot(self):

        if self.image_actor is None:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
        reply = QtWidgets.QMessageBox.question(self,"vision6D", "Reset Camera?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes: camera = self.camera.copy()
        else: camera = self.plotter.camera.copy()

        image = self.render_image(self.image_actor, camera)
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Image Files (*.png)")
        if output_path:
            if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
            rendered_image = PIL.Image.fromarray(image)
            rendered_image.save(output_path)
            self.output_text.append(f"-> Export image render to:\n {str(output_path)}")

    def export_mask_plot(self):
        if self.mask_actor is None:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a mask first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
        reply = QtWidgets.QMessageBox.question(self,"vision6D", "Reset Camera?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes: camera = self.camera.copy()
        else: camera = self.plotter.camera.copy()

        image = self.render_image(self.mask_actor, camera)
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Mask Files (*.png)")
        if output_path:
            if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
            rendered_image = PIL.Image.fromarray(image)
            rendered_image.save(output_path)
            self.output_text.append(f"-> Export mask render to:\n {str(output_path)}")

    def export_mesh_plot(self, reply_reset_camera=None, reply_render_mesh=None, reply_export_surface=None, save_render=True):

        if self.reference is None:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

        if reply_reset_camera is None and reply_render_mesh is None and reply_export_surface is None:
            reply_reset_camera = QtWidgets.QMessageBox.question(self,"vision6D", "Reset Camera?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            reply_render_mesh = QtWidgets.QMessageBox.question(self,"vision6D", "Only render the reference mesh?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            reply_export_surface = QtWidgets.QMessageBox.question(self,"vision6D", "Export the mesh as surface?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            
        if reply_reset_camera == QtWidgets.QMessageBox.Yes: camera = self.camera.copy()
        else: camera = self.plotter.camera.copy()
        if reply_render_mesh == QtWidgets.QMessageBox.No: render_all_meshes = True
        else: render_all_meshes = False
        if reply_export_surface == QtWidgets.QMessageBox.No: point_clouds = True
        else: point_clouds = False
        
        image = self.render_mesh(render_all_meshes, camera, point_clouds)
        
        if save_render:
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Mesh Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.output_text.append(f"-> Export reference mesh render to:\n {str(output_path)}")

        return image

    def export_segmesh_plot(self):

        if self.reference is None:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
        if self.mask_actor is None: 
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a segmentation mask first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

        reply_reset_camera = QtWidgets.QMessageBox.question(self,"vision6D", "Reset Camera?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        reply_export_surface = QtWidgets.QMessageBox.question(self,"vision6D", "Export the mesh as surface?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply_reset_camera == QtWidgets.QMessageBox.Yes: camera = self.camera.copy()
        else: camera = self.plotter.camera.copy()
        if reply_export_surface == QtWidgets.QMessageBox.No: point_clouds = True
        else: point_clouds = False

        segmask = self.render_image(self.mask_actor, camera)
        if np.max(segmask) > 1: segmask = segmask / 255
        image = self.render_mesh(False, camera, point_clouds)
        image = (image * segmask).astype(np.uint8)
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "SegMesh Files (*.png)")
        if output_path:
            if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
            rendered_image = PIL.Image.fromarray(image)
            rendered_image.save(output_path)
            self.output_text.append(f"-> Export segmask render:\n to {str(output_path)}")
        
    def export_pose(self):
        if self.reference is None: 
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        self.update_gt_pose()
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Pose Files (*.npy)")
        if output_path:
            if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.npy')
            np.save(output_path, self.transformation_matrix)
            self.output_text.append(f"-> Saved:\n{self.transformation_matrix}\nExport to:\n {str(output_path)}")

    def copy_output_text(self):
        self.clipboard.setText(self.output_text.toPlainText())
        
    def clear_output_text(self):
        self.output_text.clear()

    def add_button_actor_name(self, actor_name):
        button = QtWidgets.QPushButton(actor_name)
        button.setCheckable(True)  # Set the button to be checkable
        button.clicked.connect(lambda _, text=actor_name: self.button_actor_name_clicked(text))
        button.setChecked(True)
        button.setFixedSize(self.display.size().width(), 50)
        self.button_layout.insertWidget(0, button) # insert from the top # self.button_layout.addWidget(button)
        self.button_group_actors_names.addButton(button)
        self.button_actor_name_clicked(actor_name)

    def update_color_button_text(self, text, popup):
        self.color_button.setText(text)
        popup.close() # automatically close the popup window

    def show_color_popup(self):

        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button is None:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

        actor_name = checked_button.text()

        if actor_name not in self.mesh_actors:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Only be able to color mesh actors", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

        popup = vis.PopUpDialog(self, on_button_click=lambda text: self.update_color_button_text(text, popup))
        button_position = self.color_button.mapToGlobal(QPoint(0, 0))
        popup.move(button_position + QPoint(self.color_button.width(), 0))
        popup.exec_()

        text = self.color_button.text()
        self.mesh_colors[actor_name] = text
        if text == 'nocs': self.set_scalar(True, actor_name)
        elif text == 'latlon': self.set_scalar(False, actor_name)
        else: self.set_color(text, actor_name)

    def remove_actors_button(self):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button is None: 
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else: self.remove_actor(checked_button)

    def opacity_value_change(self, value):
        if self.ignore_spinbox_value_change: return 0
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button is not None:
            actor_name = checked_button.text()
            if actor_name == 'image': 
                self.set_image_opacity(value)
            elif actor_name == 'mask': 
                self.set_mask_opacity(value)
            elif actor_name in self.mesh_actors: 
                self.store_mesh_opacity[actor_name] = copy.deepcopy(self.mesh_opacity[actor_name])
                self.mesh_opacity[actor_name] = value
                self.set_mesh_opacity(actor_name, self.mesh_opacity[actor_name])
        else:
            self.ignore_spinbox_value_change = True
            self.opacity_spinbox.setValue(value)
            self.ignore_spinbox_value_change = False
            return 0
        
    def toggle_hide_meshes_button(self):
        self.toggle_hide_meshes_flag = not self.toggle_hide_meshes_flag
        
        if self.toggle_hide_meshes_flag:
            for button in self.button_group_actors_names.buttons():
                if button.text() in self.mesh_actors:
                    button.setChecked(True); self.opacity_value_change(0)
    
            checked_button = self.button_group_actors_names.checkedButton()
            if checked_button is not None: 
                self.ignore_spinbox_value_change = True
                self.opacity_spinbox.setValue(0.0)
                self.ignore_spinbox_value_change = False
            else: QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        
        else:
            for button in self.button_group_actors_names.buttons():
                if button.text() in self.mesh_actors:
                    button.setChecked(True); self.opacity_value_change(self.store_mesh_opacity[button.text()])

            checked_button = self.button_group_actors_names.checkedButton()
            if checked_button is not None:
                self.ignore_spinbox_value_change = True
                if checked_button.text() in self.mesh_actors: self.opacity_spinbox.setValue(self.mesh_opacity[checked_button.text()])
                self.ignore_spinbox_value_change = False
            else: QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def play_video(self):
        if self.video_path != None and self.video_path != '':
            self.video_player.exec_()
            self.current_frame = self.video_player.current_frame
            self.output_text.append(f"-> Current frame is ({self.current_frame}/{self.video_player.frame_count})")
            self.play_video_button.setText(f"Play ({self.current_frame}/{self.video_player.frame_count})")
            self.load_per_frame_info()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
    
    def prev_frame(self):
        if self.video_path != None and self.video_path != '':
            current_frame = self.video_player.current_frame - self.fps
            if current_frame >= 0: self.current_frame = current_frame
            self.output_text.append(f"-> Current frame is ({self.current_frame}/{self.video_player.frame_count})")
            pose_path = pathlib.Path(self.video_path).parent / f"{pathlib.Path(self.video_path).stem}_vision6D" / "gt_poses" / f"pose_{self.current_frame}.npy"
            if os.path.isfile(pose_path): 
                self.transformation_matrix = np.load(pose_path)
                self.register_pose(self.transformation_matrix)
                self.output_text.append(f"-> Load saved frame {self.current_frame} pose: \n{self.transformation_matrix}")
            else: self.output_text.append(f"-> No saved pose for frame {self.current_frame}")
            self.play_video_button.setText(f"Play ({self.current_frame}/{self.video_player.frame_count})")
            self.video_player.slider.setValue(self.current_frame)
            self.load_per_frame_info()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def next_frame(self):
        if self.video_path != None and self.video_path != '':
            # save pose from the previous frame 
            self.load_per_frame_info(save=True)
            current_frame = self.video_player.current_frame + self.fps
            if current_frame <= self.video_player.frame_count: self.current_frame = current_frame
            self.output_text.append(f"-> Current frame is ({self.current_frame}/{self.video_player.frame_count})")
            # load pose for the current frame if the pose file exist
            pose_path = pathlib.Path(self.video_path).parent / f"{pathlib.Path(self.video_path).stem}_vision6D" / "gt_poses" / f"pose_{self.current_frame}.npy"
            if os.path.isfile(pose_path): 
                self.transformation_matrix = np.load(pose_path)
                self.register_pose(self.transformation_matrix)
                self.output_text.append(f"-> Load saved frame {self.current_frame} pose: \n{self.transformation_matrix}")
            self.play_video_button.setText(f"Play ({self.current_frame}/{self.video_player.frame_count})")
            self.video_player.slider.setValue(self.current_frame)
            self.load_per_frame_info()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def delete_video(self):
        self.output_text.append(f"-> Delete video {self.video_path} from vision6D")
        self.play_video_button.setText("Play Video")
        self.video_path = None
        self.current_frame = 0