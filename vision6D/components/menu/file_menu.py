from functools import partial
import json
import pathlib
import os
import re

import cv2
import trimesh
import pyvista as pv
import PIL.Image
import numpy as np
from PyQt5 import QtWidgets

from ... import utils
from ...widgets import VideoPlayer
from ...stores import MainStore, QtStore

class FileMenu():

    def __init__(self, menu):

        # Create references to stores
        self.qt_store = QtStore()

        # Save parameter
        self.menu = menu
        self.file_dialog = QtWidgets.QFileDialog()

        self.menu.addAction('Add Workspace', partial(self.add_workspace, prompt=True))
        self.menu.addAction('Add Folder', partial(self.add_folder, prompt=True))
        self.menu.addAction('Add Video', partial(self.add_video_file, prompt=True))
        self.menu.addAction('Add Image', partial(self.add_image_file, prompt=True))
        self.menu.addAction('Add Mask', partial(self.add_mask_file, prompt=True))
        self.menu.addAction('Add Mesh', partial(self.add_mesh_file, prompt=True))
        # self.menu.addAction('Draw Mask', self.draw_mask)
        # self.menu.addAction('Clear', self.clear_plot)

    def add_workspace(self, prompt=False):
        if prompt:
            self.workspace_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.json)")
        if self.workspace_path:
            self.hintLabel.hide()
            with open(str(self.workspace_path), 'r') as f: 
                workspace = json.load(f)

            if 'image_path' in workspace:
                self.image_path = workspace['image_path']
                self.add_image_file()
            if 'video_path' in workspace:
                self.video_path = workspace['video_path']
                self.add_video_file()
            if 'mask_path' in workspace:
                self.mask_path = workspace['mask_path']
                self.add_mask_file()
            if 'pose_path' in workspace:
                self.pose_path = workspace['pose_path']
                self.add_pose_file()
            if 'mesh_path' in workspace:
                mesh_path = workspace['mesh_path']
                for path in mesh_path:
                    self.mesh_path = path
                    self.add_mesh_file()
            
            # reset camera
            self.reset_camera()
    
    def get_files_from_folder(self, category):
        dir = pathlib.Path(self.folder_path) / category
        folders = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        if len(folders) == 1: dir = pathlib.Path(self.folder_path) / category / folders[0]
        # Retrieve files
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        self.total_count = len(files)
        # Sort files
        files.sort(key=lambda f: int(re.sub('\D', '', f)))
        return files, dir

    def add_folder(self, prompt=False):
        if prompt: 
            self.folder_path = self.file_dialog.getExistingDirectory(self, "Select Folder")
        
        if self.folder_path:
            folders = [d for d in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, d))]
            flag = True

            if 'images' in folders:
                flag = False
                image_files, image_dir = self.get_files_from_folder('images')
                self.image_path = str(image_dir / image_files[self.current_frame])
                if os.path.isfile(self.image_path): self.add_image_file()

            if 'masks' in folders:
                flag = False
                mask_files, mask_dir = self.get_files_from_folder('masks')
                self.mask_path = str(mask_dir / mask_files[self.current_frame])
                if os.path.isfile(self.mask_path): self.add_mask_file()
                    
            if 'poses' in folders:
                flag = False
                pose_files, pose_dir = self.get_files_from_folder('poses')
                self.pose_path = str(pose_dir / pose_files[self.current_frame])
                if os.path.isfile(self.pose_path): self.add_pose_file()
                    
            if self.current_frame == 0:
                if 'meshes' in folders:
                    flag = False
                    dir = pathlib.Path(self.folder_path) / "meshes"
                    if os.path.isfile(dir / 'mesh_path.txt'):
                        with open(dir / 'mesh_path.txt', 'r') as f: mesh_path = f.read().splitlines()
                        for path in mesh_path:
                            self.mesh_path = path
                            self.add_mesh_file()

            if flag:
                self.delete_video_folder()
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Not a valid folder, please reload a folder", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return 0
            else:
                self.video_path = None # make sure video_path and folder_path are exclusive
                self.output_text.append(f"-> After reset GT pose, current slide is ({self.current_frame}/{self.total_count})")
                self.reset_camera()

    def add_video_file(self, prompt=False):
        if prompt:
            self.video_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.avi *.mp4 *.mkv *.mov *.fly *.wmv *.mpeg *.asf *.webm)")
        if self.video_path:
            self.hintLabel.hide()
            self.folder_path = None # make sure video_path and folder_path are exclusive
            self.video_player = VideoPlayer(self.video_path, self.current_frame)
            self.play_video_button.setText("Play Video")
            self.output_text.append(f"-> Load video {self.video_path} into vision6D")
            self.output_text.append(f"-> Current frame is ({self.current_frame}/{self.video_player.frame_count})")
            self.fps = round(self.video_player.fps)
            self.load_per_frame_info(True)
            self.sample_video()
    
    def add_image(self, image_source):

        if isinstance(image_source, pathlib.WindowsPath) or isinstance(image_source, str):
            image_source = np.array(PIL.Image.open(image_source), dtype='uint8')
        if len(image_source.shape) == 2: image_source = image_source[..., None]

        if self.mirror_x: image_source = image_source[:, ::-1, :]
        if self.mirror_y: image_source = image_source[::-1, :, :]

        dim = image_source.shape
        h, w, channel = dim[0], dim[1], dim[2]

        # Create the render based on the image size
        self.render = pv.Plotter(window_size=[w, h], lighting=None, off_screen=True) 
        self.render.set_background('black'); assert self.render.background_color == "black", "render's background need to be black"

        image = pv.UniformGrid(dimensions=(w, h, 1), spacing=[0.01, 0.01, 1], origin=(0.0, 0.0, 0.0))
        image.point_data["values"] = image_source.reshape((w * h, channel)) # order = 'C
        image = image.translate(-1 * np.array(image.center), inplace=False)

        # Then add it to the plotter
        image = self.plotter.add_mesh(image, cmap='gray', opacity=self.image_opacity, name='image') if channel == 1 else self.plotter.add_mesh(image, rgb=True, opacity=self.image_opacity, name='image')
        actor, _ = self.plotter.add_actor(image, pickable=False, name='image')
        # Save actor for later
        self.image_actor = actor
        
        # get the image scalar
        image_data = utils.get_image_actor_scalars(self.image_actor)
        assert (image_data == image_source).all() or (image_data*255 == image_source).all(), "image_data and image_source should be equal"

        # add remove current image to removeMenu
        if 'image' not in self.track_actors_names:
            self.track_actors_names.append('image')
            self.add_button_actor_name('image')

        self.qt_store.check_button('image')

    def add_mesh(self, mesh_name, mesh_source, transformation_matrix=None):
        """ add a mesh to the pyqt frame """

        flag = False
                              
        if isinstance(mesh_source, pathlib.WindowsPath) or isinstance(mesh_source, str):
            # Load the '.mesh' file
            if pathlib.Path(mesh_source).suffix == '.mesh': mesh_source = utils.load_trimesh(mesh_source)
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
        self.qt_store.color_button.setText(self.mesh_colors[mesh_name])
        mesh = self.plotter.add_mesh(mesh_data, color=mesh_color, opacity=self.mesh_opacity[mesh_name], name=mesh_name)

        mesh.user_matrix = self.transformation_matrix if transformation_matrix is None else transformation_matrix
        self.initial_pose = mesh.user_matrix
                
        # Add and save the actor
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=mesh_name)

        actor_vertices, actor_faces = utils.get_mesh_actor_vertices_faces(actor)
        assert (actor_vertices == source_verts).all(), "vertices should be the same"
        assert (actor_faces == source_faces).all(), "faces should be the same"
        assert actor.name == mesh_name, "actor's name should equal to mesh_name"
        
        self.mesh_actors[mesh_name] = actor

        # add remove current mesh to removeMenu
        if mesh_name not in self.track_actors_names:
            self.track_actors_names.append(mesh_name)
            self.add_button_actor_name(mesh_name)

        self.qt_store.check_button(mesh_name)
            
    def add_image_file(self, prompt=False):
        if prompt:
            self.image_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)")
        if self.image_path:
            self.hintLabel.hide()
            image_source = np.array(PIL.Image.open(self.image_path), dtype='uint8')
            if len(image_source.shape) == 2: image_source = image_source[..., None]
            self.add_image(image_source)
            
    def add_mask_file(self, prompt=False):
        if prompt:
            self.mask_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)") 
        if self.mask_path:
            self.hintLabel.hide()
            mask_source = np.array(PIL.Image.open(self.mask_path), dtype='uint8')
            self.add_mask(mask_source)

    def add_mesh_file(self, prompt=False):
        if prompt: 
            self.mesh_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.mesh *.ply *.stl *.obj *.off *.dae *.fbx *.3ds *.x3d)") 
        if self.mesh_path:
            self.hintLabel.hide()
            mesh_name = pathlib.Path(self.mesh_path).stem
            self.meshdict[mesh_name] = self.mesh_path
            self.mesh_opacity[mesh_name] = self.surface_opacity
            transformation_matrix = self.transformation_matrix
            if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix                   
            self.add_mesh(mesh_name, self.mesh_path, transformation_matrix)

    def clear_plot(self):
        MainStore().clear_plot()
