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
from ...stores import MainStore, QtStore, PvQtStore

class FileMenu():

    def __init__(self, menu):

        # Create references to stores
        self.qt_store = QtStore()
        self.pvqt_store = PvQtStore()

        # Save parameter
        self.menu = menu
        self.file_dialog = QtWidgets.QFileDialog()

        self.menu.addAction('Add Workspace', partial(self.pvqt_store.add_workspace, prompt=True))
        self.menu.addAction('Add Folder', partial(self.add_folder, prompt=True))
        self.menu.addAction('Add Video', partial(self.add_video_file, prompt=True))
        self.menu.addAction('Add Image', partial(self.add_image_file, prompt=True))
        self.menu.addAction('Add Mask', partial(self.add_mask_file, prompt=True))
        self.menu.addAction('Add Mesh', partial(self.add_mesh_file, prompt=True))
        self.menu.addAction('Draw Mask', self.draw_mask)
        self.menu.addAction('Clear', self.clear_plot)
    
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
        self.pvqt_store.image_store.add_image(image_source)
        # add remove current image to removeMenu
        if 'image' not in self.pvqt_store.track_actors_names:
            self.pvqt_store.track_actors_names.append('image')
            self.qt_store.add_button_actor_name('image')

        self.qt_store.check_button('image')

    def add_mask(self, mask_source):
        self.pvqt_store.mask_store.add_mask(mask_source)
        # Add remove current image to removeMenu
        if 'mask' not in self.track_actors_names:
            self.pvqt_store.track_actors_names.append('mask')
            self.qt_store.add_button_actor_name('mask')
        self.qt_store.check_button('mask')

    def add_mesh(self, mesh_name, mesh_source, transformation_matrix=None):
        """ add a mesh to the pyqt frame """
        self.pvqt_store.mesh_store.add_mesh(mesh_source, transformation_matrix)

        # add remove current mesh to removeMenu
        if mesh_name not in self.pvqt_store.track_actors_names:
            self.pvqt_store.track_actors_names.append(mesh_name)
            self.qt_store.add_button_actor_name(mesh_name)

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

    def draw_mask(self):
        self.pvqt_store.mask_store.draw_mask()

    def clear_plot(self):
        MainStore().clear_plot()
