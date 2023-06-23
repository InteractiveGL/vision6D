from functools import partial
import PIL.Image
import numpy as np
from PyQt5 import QtWidgets

from ...stores import QtStore
from ...stores import PvQtStore
from ...stores import FolderStore
from ...stores import WorkspaceStore
from ...stores import PlotStore
from ...stores import VideoStore
from ...stores import ImageStore
from ...stores import MaskStore
from ...stores import MeshStore

class FileMenu():

    def __init__(self):

        # Create references to stores
        self.qt_store = QtStore()
        self.pvqt_store = PvQtStore()
        self.folder_store = FolderStore()
        self.workspace_store = WorkspaceStore()

        self.plot_store = PlotStore()
        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.mesh_store = MeshStore()
        self.video_store = VideoStore()

    def add_workspace(self, prompt=False):
        if prompt:
            workspace_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.json)")
        if workspace_path:
            self.qt_store.hintLabel.hide()
            self.workspace_store.add_workspace(workspace_path)
            if self.video_store.video_path:
                self.qt_store.output_text.append(f"-> Load video {self.video_store.video_path} into vision6D")
                self.qt_store.output_text.append(f"-> Current frame is ({self.video_store.current_frame}/{self.video_store.total_frame})")

            # reset camera
            self.plot_store.reset_camera()

    def add_folder(self, prompt=False):
        if prompt: 
            folder_path = QtWidgets.QFileDialog().getExistingDirectory(self, "Select Folder")
            flag = self.folder_store.add_folder(folder_path)
            if flag:
                self.folder_store.reset()
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Not a valid folder, please reload a folder", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return 0
            else:
                self.qt_store.hintLabel.hide()
                self.video_store.video_path = None
                self.qt_store.output_text.append(f"-> After reset GT pose, current slide is ({self.current_frame}/{self.total_count})")
                self.plot_store.reset_camera()

    def add_video(self, prompt=False):
        if prompt:
            video_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.avi *.mp4 *.mkv *.mov *.fly *.wmv *.mpeg *.asf *.webm)")
        if video_path:
            self.qt_store.hintLabel.hide()
            self.folder_store.folder_path = None # make sure video_path and folder_path are exclusive
            self.video_store.add_video(video_path)
            video_frame = self.video_store.load_per_frame_info()
            self.image_store.add_image(video_frame)
            self.qt_store.output_text.append(f"-> Load video {self.video_store.video_path} into vision6D")
            self.qt_store.output_text.append(f"-> Current frame is ({self.video_store.current_frame}/{self.video_store.total_frame})")
    
    def add_image(self, prompt=False):
        if prompt:
            image_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)")
        if image_path:
            self.qt_store.hintLabel.hide()
            self.image_store.add_image(image_path)
            # add remove current image to removeMenu
            if 'image' not in self.pvqt_store.track_actors_names:
                self.pvqt_store.track_actors_names.append('image')
                self.pvqt_store.add_button_actor_name('image')
            self.pvqt_store.check_button('image')
            
    def add_mask(self, prompt=False):
        if prompt:
            mask_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)") 
        if mask_path:
            self.qt_store.hintLabel.hide()
            self.mask_store.add_mask(mask_path)
            # Add remove current image to removeMenu
            if 'mask' not in self.pvqt_store.track_actors_names:
                self.pvqt_store.track_actors_names.append('mask')
                self.pvqt_store.add_button_actor_name('mask')
            self.pvqt_store.check_button('mask')

    def add_mesh(self, prompt=False):
        if prompt: 
            mesh_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.mesh *.ply *.stl *.obj *.off *.dae *.fbx *.3ds *.x3d)") 
        if mesh_path:
            self.qt_store.hintLabel.hide()
            # TODO
            transformation_matrix = self.mesh_store.transformation_matrix
            if self.plot_store.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            if self.plot_store.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix                   
            self.mesh_store.add_mesh(mesh_path, transformation_matrix)
            # add remove current mesh to removeMenu
            if self.mesh_store.mesh_name:
                if self.mesh_store.mesh_name not in self.pvqt_store.track_actors_names:
                    self.pvqt_store.track_actors_names.append(self.mesh_store.mesh_name)
                    self.pvqt_store.add_button_actor_name(self.mesh_store.mesh_name)
                self.pvqt_store.check_button(self.mesh_store.mesh_name)

    def draw_mask(self):
        if self.image_store.image_path:
            self.mask_store.draw_mask(self.image_store.image_path)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
    def remove_actor(self, button):
        name = button.text()
        if name == 'image': actor = self.image_store.remove_actor()
        elif name == 'mask': actor =self.mask_store.remove_actor()
        else: 
            actor = self.mesh_store.remove_actor(name)
            self.pvqt_store.color_button.setText("Color")
        
        if actor: self.mesh_store.track_actors_names.remove(name)

        # clear out the plot if there is no actor
        if len(self.pvqt_store.button_group_actors_names.buttons()) == 0: self.clear_plot()

        self.pvqt_store.remove_actor_button(name, button)
        
    def clear_plot(self):
        # Clear out everything in the remove menu
        for button in self.pvqt_store.button_group_actors_names.buttons():
            self.remove_actor(button)

        self.qt_store.hintLabel.show()
        self.qt_store.output_text.clear()
        self.pvqt_store.color_button.setText("Color")

        self.pvqt_store.track_actors_names = []
        # Reset stores
        self.plot_store.reset()
        self.image_store.reset()
        self.mask_store.reset()
        self.mesh_store.reset()
        self.folder_store.reset()
        self.video_store.reset()
        self.workspace_store.reset()