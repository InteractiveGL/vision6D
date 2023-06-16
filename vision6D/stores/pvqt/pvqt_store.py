import os
import ast
import math

import numpy as np
import pyvista as pv
import vision6D as vis

# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"

from ..singleton import Singleton
from .plot_store import PlotStore
from .camera_store import CameraStore
from .image_store import ImageStore
from .mask_store import MaskStore
from .mesh_store import MeshStore
from .video_store import VideoStore

np.set_printoptions(suppress=True)

class PvQtStore(metaclass=Singleton):

    def __init__(self, signal_close):
        super().__init__()

        # Saving parameters
        self.plot_store = PlotStore(signal_close)
        self.camera_store = CameraStore()
        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.mesh_store = MeshStore()

        # Utilities
        self.video_store = VideoStore()
        
        
    def show_plot(self):
        self.plotter.enable_joystick_actor_style()
        self.plotter.enable_trackball_actor_style()
        
        self.plotter.add_axes()
        self.plotter.add_camera_orientation_widget()

        self.plotter.show()

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

    def update_opacity(self, attr_name, change):
        attr = getattr(self, attr_name)
        setattr(self, attr_name, np.clip(attr + change, 0, 1))

    def reset(self):

        self.track_actors_names = []

        # Reset stores
        self.camera_store.reset()
        self.image_store.reset()
        self.mask_store.reset()
        self.mesh_store.reset()
