'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: camera_store.py
@time: 2023-07-03 20:22
@desc: create store for camera related base functions
'''

import math
import pyvista as pv
import numpy as np
from . import Singleton

class CameraStore(metaclass=Singleton):
    def __init__(self, window_size):
        self.window_size = window_size
        self.reset()
        
    def reset(self):
        self.camera = pv.Camera()
        self.fx = 10000
        self.fy = 10000
        self.cx = self.window_size[0] // 2
        self.cy = self.window_size[1] // 2
        self.cam_viewup = (0, 0, 0)

    def set_camera_extrinsics(self):
        self.camera.SetPosition((0, 0, -1e-8)) # Set the camera position at the origin of the world coordinate system
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


    