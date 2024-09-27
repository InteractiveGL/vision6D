import pyvista as pv
import numpy as np
import math
from . import Singleton

class CameraStore(metaclass=Singleton):
    def __init__(self, plotter):
        self.plotter = plotter
        self.camera = pv.Camera() # camera related parameters, do not reset the camera!

    def reset_camera(self):
        self.plotter.camera = self.camera.copy()

    def zoom_in(self):
        self.plotter.camera.zoom(2)

    def zoom_out(self):
        self.plotter.camera.zoom(0.5)

    def set_camera_extrinsics(self, cam_viewup):
        self.camera.SetPosition((0, 0, -1e-8)) # Set the camera position at the origin of the world coordinate system
        self.camera.SetFocalPoint((0, 0, 0)) # Get the camera window center
        self.camera.SetViewUp(cam_viewup)
    
    def set_camera_intrinsics(self, fx, fy, cx, cy, height):
        # Set camera intrinsic attribute
        self.camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        view_angle = (180 / math.pi) * (2.0 * math.atan2(height/2.0, fy)) # or view_angle = np.degrees(2.0 * math.atan2(height/2.0, f)) # focal_length = (height / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
        self.camera.SetViewAngle(view_angle) # view angle should be in degrees