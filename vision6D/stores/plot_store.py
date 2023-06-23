import math

import pyvista as pv
import numpy as np

from ..widgets import CustomQtInteractor
from .singleton import Singleton
from .. import utils

class PlotStore(metaclass=Singleton):

    plotter: CustomQtInteractor

    def __init__(self, frame, window_size):

        self.frame = frame
        self.window_size = window_size
        self.frame.setFixedSize(*self.window_size)
        self.plotter = CustomQtInteractor(self.frame)
        utils.create_render(self.window_size[0], self.window_size[1])
        # Initialize
        self.reset()

    def reset(self):
        
        self.mirror_x = False
        self.mirror_y = False

        # Set the camera
        self.camera = pv.Camera()
        self.fx = 50000
        self.fy = 50000
        self.cx = 960
        self.cy = 540
        self.cam_viewup = (0, -1, 0)
        self.cam_position = -500
        self.set_camera_props()

    def show_plot(self):
        self.plotter.enable_joystick_actor_style()
        self.plotter.enable_trackball_actor_style()
        self.plotter.add_axes()
        self.plotter.add_camera_orientation_widget()
        self.plotter.show()

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
 
    def reset_camera(self):
        self.plotter.camera = self.camera.copy()

    def zoom_in(self):
        self.plotter.camera.zoom(2)

    def zoom_out(self):
        self.plotter.camera.zoom(0.5)

    