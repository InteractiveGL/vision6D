import math
import ast

import numpy as np
import pyvista as pv

from ..singleton import Singleton
from .plot_store import PlotStore

class CameraStore(metaclass=Singleton):

    def __init__(self):

        self.plot_store = PlotStore()

        self.reset()
        
    def reset(self):
        self.transformation_matrix = np.eye(4)
        self.initial_pose = self.transformation_matrix
        self.undo_poses = {}

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
        wcx = -2*(self.cx - float(self.plot_store.window_size[0])/2) / self.plot_store.window_size[0]
        wcy =  2*(self.cy - float(self.plot_store.window_size[1])/2) / self.plot_store.window_size[1]
        self.camera.SetWindowCenter(wcx, wcy) # (0,0)
        
        # Setting the view angle in degrees
        view_angle = (180 / math.pi) * (2.0 * math.atan2(self.plot_store.window_size[1]/2.0, self.fx)) # or view_angle = np.degrees(2.0 * math.atan2(height/2.0, f)) # focal_length = (1080 / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
        self.camera.SetViewAngle(view_angle) # view angle should be in degrees
 
    def set_camera_props(self):
        self.set_camera_intrinsics()
        self.set_camera_extrinsics()
        self.plot_store.plotter.camera = self.camera.copy()

    def set_camera(self, fx, fy, cx, cy, cam_viewup, cam_position):
        pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup, pre_cam_position = self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position
        self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position = ast.literal_eval(fx), ast.literal_eval(fy), ast.literal_eval(cx), ast.literal_eval(cy), ast.literal_eval(cam_viewup), ast.literal_eval(cam_position)
        try:
            self.set_camera_props()
            return True
        except:
            self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position = pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup, pre_cam_position
            return False
    
    def add_pose(self, matrix:np.ndarray=None, rot:np.ndarray=None, trans:np.ndarray=None):
        if matrix is not None: 
            self.initial_pose = matrix
            self.reset_gt_pose()
            self.reset_camera()
        else:
            if (rot and trans): matrix = np.vstack((np.hstack((rot, trans)), [0, 0, 0, 1]))

    def set_transformation_matrix(self, transformation_matrix):
        self.transformation_matrix = transformation_matrix
        if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        self.add_pose(matrix=transformation_matrix)

    def mirror_transformation_matrix(self):
        transformation_matrix = self.transformation_matrix
        if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        return transformation_matrix

    def register_pose(self, pose):
        for actor_name, actor in self.mesh_actors.items():
            actor.user_matrix = pose
            self.plot_store.add_actor(actor, pickable=True, name=actor_name)

    def current_pose(self):
        if len(self.mesh_actors) == 1: self.reference = list(self.mesh_actors.keys())[0]
        if self.reference:
            self.transformation_matrix = self.mesh_actors[self.reference].user_matrix
            self.register_pose(self.transformation_matrix)
            return True
        
        return False