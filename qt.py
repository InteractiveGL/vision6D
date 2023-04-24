import sys
from typing import List, Dict, Tuple
import pathlib
import logging
import numpy as np
import math
import trimesh
import functools
import numpy as np
import PIL

# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow
import vision6D as vis

class MyMainWindow(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)
        
        # setting title
        self.setWindowTitle("Vision6D")
        self.showMaximized()
        # QtWidgets.QWidget.resize(self, 1920, 1080)
        
        # create the frame
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout()

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        vlayout.addWidget(self.plotter.interactor)
        self.signal_close.connect(self.plotter.close)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()

        # allow to add files
        fileMenu = mainMenu.addMenu('File')
        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # allow adding a sphere
        meshMenu = mainMenu.addMenu('Mesh')
        self.add_mesh_action = QtWidgets.QAction('Add Mesh', self)
        self.add_mesh_action.triggered.connect(self.add_mesh)
        meshMenu.addAction(self.add_mesh_action)

        # allow to add an image
        meshMenu = mainMenu.addMenu('Image')
        self.add_image_action = QtWidgets.QAction('Add Image', self)
        self.add_image_action.triggered.connect(self.add_image)
        meshMenu.addAction(self.add_image_action)

        # allow to add actions
        meshMenu = mainMenu.addMenu('Actions')
        self.add_reset_camera_action = QtWidgets.QAction('Reset Camera', self)
        self.add_reset_camera_action.triggered.connect(self.reset_camera)
        meshMenu.addAction(self.add_reset_camera_action)

        self.add_zoom_in_action = QtWidgets.QAction('Zoom in', self)
        self.add_zoom_in_action.triggered.connect(self.zoom_in)
        meshMenu.addAction(self.add_zoom_in_action)

        self.add_zoom_out_action = QtWidgets.QAction('Zoom out', self)
        self.add_zoom_out_action.triggered.connect(self.zoom_out)
        meshMenu.addAction(self.add_zoom_out_action)

        if show:
            self.plotter.enable_joystick_actor_style()
            self.plotter.enable_trackball_actor_style()

            self.plotter.add_axes()
            self.plotter.add_camera_orientation_widget()
            self.plotter.show()
            self.show()

    
class App(MyMainWindow):
    def __init__(self):
        super().__init__()

        self.window_size = (1920, 1080)

        self.nocs_color = True
        self.point_clouds = False
        self.mirror_objects = False

        self.reference = 'ossicles'

        # initial the dictionaries
        self.mesh_actors = {}
        self.image_polydata = {}
        self.mesh_polydata = {}
        self.binded_meshes = {}
        self.undo_poses = []
        self.redo_poses = []
        
        # default opacity for image and surface
        self.set_image_opacity(1) # self.image_opacity = 0.35
        self.set_mesh_opacity(1) # self.surface_opacity = 1

        # Set up the camera
        self.camera = pv.Camera()
        self.cam_focal_length = 50000
        self.cam_viewup = (0, -1, 0)
        self.cam_position = -500 # -500mm
        self.set_camera_intrinsics(self.window_size[0], self.window_size[1], self.cam_focal_length)
        self.set_camera_extrinsics(self.cam_position, self.cam_viewup)

        self.plotter.camera = self.camera.copy()

        self.transformation_matrix = vis.config.gt_pose_5997_right
 
    def set_image_opacity(self, image_opacity: float):
        self.image_opacity = image_opacity
    
    def set_mesh_opacity(self, surface_opacity: float):
        self.surface_opacity = surface_opacity

    def set_mesh_info(self, name:str, mesh: trimesh.Trimesh()):
        assert mesh.vertices.shape[1] == 3, "it should be N by 3 matrix"
        assert mesh.faces.shape[1] == 3, "it should be N by 3 matrix"
        setattr(self, f"{name}_mesh", mesh)
    
    def set_camera_extrinsics(self, cam_position, cam_viewup):
        self.camera.SetPosition((0,0,cam_position))
        self.camera.SetFocalPoint((0,0,0))
        self.camera.SetViewUp(cam_viewup)
    
    def set_camera_intrinsics(self, width, height, cam_focal_length):
        
        # Set camera intrinsic attribute
        self.camera_intrinsics = np.array([
            [cam_focal_length, 0, width/2],
            [0, cam_focal_length, height/2],
            [0, 0, 1]
        ])
        
        cx = self.camera_intrinsics[0,2]
        cy = self.camera_intrinsics[1,2]
        f = self.camera_intrinsics[0,0]
        
        # convert the principal point to window center (normalized coordinate system) and set it
        wcx = -2*(cx - float(width)/2) / width
        wcy =  2*(cy - float(height)/2) / height
        self.camera.SetWindowCenter(wcx, wcy) # (0,0)
        
        # Setting the view angle in degrees
        view_angle = (180 / math.pi) * (2.0 * math.atan2(height/2.0, f)) # or view_angle = np.degrees(2.0 * math.atan2(height/2.0, f))
        self.camera.SetViewAngle(view_angle) # view angle should be in degrees

    def add_mesh(self):
        """ add a mesh to the pyqt frame """
        assert self.transformation_matrix is not None, "Need to set the transformation matrix first!"
                
        paths = {
            'ossicles': vis.config.OSSICLES_MESH_PATH_5997_right, 
            'facial_nerve': vis.config.FACIAL_NERVE_MESH_PATH_5997_right, 
            'chorda': vis.config.CHORDA_MESH_PATH_5997_right,
        }

        if len(paths) == 1: self.set_reference(paths.keys()[0])

        for mesh_name, mesh_source in paths.items():
            
            if isinstance(mesh_source, pathlib.WindowsPath) or isinstance(mesh_source, str):
                # Load the '.mesh' file
                if '.mesh' in str(mesh_source): 
                    mesh_source = vis.utils.load_trimesh(mesh_source)
                    # Set vertices and faces attribute
                    self.set_mesh_info(mesh_name, mesh_source)
                    colors = vis.utils.color_mesh(mesh_source.vertices, self.nocs_color)
                    if colors.shape != mesh_source.vertices.shape: colors = np.ones((len(mesh_source.vertices), 3)) * 0.5
                    assert colors.shape == mesh_source.vertices.shape, "colors shape should be the same as mesh_source.vertices shape"
                    mesh_data = pv.wrap(mesh_source)

                # Load the '.ply' file
                elif '.ply' in str(mesh_source): mesh_data = pv.read(mesh_source)

            elif isinstance(mesh_source, pv.PolyData):
                mesh_data = mesh_source
                colors = vis.utils.color_mesh(mesh_source.points, self.nocs_color)
                if colors.shape != mesh_source.points.shape: colors = np.ones((len(mesh_source.points), 3)) * 0.5
                assert colors.shape == mesh_source.points.shape, "colors shape should be the same as mesh_source.points shape"

            if self.nocs_color: # color array is(2454, 3)
                mesh = self.plotter.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=self.surface_opacity, name=mesh_name) if not self.point_clouds else self.plotter.add_mesh(mesh_data, scalars=colors, rgb=True, style='points', point_size=1, render_points_as_spheres=False, opacity=self.surface_opacity, name=mesh_name) #, show_edges=True)
            else: # color array is (2454, )
                if mesh_name == "ossicles": self.latlon = colors
                mesh = self.plotter.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=self.surface_opacity, name=mesh_name) if not self.point_clouds else self.plotter.add_mesh(mesh_data, scalars=colors, rgb=True, style='points', point_size=1, render_points_as_spheres=False, opacity=self.surface_opacity, name=mesh_name) #, show_edges=True)

            mesh.user_matrix = self.transformation_matrix if not self.mirror_objects else np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ self.transformation_matrix
            self.initial_pose = self.transformation_matrix
            
            # Add and save the actor
            actor, _ = self.plotter.add_actor(mesh, pickable=True, name=mesh_name)
            self.mesh_actors[mesh_name] = actor

            # Save the mesh data to dictionary
            self.mesh_polydata[mesh_name] = (mesh_data, colors)


    def add_image(self):
        """ add a image to the pyqt frame """
        image_source = np.array(PIL.Image.open(vis.config.IMAGE_PATH_5997))
        image = pv.UniformGrid(dimensions=(1920, 1080, 1), spacing=[0.01,0.01,1], origin=(0.0, 0.0, 0.0))
        image.point_data["values"] = image_source.reshape((1920*1080, 3)) # order = 'C
        image = image.translate(-1 * np.array(image.center), inplace=False)

        # Then add it to the plotter
        image = self.plotter.add_mesh(image, rgb=True, opacity=self.image_opacity, name='image')
        actor, _ = self.plotter.add_actor(image, pickable=False, name="image")

        # Save actor for later
        self.image_actor = actor 

    def reset_camera(self):
        self.plotter.camera = self.camera.copy()

    def zoom_in(self):
        self.plotter.camera.zoom(2)

    def zoom_out(self):
        self.plotter.camera.zoom(0.5)
          

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    sys.exit(app.exec_())