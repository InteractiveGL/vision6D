import sys
from typing import List, Dict, Tuple, Optional
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
from PyQt5.QtWidgets import QLineEdit, QMessageBox, QPushButton, QInputDialog, QFileDialog 
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow
import vision6D as vis

np.set_printoptions(suppress=True)

def try_except(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            if isinstance(args[0], MainWindow): QMessageBox.warning(args[0], 'vision6D', "Need to load a mesh first!", QMessageBox.Ok, QMessageBox.Ok)

    return wrapper
    
class MyMainWindow(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)
        
        # setting title
        self.setWindowTitle("Vision6D")
        self.showMaximized()
        
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
        # simple dialog to record users input info
        self.input_dialog = QInputDialog()
        self.file_dialog = QFileDialog()
        
        self.initial_dir = pathlib.Path('E:\GitHub\ossicles_6D_pose_estimation\data')
        self.meshdict = {}
                
        # allow to add files
        fileMenu = mainMenu.addMenu('File')

        self.add_image_action = QtWidgets.QAction('Add Image', self)
        self.add_image_action.triggered.connect(self.add_image_file)
        fileMenu.addAction(self.add_image_action)

        self.add_mesh_action = QtWidgets.QAction('Add Mesh', self)
        self.add_mesh_action.triggered.connect(self.add_mesh_file)
        fileMenu.addAction(self.add_mesh_action)

        self.add_pose_action = QtWidgets.QAction('Add Pose', self)
        self.add_pose_action.triggered.connect(self.add_pose_file)
        fileMenu.addAction(self.add_pose_action)

        # Add set attribute menu
        setAttrMenu = mainMenu.addMenu('Set')
        self.add_set_reference_action = QtWidgets.QAction('Set Reference', self)
        on_click_set_reference = functools.partial(self.on_click, info="Set Reference Mesh Name", hints='ossicles')
        self.add_set_reference_action.triggered.connect(on_click_set_reference)
        setAttrMenu.addAction(self.add_set_reference_action)

        self.add_set_image_opacity_action = QtWidgets.QAction('Set Image Opacity', self)
        on_click_set_image_opacity = functools.partial(self.on_click, info="Set Image Opacity (range from 0 to 1)", hints='0.99')
        self.add_set_image_opacity_action.triggered.connect(on_click_set_image_opacity)
        setAttrMenu.addAction(self.add_set_image_opacity_action)

        self.add_set_mesh_opacity_action = QtWidgets.QAction('Set Mesh Opacity', self)
        on_click_set_mesh_opacity = functools.partial(self.on_click, info="Set Mesh Opacity (range from 0 to 1)", hints='0.99')
        self.add_set_mesh_opacity_action.triggered.connect(on_click_set_mesh_opacity)
        setAttrMenu.addAction(self.add_set_mesh_opacity_action)

        # Add camera related actions
        CameraMenu = mainMenu.addMenu('Camera')
        self.add_reset_camera_action = QtWidgets.QAction('Reset Camera (c)', self)
        self.add_reset_camera_action.triggered.connect(self.reset_camera)
        CameraMenu.addAction(self.add_reset_camera_action)

        self.add_zoom_in_action = QtWidgets.QAction('Zoom In (x)', self)
        self.add_zoom_in_action.triggered.connect(self.zoom_in)
        CameraMenu.addAction(self.add_zoom_in_action)

        self.add_zoom_out_action = QtWidgets.QAction('Zoom Out (z)', self)
        self.add_zoom_out_action.triggered.connect(self.zoom_out)
        CameraMenu.addAction(self.add_zoom_out_action)

        # Add register related actions
        RegisterMenu = mainMenu.addMenu('Regiter')
        self.add_reset_gt_pose_action = QtWidgets.QAction('Reset GT Pose (k)', self)
        self.add_reset_gt_pose_action.triggered.connect(self.reset_gt_pose)
        RegisterMenu.addAction(self.add_reset_gt_pose_action)

        self.add_update_gt_pose_action = QtWidgets.QAction('Update GT Pose (l)', self)
        self.add_update_gt_pose_action.triggered.connect(self.update_gt_pose)
        RegisterMenu.addAction(self.add_update_gt_pose_action)

        self.add_current_pose_action = QtWidgets.QAction('Current Pose (t)', self)
        self.add_current_pose_action.triggered.connect(self.current_pose)
        RegisterMenu.addAction(self.add_current_pose_action)

        self.add_undo_pose_action = QtWidgets.QAction('Undo Pose (s)', self)
        self.add_undo_pose_action.triggered.connect(self.undo_pose)
        RegisterMenu.addAction(self.add_undo_pose_action)

        if show:
            self.plotter.enable_joystick_actor_style()
            self.plotter.enable_trackball_actor_style()
            self.plotter.track_click_position(callback=self.track_click_callback, side='l')

            # camera related key bindings
            self.plotter.add_key_event('c', self.reset_camera)
            self.plotter.add_key_event('z', self.zoom_out)
            self.plotter.add_key_event('x', self.zoom_in)

            # registration related key bindings
            self.plotter.add_key_event('k', self.reset_gt_pose)
            self.plotter.add_key_event('l', self.update_gt_pose)
            self.plotter.add_key_event('t', self.current_pose)
            self.plotter.add_key_event('s', self.undo_pose)

            self.plotter.add_axes()
            self.plotter.add_camera_orientation_widget()

            self.plotter.show()
            self.show()

    def on_click(self, info, hints):
        output, ok = self.input_dialog.getText(self, 'Input', info, text=hints)
        info = info.upper()
        if ok: 
            if 'reference'.upper() in info:
                try:
                    self.set_reference(output)
                except AssertionError:
                    QMessageBox.warning(self, 'vision6D', "Reference name does not exist in the paths", QMessageBox.Ok, QMessageBox.Ok)
            elif 'image opacity'.upper() in info:
                try:
                    self.set_image_opacity(float(output))
                except AssertionError:
                    QMessageBox.warning(self, 'vision6D', "Image opacity should range from 0 to 1", QMessageBox.Ok, QMessageBox.Ok)
            elif 'mesh opacity'.upper() in info:
                try:
                    self.set_mesh_opacity(float(output))
                except AssertionError:
                    QMessageBox.warning(self, 'vision6D', "Mesh opacity should range from 0 to 1", QMessageBox.Ok, QMessageBox.Ok)

    def add_image_file(self):
        image_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(self.initial_dir / "frames"), "Files (*.png *.jpg)")
        if image_path != '': self.add_image(image_path)

    def add_mesh_file(self):
        mesh_source, _ = self.file_dialog.getOpenFileName(None, "Open file", str(self.initial_dir / "surgical_planning"), "Files (*.mesh *.ply)")
        if mesh_source != '':
            mesh_name, ok = self.input_dialog.getText(self, 'Input', 'Specify the object Class name', text='ossicles')
            if ok: 
                self.meshdict[mesh_name] = mesh_source
                self.add_mesh(mesh_name, mesh_source)
            if self.reference is None: 
                reply = QMessageBox.question(self,"vision6D", "Do you want to make this mesh as a reference?", QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.Yes: self.reference = mesh_name
      
    def add_pose_file(self):
        pose_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(self.initial_dir / "gt_poses"), "Files (*.npy)")
        if pose_path != '': self.set_transformation_matrix(matrix=np.load(pose_path))
    
class App(MyMainWindow):
    def __init__(self):
        super().__init__()

        self.window_size = (1920, 1080)

        self.nocs_color = True
        self.point_clouds = False
        self.mirror_objects = False

        self.reference = None
        self.transformation_matrix = np.eye(4)

        # initial the dictionaries
        self.mesh_actors = {}
        self.image_polydata = {}
        self.mesh_polydata = {}
        self.binded_meshes = {}
        self.undo_poses = []
        
        # default opacity for image and surface
        self.set_image_opacity(0.99) # self.image_opacity = 0.35
        self.set_mesh_opacity(0.8) # self.surface_opacity = 1

        # Set up the camera
        self.camera = pv.Camera()
        self.cam_focal_length = 50000
        self.cam_viewup = (0, -1, 0)
        self.cam_position = -500 # -500mm
        self.set_camera_intrinsics(self.window_size[0], self.window_size[1], self.cam_focal_length)
        self.set_camera_extrinsics(self.cam_position, self.cam_viewup)

        self.plotter.camera = self.camera.copy()

    def set_reference(self, name:str):     
        assert name in self.meshdict.keys(), "reference name is not in the path!"
        self.reference = name
  
    def set_transformation_matrix(self, matrix:np.ndarray=None, rot:np.ndarray=None, trans:np.ndarray=None):
        if matrix is None: matrix = np.vstack((np.hstack((rot, trans)), [0, 0, 0, 1]))
        self.transformation_matrix = matrix
        self.initial_pose = self.transformation_matrix
        self.reset_gt_pose()
        self.reset_camera()

    def set_image_opacity(self, image_opacity: float):
        assert image_opacity>=0 and image_opacity<=1, "image opacity should range from 0 to 1!"
        self.image_opacity = image_opacity
        try:
            self.image_actor.GetProperty().opacity = self.image_opacity
            self.plotter.add_actor(self.image_actor, pickable=False, name="image")
        except AttributeError:
            pass

    def set_mesh_opacity(self, surface_opacity: float):
        assert surface_opacity>=0 and surface_opacity<=1, "mesh opacity should range from 0 to 1!"
        self.surface_opacity = surface_opacity
        try:
            transformation_matrix = self.mesh_actors[self.reference].user_matrix
            for actor_name, actor in self.mesh_actors.items():
                actor.user_matrix = transformation_matrix if not "_mirror" in actor_name else np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                actor.GetProperty().opacity = self.surface_opacity
                self.plotter.add_actor(actor, pickable=True, name=actor_name)
        except KeyError:
            pass
     
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

    def add_image(self, file_path):

        """ add a image to the pyqt frame """
        image_source = np.array(PIL.Image.open(file_path))
        image = pv.UniformGrid(dimensions=(1920, 1080, 1), spacing=[0.01,0.01,1], origin=(0.0, 0.0, 0.0))
        image.point_data["values"] = image_source.reshape((1920*1080, 3)) # order = 'C
        image = image.translate(-1 * np.array(image.center), inplace=False)

        # Then add it to the plotter
        image = self.plotter.add_mesh(image, rgb=True, opacity=self.image_opacity, name='image')
        actor, _ = self.plotter.add_actor(image, pickable=False, name="image")

        # Save actor for later
        self.image_actor = actor 

    def add_mesh(self, mesh_name, mesh_source):
        """ add a mesh to the pyqt frame """
                              
        if isinstance(mesh_source, pathlib.WindowsPath) or isinstance(mesh_source, str):
            # Load the '.mesh' file
            if '.mesh' in str(mesh_source): 
                mesh_source = vis.utils.load_trimesh(mesh_source)
                assert (mesh_source.vertices.shape[1] == 3 and mesh_source.faces.shape[1] == 3), "it should be N by 3 matrix"
                # Set vertices and faces attribute
                setattr(self, f"{mesh_name}_mesh", mesh_source)
                colors = vis.utils.color_mesh(mesh_source.vertices, self.nocs_color)
                if colors.shape != mesh_source.vertices.shape: colors = np.ones((len(mesh_source.vertices), 3)) * 0.5
                assert colors.shape == mesh_source.vertices.shape, "colors shape should be the same as mesh_source.vertices shape"
                mesh_data = pv.wrap(mesh_source)

            # Load the '.ply' file
            elif '.ply' in str(mesh_source): 
                mesh_data = pv.read(mesh_source)
                colors = vis.utils.color_mesh(mesh_data.points, self.nocs_color)
                if colors.shape != mesh_data.points.shape: colors = np.ones((len(mesh_data.points), 3)) * 0.5
                assert colors.shape == mesh_data.points.shape, "colors shape should be the same as mesh_data.points shape"

        if self.nocs_color: # color array is(2454, 3)
            mesh = self.plotter.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=self.surface_opacity, name=mesh_name) if not self.point_clouds else self.plotter.add_mesh(mesh_data, scalars=colors, rgb=True, style='points', point_size=1, render_points_as_spheres=False, opacity=self.surface_opacity, name=mesh_name) #, show_edges=True)
        else: # color array is (2454, )
            if mesh_name == "ossicles": self.latlon = colors
            mesh = self.plotter.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=self.surface_opacity, name=mesh_name) if not self.point_clouds else self.plotter.add_mesh(mesh_data, scalars=colors, rgb=True, style='points', point_size=1, render_points_as_spheres=False, opacity=self.surface_opacity, name=mesh_name) #, show_edges=True)

        mesh.user_matrix = self.transformation_matrix if not self.mirror_objects else np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ self.transformation_matrix
        self.initial_pose = mesh.user_matrix
                
        # Add and save the actor
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=mesh_name)
        self.mesh_actors[mesh_name] = actor

        # Save the mesh data to dictionary
        self.mesh_polydata[mesh_name] = (mesh_data, colors)

    def reset_camera(self, *args):
        self.plotter.camera = self.camera.copy()

    def zoom_in(self, *args):
        self.plotter.camera.zoom(2)

    def zoom_out(self, *args):
        self.plotter.camera.zoom(0.5)

    def track_click_callback(self, *args):
        if len(self.undo_poses) > 20: self.undo_poses.pop(0)
        if self.reference is not None: self.undo_poses.append(self.mesh_actors[self.reference].user_matrix)

    @try_except
    def reset_gt_pose(self, *args):
        print(f"\nRT: \n{self.initial_pose}\n")
        for actor_name, actor in self.mesh_actors.items():
            actor.user_matrix = self.initial_pose
            self.plotter.add_actor(actor, pickable=True, name=actor_name)

    def update_gt_pose(self, *args):
        if self.reference is not None:
            self.transformation_matrix = self.mesh_actors[self.reference].user_matrix
            self.initial_pose = self.transformation_matrix
            print(f"\nRT: \n{self.initial_pose}\n")
            for actor_name, actor in self.mesh_actors.items():
                # update the the actor's user matrix
                self.transformation_matrix = self.transformation_matrix if not '_mirror' in actor_name else np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ self.transformation_matrix
                actor.user_matrix = self.transformation_matrix
                self.plotter.add_actor(actor, pickable=True, name=actor_name)

    def current_pose(self, *args):
        if self.reference is not None:
            transformation_matrix = self.mesh_actors[self.reference].user_matrix
            print(f"\nRT: \n{transformation_matrix}\n")
            for actor_name, actor in self.mesh_actors.items():
                actor.user_matrix = transformation_matrix if not "_mirror" in actor_name else np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                self.plotter.add_actor(actor, pickable=True, name=actor_name)

    def undo_pose(self, *args):
        if len(self.undo_poses) != 0: 
            transformation_matrix = self.undo_poses.pop()
            if (transformation_matrix == self.mesh_actors[self.reference].user_matrix).all():
                if len(self.undo_poses) != 0: transformation_matrix = self.undo_poses.pop()
            for actor_name, actor in self.mesh_actors.items():
                actor.user_matrix = transformation_matrix if not "_mirror" in actor_name else np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                self.plotter.add_actor(actor, pickable=True, name=actor_name)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    sys.exit(app.exec_())