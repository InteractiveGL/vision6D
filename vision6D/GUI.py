import os
import re
import numpy as np
import pyvista as pv
import trimesh
import pathlib
import PIL
import ast
import json
import math
import copy
import cv2
import pyvista as pv

# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QPoint
from .mainwindow import MyMainWindow

from . import utils
from . import widgets_gui

np.set_printoptions(suppress=True)

class Interface(MyMainWindow):
    def __init__(self):
        super().__init__()
        # initialize
        self.reference = None
        self.transformation_matrix = np.eye(4)
        self.initial_pose = None

        self.mirror_x = False
        self.mirror_y = False

        
        self.mesh_actors = {}
        self.meshdict = {}
        
        self.undo_poses = {}
        self.latlon = utils.load_latitude_longitude()

        self.colors = ["cyan", "magenta", "yellow", "lime", "dodgerblue", "darkviolet", "darkorange", "forestgreen"]
        self.used_colors = []
        self.mesh_colors = {}

        self.mesh_opacity = {}
        self.store_mesh_opacity = {}
        
        self.surface_opacity = self.opacity_spinbox.value()
        
        # Set mesh spacing
        self.mesh_spacing = [1, 1, 1]
        
        # Set the camera
        self.camera = pv.Camera()
        self.fx = 50000
        self.fy = 50000
        self.cx = self.window_size[0] // 2
        self.cy = self.window_size[1] // 2
        self.cam_viewup = (0, -1, 0)
        self.cam_position = -500
        self.set_camera_props()

        self.track_actors_names = []
        self.button_group_actors_names = QtWidgets.QButtonGroup(self)
        self.toggle_hide_meshes_flag = False

        # Shortcut key bindings
        # camera related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("c"), self).activated.connect(self.reset_camera)
        QtWidgets.QShortcut(QtGui.QKeySequence("z"), self).activated.connect(self.zoom_out)
        QtWidgets.QShortcut(QtGui.QKeySequence("x"), self).activated.connect(self.zoom_in)

        # registration related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("k"), self).activated.connect(self.reset_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("l"), self).activated.connect(self.update_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("s"), self).activated.connect(self.undo_pose)

        # Reset the mask location
        QtWidgets.QShortcut(QtGui.QKeySequence("t"), self).activated.connect(self.reset_mask)

        # change image opacity key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("b"), self).activated.connect(lambda up=True: self.toggle_image_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("n"), self).activated.connect(lambda up=False: self.toggle_image_opacity(up))

        # change mask opacity key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("g"), self).activated.connect(lambda up=True: self.toggle_mask_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("h"), self).activated.connect(lambda up=False: self.toggle_mask_opacity(up))

        # change mesh opacity key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("y"), self).activated.connect(lambda up=True: self.toggle_surface_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("u"), self).activated.connect(lambda up=False: self.toggle_surface_opacity(up))

        # Add shortcut to the right, left, space buttons
        QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self).activated.connect(self.next_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self).activated.connect(self.prev_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self).activated.connect(self.play_video)

    def button_actor_name_clicked(self, text):
        if text in self.mesh_actors:
            # set the current mesh color
            self.color_button.setText(self.mesh_colors[text])
            # set mesh reference
            self.reference = text
            self.current_pose()
            curr_opacity = self.mesh_actors[self.reference].GetProperty().opacity
            self.opacity_spinbox.setValue(curr_opacity)
        else:
            self.color_button.setText("Color")
            if text == 'image': curr_opacity = self.image_store.image_opacity
            elif text == 'mask': curr_opacity = self.mask_store.mask_opacity
            else: curr_opacity = self.opacity_spinbox.value()
            self.opacity_spinbox.setValue(curr_opacity)
            # self.reference = None
        
        output = f"-> Actor {text}, and its opacity is {curr_opacity}"
        if output not in self.output_text.toPlainText(): self.output_text.append(output)
                                            
    def set_image_opacity(self, image_opacity: float):
        self.image_store.image_opacity = image_opacity
        self.image_store.image_actor.GetProperty().opacity = image_opacity
        self.plotter.add_actor(self.image_store.image_actor, pickable=False, name='image')

    def set_mask_opacity(self, mask_opacity: float):
        self.mask_store.mask_opacity = mask_opacity
        self.mask_store.mask_actor.GetProperty().opacity = mask_opacity
        self.plotter.add_actor(self.mask_store.mask_actor, pickable=True, name='mask')

    def set_mesh_opacity(self, name: str, surface_opacity: float):
        assert surface_opacity>=0 and surface_opacity<=1, "mesh opacity should range from 0 to 1!"
        self.mesh_opacity[name] = surface_opacity
        self.mesh_actors[name].user_matrix = pv.array_from_vtkmatrix(self.mesh_actors[name].GetMatrix())
        self.mesh_actors[name].GetProperty().opacity = surface_opacity
        self.plotter.add_actor(self.mesh_actors[name], pickable=True, name=name)

    def add_image(self, image_source):

        image, original_image, channel = self.image_store.add_image(image_source, self.mirror_x, self.mirror_y)

        # Then add it to the plotter
        if channel == 1: 
            image = self.plotter.add_mesh(image, cmap='gray', opacity=self.image_store.image_opacity, name='image')
        else: 
            image = self.plotter.add_mesh(image, rgb=True, opacity=self.image_store.image_opacity, name='image')
        
        actor, _ = self.plotter.add_actor(image, pickable=False, name='image')

        # Save actor for later
        self.image_store.image_actor = actor

        # get the image scalar
        image_data = utils.get_image_actor_scalars(self.image_store.image_actor)
        assert (image_data == original_image).all() or (image_data*255 == original_image).all(), "image_data and image_source should be equal"
        
        # add remove current image to removeMenu
        if 'image' not in self.track_actors_names:
            self.track_actors_names.append('image')
            self.add_button_actor_name('image')

        self.check_button('image')

    def load_mask(self, mask_surface, points):
        # Add mask surface object to the plot
        mask_mesh = self.plotter.add_mesh(mask_surface, color="white", style='surface', opacity=self.mask_store.mask_opacity)
        actor, _ = self.plotter.add_actor(mask_mesh, pickable=True, name='mask')
        self.mask_store.mask_actor = actor
        mask_point_data = utils.get_mask_actor_points(self.mask_store.mask_actor)
        assert np.isclose(((mask_point_data+self.mask_store.mask_bottom_point-self.mask_store.mask_offset) - points), 0).all(), "mask_point_data and points should be equal"

    def add_mask(self, mask_source):
        mask_surface, points = self.mask_store.add_mask(mask_source, self.mirror_x, self.mirror_y)
        self.load_mask(mask_surface, points)
        
        # Add remove current image to removeMenu
        if 'mask' not in self.track_actors_names:
            self.track_actors_names.append('mask')
            self.add_button_actor_name('mask')

        self.check_button('mask')

    def reset_mask(self):
        if self.mask_store.mask_path:
            mask_surface, points = self.mask_store.add_mask(self.mask_store.mask_path, self.mirror_x, self.mirror_y)
            self.load_mask(mask_surface, points)

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

        if flag:
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
            self.color_button.setText(self.mesh_colors[mesh_name])
            mesh = self.plotter.add_mesh(mesh_data, color=mesh_color, opacity=self.mesh_opacity[mesh_name], name=mesh_name)

            mesh.user_matrix = self.transformation_matrix if transformation_matrix is None else transformation_matrix
            if self.initial_pose is None: self.initial_pose = self.transformation_matrix
                    
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

            self.check_button(mesh_name)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "The mesh format is not supported!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def reset_camera(self):
        self.plotter.camera = self.camera.copy()

    def zoom_in(self):
        self.plotter.camera.zoom(2)

    def zoom_out(self):
        self.plotter.camera.zoom(0.5)

    def check_button(self, actor_name):
        for button in self.button_group_actors_names.buttons():
            if button.text() == actor_name: 
                button.setChecked(True)
                self.button_actor_name_clicked(actor_name)
                break

    def add_button_actor_name(self, actor_name):
        button = QtWidgets.QPushButton(actor_name)
        button.setCheckable(True)  # Set the button to be checkable
        button.clicked.connect(lambda _, text=actor_name: self.button_actor_name_clicked(text))
        button.setChecked(True)
        button.setFixedSize(self.display.size().width(), 50)
        self.button_layout.insertWidget(0, button) # insert from the top # self.button_layout.addWidget(button)
        self.button_group_actors_names.addButton(button)
        self.button_actor_name_clicked(actor_name)

    def toggle_image_opacity(self, up):
        change = 0.05
        if not up: change *= -1
        self.image_store.update_opacity(change)
        self.plotter.add_actor(self.image_store.image_actor, pickable=False, name="image")
        self.ignore_spinbox_value_change = True
        self.opacity_spinbox.setValue(self.image_store.image_opacity)
        self.ignore_spinbox_value_change = False

    def toggle_mask_opacity(self, up):
        change = 0.05
        if not up: change *= -1
        self.mask_store.update_opacity(change)
        self.plotter.add_actor(self.mask_store.mask_actor, pickable=True, name="mask")
        self.ignore_spinbox_value_change = True
        self.opacity_spinbox.setValue(self.mask_store.mask_opacity)
        self.ignore_spinbox_value_change = False

    def toggle_surface_opacity(self, up):
        current_opacity = self.opacity_spinbox.value()
        if up:
            current_opacity += 0.05
            if current_opacity > 1: current_opacity = 1
        else:
            current_opacity -= 0.05
            if current_opacity < 0: current_opacity = 0
        self.opacity_spinbox.setValue(current_opacity)

    def opacity_value_change(self, value):
        if self.ignore_spinbox_value_change: return 0
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button:
            actor_name = checked_button.text()
            if actor_name == 'image': self.set_image_opacity(value)
            elif actor_name == 'mask': self.set_mask_opacity(value)
            elif actor_name in self.mesh_actors: 
                self.store_mesh_opacity[actor_name] = copy.deepcopy(self.mesh_opacity[actor_name])
                self.mesh_opacity[actor_name] = value
                self.set_mesh_opacity(actor_name, self.mesh_opacity[actor_name])
        else:
            self.ignore_spinbox_value_change = True
            self.opacity_spinbox.setValue(value)
            self.ignore_spinbox_value_change = False
            return 0
        
    def toggle_hide_meshes_button(self):
        self.toggle_hide_meshes_flag = not self.toggle_hide_meshes_flag
        
        if self.toggle_hide_meshes_flag:
            for button in self.button_group_actors_names.buttons():
                if button.text() in self.mesh_actors:
                    button.setChecked(True); self.opacity_value_change(0)
    
            checked_button = self.button_group_actors_names.checkedButton()
            if checked_button: 
                self.ignore_spinbox_value_change = True
                self.opacity_spinbox.setValue(0.0)
                self.ignore_spinbox_value_change = False
            else: QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        
        else:
            for button in self.button_group_actors_names.buttons():
                if button.text() in self.mesh_actors:
                    button.setChecked(True)
                    self.opacity_value_change(self.store_mesh_opacity[button.text()])

            checked_button = self.button_group_actors_names.checkedButton()
            if checked_button:
                self.ignore_spinbox_value_change = True
                if checked_button.text() in self.mesh_actors: self.opacity_spinbox.setValue(self.mesh_opacity[checked_button.text()])
                self.ignore_spinbox_value_change = False
            else: QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

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

    def camera_calibrate(self):
        if self.image_store.image_path:
            original_image = np.array(PIL.Image.open(self.image_store.image_path), dtype='uint8')
            # make the the original image shape is [h, w, 3] to match with the rendered calibrated_image
            original_image = original_image[..., :3]
            if len(original_image.shape) == 2: original_image = original_image[..., None]
            if original_image.shape[-1] == 1: original_image = np.dstack((original_image, original_image, original_image))
            calibrated_image = np.array(self.render_image(self.plotter.camera.copy()), dtype='uint8')
            if original_image.shape != calibrated_image.shape:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Original image shape is not equal to calibrated image shape!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return 0
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
        calibrate_pop = widgets_gui.CalibrationPopWindow(calibrated_image, original_image)
        calibrate_pop.exec_()

    def set_camera(self):
        dialog = widgets_gui.CameraPropsInputDialog(
            line1=("Fx", self.fx), 
            line2=("Fy", self.fy), 
            line3=("Cx", self.cx), 
            line4=("Cy", self.cy), 
            line5=("View Up", self.cam_viewup), 
            line6=("Cam Position", self.cam_position))
        if dialog.exec():
            fx, fy, cx, cy, cam_viewup, cam_position = dialog.getInputs()
            pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup, pre_cam_position = self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position
            if not (fx == '' or fy == '' or cx == '' or cy == '' or cam_viewup == '' or cam_position == ''):
                try:
                    self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position = ast.literal_eval(fx), ast.literal_eval(fy), ast.literal_eval(cx), ast.literal_eval(cy), ast.literal_eval(cam_viewup), ast.literal_eval(cam_position)
                    self.set_camera_props()
                except:
                    self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position = pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup, pre_cam_position
                    QtWidgets.QMessageBox.warning(self, 'vision6D', "Error occured, check the format of the input values", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def set_pose(self):
        # get the gt pose
        res = self.get_text_dialog.exec_()

        if res == QtWidgets.QDialog.Accepted:
            try:
                gt_pose = ast.literal_eval(self.get_text_dialog.user_text)
                gt_pose = np.array(gt_pose)
                if gt_pose.shape != (4, 4): 
                    QtWidgets.QMessageBox.warning(self, 'vision6D', "It needs to be a 4 by 4 matrix", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok) 
                    return None
                else:
                    self.hintLabel.hide()
                    transformation_matrix = gt_pose
                    self.transformation_matrix = transformation_matrix
                    if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                    if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                    self.add_pose(matrix=transformation_matrix)
                    return 0
            except: 
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return None
        else: 
            return None

    def add_workspace(self, prompt=False):
        if prompt:
            self.workspace_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.json)")
        if self.workspace_path:
            self.hintLabel.hide()
            with open(str(self.workspace_path), 'r') as f: 
                workspace = json.load(f)

            if 'image_path' in workspace: self.add_image_file(image_path=workspace['image_path'])
            if 'video_path' in workspace:
                self.video_path = workspace['video_path']
                self.add_video_file()
            if 'mask_path' in workspace: self.add_mask_file(workspace['mask_path'])
            if 'pose_path' in workspace: # need to load pose before loading meshes
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
            self.folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        
        if self.folder_path:
            folders = [d for d in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, d))]
            flag = True

            if 'images' in folders:
                flag = False
                image_files, image_dir = self.get_files_from_folder('images')
                image_path = str(image_dir / image_files[self.current_frame])
                if os.path.isfile(image_path): self.add_image_file(image_path=image_path)

            if 'masks' in folders:
                flag = False
                mask_files, mask_dir = self.get_files_from_folder('masks')
                mask_path = str(mask_dir / mask_files[self.current_frame])
                if os.path.isfile(mask_path): self.add_mask_file(mask_path=mask_path)
                    
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
            self.video_player = widgets_gui.VideoPlayer(self.video_path, self.current_frame)
            self.play_video_button.setText("Play Video")
            self.output_text.append(f"-> Load video {self.video_path} into vision6D")
            self.output_text.append(f"-> Current frame is ({self.current_frame}/{self.video_player.frame_count})")
            self.fps = round(self.video_player.fps)
            self.load_per_frame_info(True)
            self.sample_video()
            
    def add_image_file(self, image_path='', prompt=False):
        if prompt:
            image_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)")
        if image_path:
            self.hintLabel.hide()
            self.add_image(image_path)
            
    def add_mask_file(self, mask_path='', prompt=False):
        if prompt:
            mask_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)") 
        if mask_path:
            self.hintLabel.hide()
            self.add_mask(mask_path)

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

    def add_pose(self, matrix:np.ndarray=None, rot:np.ndarray=None, trans:np.ndarray=None):
        if matrix is not None: 
            self.initial_pose = matrix
            self.reset_gt_pose()
            self.reset_camera()
        else:
            if (rot and trans): matrix = np.vstack((np.hstack((rot, trans)), [0, 0, 0, 1]))

    def add_pose_file(self):
        if self.pose_path:
            self.hintLabel.hide()
            transformation_matrix = np.load(self.pose_path)
            self.transformation_matrix = transformation_matrix
            if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            self.add_pose(matrix=transformation_matrix)
    
    def register_pose(self, pose):
        for actor_name, actor in self.mesh_actors.items():
            actor.user_matrix = pose
            self.plotter.add_actor(actor, pickable=True, name=actor_name)

    def reset_gt_pose(self):
        self.output_text.append(f"-> Reset the GT pose to: \n{self.initial_pose}")
        self.register_pose(self.initial_pose)

    def update_gt_pose(self):
        self.initial_pose = self.transformation_matrix
        self.current_pose()
        self.output_text.append(f"Update the GT pose to: \n{self.initial_pose}")
            
    def current_pose(self):
        if len(self.mesh_actors) == 1: self.reference = list(self.mesh_actors.keys())[0]
        if self.reference:
            self.transformation_matrix = self.mesh_actors[self.reference].user_matrix
            self.output_text.append(f"-> Current reference mesh is: <span style='background-color:yellow; color:black;'>{self.reference}</span>")
            self.output_text.append(f"Current pose is: \n{self.transformation_matrix}")
            self.register_pose(self.transformation_matrix)

    def undo_pose(self):
        if self.button_group_actors_names.checkedButton():
            actor_name = self.button_group_actors_names.checkedButton().text()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Choose a mesh actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        if len(self.undo_poses[actor_name]) != 0: 
            self.transformation_matrix = self.undo_poses[actor_name].pop()
            if (self.transformation_matrix == self.mesh_actors[actor_name].user_matrix).all():
                if len(self.undo_poses[actor_name]) != 0: 
                    self.transformation_matrix = self.undo_poses[actor_name].pop()

            self.output_text.append(f"-> Current reference mesh is: <span style='background-color:yellow; color:black;'>{actor_name}</span>")
            self.output_text.append(f"Undo pose to: \n{self.transformation_matrix}")
                
            self.mesh_actors[actor_name].user_matrix = self.transformation_matrix
            self.plotter.add_actor(self.mesh_actors[actor_name], pickable=True, name=actor_name)

    def mirror_actors(self, direction):

        if direction == 'x': self.mirror_x = not self.mirror_x
        elif direction == 'y': self.mirror_y = not self.mirror_y

        #^ mirror the image actor
        if self.image_store.image_path: self.add_image(self.image_store.image_path)

        #^ mirror the mask actor
        if self.mask_store.mask_path: self.add_mask(self.mask_store.mask_path)

        #^ mirror the mesh actors
        if self.reference:
            transformation_matrix = self.initial_pose
            if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            self.add_mesh(self.reference, self.meshdict[self.reference], transformation_matrix)
            # Output the mirrored transformation matrix
            self.output_text.append(f"-> Mirrored transformation matrix is: \n{transformation_matrix}")
             
    def remove_actor(self, button):
        name = button.text()
        if name == 'image': 
            actor = self.image_store.image_actor
            self.image_store.reset()
        elif name == 'mask':
            actor = self.mask_store.mask_actor
            self.mask_store.reset()
        elif name in self.mesh_actors: 
            actor = self.mesh_actors[name]
            del self.mesh_actors[name] # remove the item from the mesh dictionary
            del self.mesh_colors[name]
            del self.mesh_opacity[name]
            del self.meshdict[name]
            self.reference = None
            self.color_button.setText("Color")
            self.mesh_spacing = [1, 1, 1]

        self.plotter.remove_actor(actor)
        self.track_actors_names.remove(name)
        self.output_text.append(f"-> Remove actor: {name}")
        # remove the button from the button group
        self.button_group_actors_names.removeButton(button)
        # remove the button from the self.button_layout widget
        self.button_layout.removeWidget(button)
        # offically delete the button
        button.deleteLater()

        # clear out the plot if there is no actor
        if self.image_store.image_actor is None and self.mask_store.mask_actor is None and len(self.mesh_actors) == 0: self.clear_plot()

    def remove_actors_button(self):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button: self.remove_actor(checked_button)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def clear_plot(self):
        
        # Clear out everything in the remove menu
        for button in self.button_group_actors_names.buttons():
            name = button.text()
            if name == 'image': actor = self.image_store.image_actor
            elif name == 'mask': actor = self.mask_store.mask_actor
            elif name in self.mesh_actors: actor = self.mesh_actors[name]
            self.plotter.remove_actor(actor)
            # remove the button from the button group
            self.button_group_actors_names.removeButton(button)
            # remove the button from the self.button_layout widget
            self.button_layout.removeWidget(button)
            # offically delete the button
            button.deleteLater()

        self.hintLabel.show()

        self.image_store.reset()
        self.mask_store.reset()

        # Re-initial the dictionaries
        self.delete_video_folder()
        self.workspace_path = None
        self.mesh_path = None
        self.pose_path = None
        self.meshdict = {}
        self.mesh_colors = {}

        # reset everything to original actor opacity
        self.mesh_opacity = {}
        self.store_mesh_opacity = {}
        self.surface_opacity = self.opacity_spinbox.value()

        self.mesh_spacing = [1, 1, 1]

        self.mirror_x = False
        self.mirror_y = False

        self.reference = None
        self.transformation_matrix = np.eye(4)
        self.mesh_actors = {}
        self.undo_poses = {}
        self.track_actors_names = []

        self.colors = ["cyan", "magenta", "yellow", "lime", "dodgerblue", "darkviolet", "darkorange", "forestgreen"]
        self.used_colors = []
        self.color_button.setText("Color")

        self.clear_output_text()

    def render_mesh(self, camera):
        self.render.clear()
        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_actors[self.reference])
        mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        colors = utils.get_mesh_actor_scalars(self.mesh_actors[self.reference])
        if colors is not None: 
            assert colors.shape == vertices.shape, "colors shape should be the same as vertices shape"
            mesh = self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=1, name=self.reference)
        else:
            mesh = self.render.add_mesh(mesh_data, color=self.mesh_colors[self.reference], style='surface', opacity=1, name=self.reference)
        mesh.user_matrix = self.mesh_actors[self.reference].user_matrix
        
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        image = self.render.last_image
        return image

    def export_image(self):
        if self.image_store.image_actor:
            image = self.image_store.render_image(camera=self.camera.copy())
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Image Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.output_text.append(f"-> Export image render to:\n {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def export_mask(self):
        if self.mask_store.mask_actor:
            image = self.mask_store.render_mask(camera=self.camera.copy())
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Mask Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.output_text.append(f"-> Export mask render to:\n {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a mask first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
   
    def export_pose(self):
        if self.reference: 
            self.update_gt_pose()
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Pose Files (*.npy)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.npy')
                np.save(output_path, self.transformation_matrix)
                self.output_text.append(f"-> Saved:\n{self.transformation_matrix}\nExport to:\n {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
    
    def export_mesh_render(self, save_render=True):

        if self.reference:
            image = self.render_mesh(camera=self.camera.copy())
            if save_render:
                output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Mesh Files (*.png)")
                if output_path:
                    if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
                    rendered_image = PIL.Image.fromarray(image)
                    rendered_image.save(output_path)
                    self.output_text.append(f"-> Export reference mesh render to:\n {str(output_path)}")
            return image
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def export_segmesh_render(self):

        if self.reference and self.mask_store.mask_actor:
            segmask = self.mask_store.render_mask(camera=self.camera.copy())
            if np.max(segmask) > 1: segmask = segmask / 255
            image = self.render_mesh(camera=self.camera.copy())
            image = (image * segmask).astype(np.uint8)
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "SegMesh Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.output_text.append(f"-> Export segmask render:\n to {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a mesh or mask first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def set_spacing(self):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button:
            actor_name = checked_button.text()
            if actor_name in self.mesh_actors:
                spacing, ok = self.input_dialog.getText(self, 'Input', "Set Spacing", text=str(self.mesh_spacing))
                if ok:
                    try: self.mesh_spacing = ast.literal_eval(spacing)
                    except: QtWidgets.QMessageBox.warning(self, 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    self.add_mesh(actor_name, self.meshdict[actor_name])
            else:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select a mesh object instead", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select a mesh actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def set_scalar(self, nocs, actor_name):
        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_actors[actor_name])
        vertices_color = vertices
        if self.mirror_x: vertices_color = utils.transform_vertices(vertices_color, np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        if self.mirror_y: vertices_color = utils.transform_vertices(vertices_color, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        # get the corresponding color
        colors = utils.color_mesh(vertices_color, nocs=nocs)
        if colors.shape != vertices.shape: 
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Cannot set the selected color", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        assert colors.shape == vertices.shape, "colors shape should be the same as vertices shape"
        # color the mesh and actor
        mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        mesh = self.plotter.add_mesh(mesh_data, scalars=colors, rgb=True, opacity=self.mesh_opacity[actor_name], name=actor_name)
        transformation_matrix = pv.array_from_vtkmatrix(self.mesh_actors[actor_name].GetMatrix())
        mesh.user_matrix = transformation_matrix
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=actor_name)
        actor_colors = utils.get_mesh_actor_scalars(actor)
        assert (actor_colors == colors).all(), "actor_colors should be the same as colors"
        assert actor.name == actor_name, "actor's name should equal to actor_name"
        self.mesh_actors[actor_name] = actor

    def set_color(self, color, actor_name):
        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_actors[actor_name])
        mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        mesh = self.plotter.add_mesh(mesh_data, color=color, opacity=self.mesh_opacity[actor_name], name=actor_name)
        transformation_matrix = pv.array_from_vtkmatrix(self.mesh_actors[actor_name].GetMatrix())
        mesh.user_matrix = transformation_matrix
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=actor_name)
        assert actor.name == actor_name, "actor's name should equal to actor_name"
        self.mesh_actors[actor_name] = actor
        
    def nocs_epnp(self, color_mask, mesh):
        vertices = mesh.vertices
        pts3d, pts2d = utils.create_2d_3d_pairs(color_mask, vertices)
        pts2d = pts2d.astype('float32')
        pts3d = pts3d.astype('float32')
        camera_intrinsics = self.camera_intrinsics.astype('float32')
        predicted_pose = utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics, self.camera.position)
        return predicted_pose

    def latlon_epnp(self, color_mask, mesh):
        binary_mask = utils.color2binary_mask(color_mask)
        idx = np.where(binary_mask == 1)
        # swap the points for opencv, maybe because they handle RGB image differently (RGB -> BGR in opencv)
        idx = idx[:2][::-1]
        pts2d = np.stack((idx[0], idx[1]), axis=1)
        pts3d = []
        
        # Obtain the rg color
        color = color_mask[pts2d[:,1], pts2d[:,0]][..., :2]
        if np.max(color) > 1: color = color / 255
        gx = color[:, 0]
        gy = color[:, 1]

        lat = np.array(self.latlon[..., 0])
        lon = np.array(self.latlon[..., 1])
        lonf = lon[mesh.faces]
        msk = (np.sum(lonf>=0, axis=1)==3) & (np.sum(lat[mesh.faces]>=0, axis=1)==3)
        for i in range(len(pts2d)):
            pt = utils.latLon2xyz(mesh, lat, lonf, msk, gx[i], gy[i])
            pts3d.append(pt)
       
        pts3d = np.array(pts3d).reshape((len(pts3d), 3))

        pts2d = pts2d.astype('float32')
        pts3d = pts3d.astype('float32')
        camera_intrinsics = self.camera_intrinsics.astype('float32')
        
        predicted_pose = utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics, self.camera.position)

        return predicted_pose

    def epnp_mesh(self):
        if len(self.mesh_actors) == 1: self.reference = list(self.mesh_actors.keys())[0]
        if self.reference:
            colors = utils.get_mesh_actor_scalars(self.mesh_actors[self.reference])
            if colors is not None and (not np.all(colors == colors[0])):
                color_mask = self.export_mesh_render(save_render=False)
                gt_pose = self.mesh_actors[self.reference].user_matrix
                if self.mirror_x: gt_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                if self.mirror_y: gt_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose

                if np.sum(color_mask):
                    if self.mesh_colors[self.reference] == 'nocs':
                        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_actors[self.reference])
                        mesh = trimesh.Trimesh(vertices, faces, process=False)
                        predicted_pose = self.nocs_epnp(color_mask, mesh)
                        if self.mirror_x: predicted_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        if self.mirror_y: predicted_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        error = np.sum(np.abs(predicted_pose - gt_pose))
                        self.output_text.append(f"-> PREDICTED POSE WITH <span style='background-color:yellow; color:black;'>NOCS COLOR</span>: ")
                        self.output_text.append(f"{predicted_pose}\nGT POSE: \n{gt_pose}\nERROR: \n{error}")
                    else:
                        QtWidgets.QMessageBox.warning(self, 'vision6D', "Only works using EPnP with latlon mask", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                else:
                    QtWidgets.QMessageBox.warning(self, 'vision6D', "The color mask is blank (maybe set the reference mesh wrong)", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            else:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "The mesh need to be colored, with gradient color", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "A mesh need to be loaded/mesh reference need to be set", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def epnp_mask(self, nocs_method):
        if self.mask_store.mask_actor:
            mask_data = self.mask_store.render_mask(camera=self.camera.copy())
            if np.max(mask_data) > 1: mask_data = mask_data / 255

            # current shown mask is binary mask
            if np.all(np.logical_or(mask_data == 0, mask_data == 1)):
                if len(self.mesh_actors) == 1: 
                    self.reference = list(self.mesh_actors.keys())[0]
                if self.reference:
                    colors = utils.get_mesh_actor_scalars(self.mesh_actors[self.reference])
                    if colors is not None and (not np.all(colors == colors[0])):
                        color_mask = self.export_mesh_render(save_render=False)
                        nocs_color = (self.mesh_colors[self.reference] == 'nocs')
                        gt_pose = self.mesh_actors[self.reference].user_matrix
                        if self.mirror_x: gt_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                        if self.mirror_y: gt_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_actors[self.reference])
                        mesh = trimesh.Trimesh(vertices, faces, process=False)
                    else:
                        QtWidgets.QMessageBox.warning(self, 'vision6D', "The mesh need to be colored, with gradient color", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                        return 0
                else: 
                    QtWidgets.QMessageBox.warning(self, 'vision6D', "A mesh need to be loaded/mesh reference need to be set", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    return 0
                color_mask = (color_mask * mask_data).astype(np.uint8)
            
            if np.sum(color_mask):
                if nocs_method == nocs_color:
                    if nocs_method: 
                        color_theme = 'NOCS'
                        predicted_pose = self.nocs_epnp(color_mask, mesh)
                        if self.mirror_x: predicted_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        if self.mirror_y: predicted_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                    else: 
                        color_theme = 'LATLON'
                        if self.mirror_x: color_mask = color_mask[:, ::-1, :]
                        if self.mirror_y: color_mask = color_mask[::-1, :, :]
                        predicted_pose = self.latlon_epnp(color_mask, mesh)
                    error = np.sum(np.abs(predicted_pose - gt_pose))
                    self.output_text.append(f"-> PREDICTED POSE WITH <span style='background-color:yellow; color:black;'>{color_theme} COLOR (MASKED)</span>: ")
                    self.output_text.append(f"{predicted_pose}\nGT POSE: \n{gt_pose}\nERROR: \n{error}")
                else:
                    QtWidgets.QMessageBox.warning(self,"vision6D", "Clicked the wrong method")
            else:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "The color mask is blank (maybe set the reference mesh wrong)", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(self,"vision6D", "please load a mask first")

    def copy_output_text(self):
        self.clipboard.setText(self.output_text.toPlainText())
        
    def clear_output_text(self):
        self.output_text.clear()

    def update_color_button_text(self, text, popup):
        self.color_button.setText(text)
        popup.close() # automatically close the popup window

    def show_color_popup(self):

        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button:
            actor_name = checked_button.text()
            if actor_name in self.mesh_actors:
                popup = widgets_gui.PopUpDialog(self, on_button_click=lambda text: self.update_color_button_text(text, popup))
                button_position = self.color_button.mapToGlobal(QPoint(0, 0))
                popup.move(button_position + QPoint(self.color_button.width(), 0))
                popup.exec_()

                text = self.color_button.text()
                self.mesh_colors[actor_name] = text
                if text == 'nocs': self.set_scalar(True, actor_name)
                elif text == 'latlon': self.set_scalar(False, actor_name)
                else: self.set_color(text, actor_name)
            else:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Only be able to color mesh actors", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def load_per_frame_info(self, save=False):
        if self.video_path:
            self.video_player.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.video_player.cap.read()
            if ret: 
                video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.add_image(video_frame)
                if save:
                    os.makedirs(pathlib.Path(self.video_path).parent / f"{pathlib.Path(self.video_path).stem}_vision6D", exist_ok=True)
                    os.makedirs(pathlib.Path(self.video_path).parent / f"{pathlib.Path(self.video_path).stem}_vision6D" / "frames", exist_ok=True)
                    output_frame_path = pathlib.Path(self.video_path).parent / f"{pathlib.Path(self.video_path).stem}_vision6D" / "frames" / f"frame_{self.current_frame}.png"
                    save_frame = PIL.Image.fromarray(video_frame)
                    
                    # save each frame
                    save_frame.save(output_frame_path)
                    self.output_text.append(f"-> Save frame {self.current_frame}: ({self.current_frame}/{self.video_player.frame_count})")
                    self.image_store.image_path = str(output_frame_path)

                    # save gt_pose for each frame
                    os.makedirs(pathlib.Path(self.video_path).parent / f"{pathlib.Path(self.video_path).stem}_vision6D" / "poses", exist_ok=True)
                    output_pose_path = pathlib.Path(self.video_path).parent / f"{pathlib.Path(self.video_path).stem}_vision6D" / "poses" / f"pose_{self.current_frame}.npy"
                    self.current_pose()
                    np.save(output_pose_path, self.transformation_matrix)
                    self.output_text.append(f"-> Save frame {self.current_frame} pose: \n{self.transformation_matrix}")
        elif self.folder_path:
            # save gt_pose for specific frame
            os.makedirs(pathlib.Path(self.folder_path) / "vision6D", exist_ok=True)
            os.makedirs(pathlib.Path(self.folder_path) / "vision6D" / "poses", exist_ok=True)
            output_pose_path = pathlib.Path(self.folder_path) / "vision6D" / "poses" / f"{pathlib.Path(self.pose_path).stem}.npy"
            self.current_pose()
            np.save(output_pose_path, self.transformation_matrix)
            self.output_text.append(f"-> Save frame {pathlib.Path(self.pose_path).stem} pose: \n{self.transformation_matrix}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
    
    def sample_video(self):
        if self.video_path:
            self.video_sampler = widgets_gui.VideoSampler(self.video_player, self.fps)
            res = self.video_sampler.exec_()
            if res == QtWidgets.QDialog.Accepted: self.fps = round(self.video_sampler.fps)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def play_video(self):
        if self.video_path:
            res = self.video_player.exec_()
            if res == QtWidgets.QDialog.Accepted:
                self.current_frame = self.video_player.current_frame
                self.output_text.append(f"-> Current frame is ({self.current_frame}/{self.video_player.frame_count})")
                self.play_video_button.setText(f"Play ({self.current_frame}/{self.video_player.frame_count})")
                self.load_per_frame_info()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
    
    def prev_frame(self):
        if self.video_path:
            current_frame = self.current_frame - self.fps
            if current_frame >= 0: 
                self.current_frame = current_frame
                self.output_text.append(f"-> Current frame is ({self.current_frame}/{self.video_player.frame_count})")
                pose_path = pathlib.Path(self.video_path).parent / f"{pathlib.Path(self.video_path).stem}_vision6D" / "poses" / f"pose_{self.current_frame}.npy"
                if os.path.isfile(pose_path): 
                    self.transformation_matrix = np.load(pose_path)
                    self.register_pose(self.transformation_matrix)
                    self.output_text.append(f"-> Load saved frame {self.current_frame} pose: \n{self.transformation_matrix}")
                else: self.output_text.append(f"-> No saved pose for frame {self.current_frame}")
                self.play_video_button.setText(f"Play ({self.current_frame}/{self.video_player.frame_count})")
                self.video_player.slider.setValue(self.current_frame)
                self.load_per_frame_info()
        elif self.folder_path:
            if self.current_frame > 0:
                self.current_frame -= 1
                self.add_folder()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def next_frame(self):
        if self.video_path:
            current_frame = self.current_frame + self.fps
            if current_frame <= self.video_player.frame_count: 
                # save pose from the previous frame 
                self.load_per_frame_info(save=True)
                self.current_frame = current_frame
                self.output_text.append(f"-> Current frame is ({self.current_frame}/{self.video_player.frame_count})")
                # load pose for the current frame if the pose exist
                pose_path = pathlib.Path(self.video_path).parent / f"{pathlib.Path(self.video_path).stem}_vision6D" / "poses" / f"pose_{self.current_frame}.npy"
                if os.path.isfile(pose_path): 
                    self.transformation_matrix = np.load(pose_path)
                    self.register_pose(self.transformation_matrix)
                    self.output_text.append(f"-> Load saved frame {self.current_frame} pose: \n{self.transformation_matrix}")
                self.play_video_button.setText(f"Play ({self.current_frame}/{self.video_player.frame_count})")
                self.video_player.slider.setValue(self.current_frame)
                self.load_per_frame_info()
        elif self.folder_path:
            if self.current_frame < self.total_count:
                self.current_frame += 1
                self.add_folder()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def delete_video_folder(self):
        # self.video_path and self.folder_path should be exclusive
        if self.video_path:
            self.output_text.append(f"-> Delete video {self.video_path} from vision6D")
            self.play_video_button.setText("Play Video")
            self.video_path = None
        elif self.folder_path:
            self.output_text.append(f"-> Delete folder {self.folder_path} from vision6D")
            self.folder_path = None
            
        self.current_frame = 0

    def draw_mask(self):
        def handle_output_path_change(output_path):
            if output_path:
                self.mask_store.mask_path = output_path
                self.mask_store.add_mask(self.mask_store.mask_path)
        if self.image_store.image_path:
            self.label_window = widgets_gui.LabelWindow(self.image_store.image_path)
            self.label_window.show()
            self.label_window.image_label.output_path_changed.connect(handle_output_path_change)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

