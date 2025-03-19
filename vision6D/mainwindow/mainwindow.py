'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: mainwindow.py
@time: 2023-07-03 20:33
@desc: the mainwindow to run application
'''

# General import
import os
import ast
os.environ["QT_API"] = "pyqt5" # Setting the Qt bindings for QtPy
import json
import pickle
import trimesh
import pathlib
import functools

import PIL.Image
import numpy as np
import pyvista as pv

# Qt5 import
from PyQt5 import QtWidgets, QtGui
from pyvistaqt import MainWindow
from PyQt5.QtCore import Qt

# self defined package import
from ..widgets import CustomQtInteractor
from ..widgets import SearchBar
from ..widgets import PnPWindow
from ..widgets import CustomImageButtonWidget
from ..widgets import CustomMeshButtonWidget
from ..widgets import CustomMaskButtonWidget
from ..widgets import GetPoseDialog
from ..widgets import GetMaskDialog
from ..widgets import CalibrationDialog
from ..widgets import DistanceInputDialog
from ..widgets import CameraPropsInputDialog
from ..widgets import MaskWindow
from ..widgets import LiveWireWindow
from ..widgets import SamWindow
from ..widgets import CustomGroupBox
from ..widgets import CameraControlWidget

from ..tools import utils
from ..tools import exception
from ..containers import Scene

from ..path import ICON_PATH, SAVE_ROOT

np.set_printoptions(suppress=True)
        
class MyMainWindow(MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        
        # Set up the main window layout
        self.setWindowTitle("Vision6D")
        self.setWindowIcon(QtGui.QIcon(str(ICON_PATH / 'logo.png')))
        # the vision6D window is maximized by default
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        self.setAcceptDrops(True)

        # Initialize
        self.workspace_path = ''
        
        self.image_button_group_actors = QtWidgets.QButtonGroup(self)
        self.mesh_button_group_actors = QtWidgets.QButtonGroup(self)
        self.mask_button_group_actors = QtWidgets.QButtonGroup(self)
        # Create the plotter
        self.create_plotter()

        self.output_text = QtWidgets.QTextEdit()

        # Add a QLabel as an overlay hint label
        self.hintLabel = QtWidgets.QLabel(self.plotter)
        self.hintLabel.setText("Drag and drop a file here (The feature only works on Windows)")
        self.hintLabel.setStyleSheet("""
                                    color: white; 
                                    background-color: rgba(0, 0, 0, 127); 
                                    padding: 10px;
                                    border: 2px dashed gray;
                                    """)
        self.hintLabel.setAlignment(Qt.AlignCenter)

        self.scene = Scene(self.plotter, self.output_text)
                
        self.toggle_hide_meshes_flag = False

        # Set bars
        self.set_panel_bar()
        self.set_menu_bars()

        # Set up the main layout with the left panel and the render window using QSplitter
        self.main_layout = QtWidgets.QHBoxLayout(self.main_widget)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.panel_widget)
        self.splitter.addWidget(self.plotter)
        self.splitter.setStretchFactor(0, 1) # for self.panel_widget
        self.splitter.setStretchFactor(1, 5) # for self.plotter

        self.main_layout.addWidget(self.splitter)

        # Show the plotter
        self.show_plot()

        # Shortcut key bindings
        self.key_bindings()

    def key_bindings(self):
        # Camera related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("c"), self).activated.connect(self.reset_camera)
        QtWidgets.QShortcut(QtGui.QKeySequence("z"), self).activated.connect(self.scene.zoom_out)
        QtWidgets.QShortcut(QtGui.QKeySequence("x"), self).activated.connect(self.scene.zoom_in)

        # Mask related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("t"), self).activated.connect(self.reset_mask)

        # Mesh related key bindings 
        QtWidgets.QShortcut(QtGui.QKeySequence("k"), self).activated.connect(self.reset_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("l"), self).activated.connect(self.update_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("s"), self).activated.connect(self.undo_actor_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("y"), self).activated.connect(lambda up=True: self.toggle_surface_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("u"), self).activated.connect(lambda up=False: self.toggle_surface_opacity(up))

        # todo: create the swith button for mesh and ct "ctrl + tap"
        QtWidgets.QShortcut(QtGui.QKeySequence("Tab"), self).activated.connect(self.scene.tap_toggle_opacity)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Tab"), self).activated.connect(self.scene.ctrl_tap_opacity)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+w"), self).activated.connect(self.clear_plot)

        QtWidgets.QShortcut(QtGui.QKeySequence("Up"), self).activated.connect(lambda up=True: self.key_next_image_button(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("Down"), self).activated.connect(lambda up=False: self.key_next_image_button(up))

    def reset_camera(self):
        self.plotter.camera = self.scene.camera.copy()
        self.on_link_mesh_button_toggle(checked=self.link_mesh_button.isChecked(), clicked=False)

    def handle_transformation_matrix(self, name, transformation_matrix):
        self.mesh_register(name, transformation_matrix)
        self.update_gt_pose()

    def mesh_register(self, name, pose):
        mesh_model = self.scene.mesh_container.meshes[name]
        mesh_model.actor.user_matrix = pose
        mesh_model.undo_poses.append(pose)
        mesh_model.undo_poses = mesh_model.undo_poses[-20:]

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])
    def update_gt_pose(self, input_pose=None):
        if self.link_mesh_button.isChecked():
            for mesh_name, mesh_model in self.scene.mesh_container.meshes.items():
                mesh_model.initial_pose = mesh_model.actor.user_matrix if input_pose is None else input_pose
                mesh_model.undo_poses.clear() # reset the undo_poses after updating the gt pose of a mesh object
                mesh_model.undo_poses.append(mesh_model.initial_pose)
                matrix = utils.get_actor_user_matrix(mesh_model)
                text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3], 
                matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3], 
                matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3],
                matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3])
                self.output_text.append(f"-> Update the {mesh_name} GT pose to:")
                self.output_text.append(text)
        else:
            mesh_model = self.scene.mesh_container.meshes[self.scene.mesh_container.reference]
            mesh_model.initial_pose = mesh_model.actor.user_matrix if input_pose is None else input_pose
            mesh_model.undo_poses.clear() # reset the undo_poses after updating the gt pose of a mesh object
            mesh_model.undo_poses.append(mesh_model.initial_pose)
            matrix = utils.get_actor_user_matrix(mesh_model)
            text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
            matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3], 
            matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3], 
            matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3],
            matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3])
            self.output_text.append(f"-> Update the {self.scene.mesh_container.reference} GT pose to:")
            self.output_text.append(text)

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])
    def reset_gt_pose(self):
        if self.link_mesh_button.isChecked():
            for mesh_name, mesh_model in self.scene.mesh_container.meshes.items():
                self.mesh_register(mesh_name, mesh_model.initial_pose)
                matrix = utils.get_actor_user_matrix(mesh_model)
                text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3], 
                matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3], 
                matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3],
                matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3])
                self.output_text.append(f"-> Reset the {mesh_name} GT pose to:")
                self.output_text.append(text)
        else:
            mesh_model = self.scene.mesh_container.meshes[self.scene.mesh_container.reference]
            self.mesh_register(self.scene.mesh_container.reference, mesh_model.initial_pose)
            matrix = utils.get_actor_user_matrix(mesh_model)
            text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
            matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3], 
            matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3], 
            matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3],
            matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3])
            self.output_text.append(f"-> Reset the {self.scene.mesh_container.reference} GT pose to:")
            self.output_text.append(text)
            
        self.reset_camera()

    def key_next_image_button(self, up=False):
        buttons = self.image_button_group_actors.buttons()
        checked_button = self.image_button_group_actors.checkedButton()
        if checked_button is not None:
            checked_button.setChecked(False)
            current_button_index = buttons.index(checked_button)
            if up: current_button_index = (current_button_index + 1) % len(buttons)
            else: current_button_index = (current_button_index - 1) % len(buttons)
            next_button = buttons[current_button_index]
            next_button.click()
            self.images_actors_group.scroll_area.ensureWidgetVisible(next_button)

    def showMaximized(self):
        super(MyMainWindow, self).showMaximized()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
            self.hintLabel.hide()  # Hide hint when dragging
        else:
            e.ignore()

    #^ Camera related
    @utils.require_attributes([('scene.image_container.reference', "Please load an image first!")])  
    def camera_calibrate(self):
        original_image = np.array(PIL.Image.open(self.scene.image_container.images[self.scene.image_container.reference].path), dtype='uint8')
        # make the the original image shape is [h, w, 3] to match with the rendered calibrated_image
        original_image = original_image[..., :3]
        if len(original_image.shape) == 2: original_image = original_image[..., None]
        if original_image.shape[-1] == 1: original_image = np.dstack((original_image, original_image, original_image))
        calibrated_image = np.array(self.scene.image_container.render_image(self.scene.image_container.reference,
                                                                            self.plotter.camera.copy()), 
                                                                            dtype='uint8')
        if original_image.shape != calibrated_image.shape:
            utils.display_warning("Original image shape is not equal to calibrated image shape!")
        else: CalibrationDialog(calibrated_image, original_image).exec_()

    def set_camera(self):
        dialog = CameraPropsInputDialog(
            line1=("Fx", self.scene.fx), 
            line2=("Fy", self.scene.fy), 
            line3=("Cx", self.scene.cx), 
            line4=("Cy", self.scene.cy), 
            line5=("Canvas Height", self.scene.canvas_height),
            line6=("Canvas Width", self.scene.canvas_width),
            line7=("Camera View Up", self.scene.cam_viewup))
        if dialog.exec():
            fx, fy, cx, cy, canvas_height, canvas_width, cam_viewup = dialog.getInputs()
            pre_fx = self.scene.fx
            pre_fy = self.scene.fy
            pre_cx = self.scene.cx
            pre_cy = self.scene.cy
            pre_canvas_height = self.scene.canvas_height
            pre_canvas_width = self.scene.canvas_width
            pre_cam_viewup = self.scene.cam_viewup
            if not (fx == '' or fy == '' or cx == '' or cy == '' or cam_viewup == ''):
                try:
                    self.scene.fx = ast.literal_eval(fx)
                    self.scene.fy = ast.literal_eval(fy)
                    self.scene.cx = ast.literal_eval(cx)
                    self.scene.cy = ast.literal_eval(cy)
                    self.scene.canvas_height = ast.literal_eval(canvas_height)
                    self.scene.canvas_width = ast.literal_eval(canvas_width)
                    self.scene.cam_viewup = ast.literal_eval(cam_viewup)
                    self.scene.set_camera_intrinsics(self.scene.fx, self.scene.fy, self.scene.cx, self.scene.cy, self.scene.canvas_height)
                    self.scene.set_camera_extrinsics(self.scene.cam_viewup)
                    if self.scene.image_container.reference is not None: 
                        self.scene.handle_image_click(self.scene.image_container.reference)
                    self.reset_camera()
                except:
                    self.scene.fx = pre_fx
                    self.scene.fy = pre_fy
                    self.scene.cx = pre_cx
                    self.scene.cy = pre_cy
                    self.scene.canvas_height = pre_canvas_height
                    self.scene.canvas_width = pre_canvas_width
                    self.scene.cam_viewup = pre_cam_viewup
                    utils.display_warning("Error occured, check the format of the input values")

    def add_image_file(self, image_paths='', prompt=False):
        if prompt:
            image_paths, _ = QtWidgets.QFileDialog().getOpenFileNames(None, "Open file", "", "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)")
        if image_paths:
            self.hintLabel.hide()
            # Set up the camera
            self.scene.set_camera_intrinsics(self.scene.fx, self.scene.fy, self.scene.cx, self.scene.cy, self.scene.canvas_height)
            self.scene.set_camera_extrinsics(self.scene.cam_viewup)
            for image_path in image_paths:
                image_model = self.scene.image_container.add_image_attributes(image_path)
                button_widget = CustomImageButtonWidget(image_model.name, image_path=image_model.path)
                button_widget.removeButtonClicked.connect(self.remove_image_button)
                image_model.opacity_spinbox = button_widget.double_spinbox
                button = button_widget.button
                button.setCheckable(True)
                button.clicked.connect(lambda _, name=image_model.name: self.scene.handle_image_click(name))
                self.image_button_group_actors.addButton(button_widget.button)
                self.images_actors_group.widget_layout.insertWidget(0, button_widget)
            self.check_image_button(image_model.name)
            self.reset_camera()

    def add_mask_file(self, mask_paths='', prompt=False):
        if prompt:
            mask_paths, _ = QtWidgets.QFileDialog().getOpenFileNames(None, "Open file", "", "Files (*.npy *.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)") 
        if mask_paths:
            self.hintLabel.hide()
            for mask_path in mask_paths:
                mask_model = self.scene.mask_container.add_mask(mask_source = mask_path,
                                                                fy = self.scene.fy,
                                                                cx = self.scene.cx,
                                                                cy = self.scene.cy)
                self.add_mask_button(mask_model.name)

    def set_mask(self):
        get_mask_dialog = GetMaskDialog()
        res = get_mask_dialog.exec_()
        if res == QtWidgets.QDialog.Accepted:
            if get_mask_dialog.mask_path: self.add_mask_file(get_mask_dialog.mask_path)
            else:
                user_text = get_mask_dialog.get_text()
                points = exception.set_data_format(user_text)
                if points is not None:
                    if points.shape[1] == 2:
                        os.makedirs(SAVE_ROOT / "output", exist_ok=True)
                        os.makedirs(SAVE_ROOT / "output" / "mask_points", exist_ok=True)
                        if self.scene.image_container.images[self.scene.image_container.reference]: 
                            mask_path = SAVE_ROOT / "output" / "mask_points" / f"{pathlib.Path(self.scene.image_container.images[self.scene.image_container.reference].path).stem}.npy"
                        else: 
                            mask_path = SAVE_ROOT / "output" / "mask_points" / "mask_points.npy"
                        np.save(mask_path, points)
                        self.add_mask_file(mask_path)
                    else:
                        utils.display_warning("It needs to be a n by 2 matrix")

    @utils.require_attributes([('scene.mask_container.reference', "Please load a mask first!")])  
    def reset_mask(self):
        mask_model = self.scene.mask_container.masks[self.scene.mask_container.reference]
        mask_model.actor.user_matrix = np.eye(4)

    def add_mesh_file(self, mesh_paths='', prompt=False):
        if prompt: 
            mesh_paths, _ = QtWidgets.QFileDialog().getOpenFileNames(None, "Open file", "", "Files (*.mesh *.ply *.stl *.obj *.off *.dae *.fbx *.3ds *.x3d)") 
        if mesh_paths:
            self.hintLabel.hide()
            # Set up the camera
            self.scene.set_camera_intrinsics(self.scene.fx, self.scene.fy, self.scene.cx, self.scene.cy, self.scene.canvas_height)
            self.scene.set_camera_extrinsics(self.scene.cam_viewup)
            for mesh_path in mesh_paths:
                mesh_model = self.scene.mesh_container.add_mesh_actor(mesh_source=mesh_path, transformation_matrix=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1e+3], [0, 0, 0, 1]]))
                button_widget = CustomMeshButtonWidget(mesh_model.name)
                button_widget.colorChanged.connect(lambda color, name=mesh_model.name: self.scene.mesh_color_value_change(name, color))
                button_widget.removeButtonClicked.connect(self.remove_mesh_button)
                mesh_model.color_button = button_widget.square_button
                mesh_model.color_button.setStyleSheet(f"background-color: {mesh_model.color}")
                mesh_model.opacity_spinbox = button_widget.double_spinbox
                mesh_model.opacity_spinbox.setValue(mesh_model.opacity)
                mesh_model.opacity_spinbox.valueChanged.connect(lambda value, name=mesh_model.name: self.scene.mesh_container.set_mesh_opacity(name, value))
                button = button_widget.button
                button.setCheckable(True)
                button.clicked.connect(lambda _, name=mesh_model.name, output_text=True: self.check_mesh_button(name, output_text))
                self.mesh_button_group_actors.addButton(button_widget.button)
                self.mesh_actors_group.widget_layout.insertWidget(0, button_widget)
            if self.scene.mesh_container.reference is None: self.set_camera_spinbox(indicator=True)
            if self.link_mesh_button.isChecked() and self.scene.mesh_container.reference is not None: self.on_link_mesh_button_toggle(checked=self.link_mesh_button.isChecked(), clicked=True)
            else: self.check_mesh_button(name=mesh_model.name, output_text=True)
            self.reset_camera()

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])   
    def set_spacing(self):
        mesh_model = self.scene.mesh_container.meshes[self.scene.mesh_container.reference]
        spacing, ok = QtWidgets.QInputDialog().getText(QtWidgets.QMainWindow(), 'Input', "Set Spacing", text=str(mesh_model.spacing))
        if ok:
            scaled_spacing = np.array(exception.set_spacing(spacing)) / np.array(mesh_model.spacing)
            vertices, _ = utils.get_mesh_actor_vertices_faces(mesh_model.actor)
            centroid = np.mean(vertices, axis=0) # Calculate the centroid
            offset = vertices - centroid
            scaled_offset = offset * scaled_spacing
            vertices = centroid + scaled_offset
            mesh_model.pv_obj.points = vertices

    def toggle_surface_opacity(self, up):
        checked_button = self.mesh_button_group_actors.checkedButton()
        if checked_button:
            name = checked_button.text()
            if name in self.scene.mesh_container.meshes: 
                change = 0.05
                if not up: change *= -1
                mesh_model = self.scene.mesh_container.meshes[name]
                current_opacity = mesh_model.opacity_spinbox.value()
                current_opacity += change
                current_opacity = np.clip(current_opacity, 0, 1)
                mesh_model.opacity_spinbox.setValue(current_opacity)
                
    def handle_hide_meshes_opacity(self, flag):
        checked_button = self.mesh_button_group_actors.checkedButton()
        checked_name = checked_button.text() if checked_button else None
        for button in self.mesh_button_group_actors.buttons():
            name = button.text()
            if name not in self.scene.mesh_container.meshes: continue
            if len(self.scene.mesh_container.meshes) != 1 and name == checked_name: continue
            mesh_model = self.scene.mesh_container.meshes[name]
            if flag: self.scene.mesh_container.set_mesh_opacity(name, 0)
            else: self.scene.mesh_container.set_mesh_opacity(name, mesh_model.previous_opacity)

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])   
    def toggle_hide_meshes_button(self):
        self.toggle_hide_meshes_flag = not self.toggle_hide_meshes_flag
        self.handle_hide_meshes_opacity(self.toggle_hide_meshes_flag)

    def add_pose_file(self, pose_path):
        if pose_path:
            self.hintLabel.hide()
            if isinstance(pose_path, list): transformation_matrix = np.array(pose_path)
            else: transformation_matrix = np.load(pose_path)
            # set the initial pose of the mesh to the loaded transformation matrix
            self.scene.mesh_container.meshes[self.scene.mesh_container.reference].initial_pose = transformation_matrix
            self.reset_gt_pose()

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])
    def set_pose(self):
        mesh_model = self.scene.mesh_container.meshes[self.scene.mesh_container.reference]
        get_pose_dialog = GetPoseDialog(utils.get_actor_user_matrix(mesh_model))
        res = get_pose_dialog.exec_()
        if res == QtWidgets.QDialog.Accepted:
            user_text = get_pose_dialog.get_text()
            if "," not in user_text:
                user_text = user_text.replace(" ", ",")
                user_text =user_text.strip().replace("[,", "[")
            input_pose = exception.set_data_format(user_text)
            if input_pose is not None:
                if input_pose.shape == (4, 4):
                    self.hintLabel.hide()
                    mesh_name = self.scene.mesh_container.reference # set the mesh to be the originally loaded mesh
                    mesh_model = self.scene.mesh_container.meshes[mesh_name]
                    transformation_matrix = utils.get_actor_user_matrix(mesh_model)
                    vertices, faces = mesh_model.source_obj.vertices, mesh_model.source_obj.faces
                    mesh_model.pv_obj = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
                    try:
                        mesh = self.plotter.add_mesh(mesh_model.pv_obj, color=mesh_model.color, opacity=mesh_model.opacity, pickable=True, name=mesh_name)
                    except ValueError:
                        self.scene.mesh_container.set_color(mesh_name, mesh_model.color)
                    mesh_model.actor = mesh
                    mesh_model.actor.user_matrix = transformation_matrix
                    self.scene.mesh_container.meshes[mesh_name] = mesh_model
                    mesh_model.undo_poses.clear()
                    mesh_model.undo_poses.append(transformation_matrix)
                    self.update_gt_pose(input_pose=input_pose)
                    self.set_camera_control_values(input_pose)
                    self.reset_gt_pose()
                else: 
                    utils.display_warning("It needs to be a 4 by 4 matrix")

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])   
    def undo_actor_pose(self):
        checked_button = self.mesh_button_group_actors.checkedButton()
        self.scene.mesh_container.get_poses_from_undo()
        checked_button.click()

    def resizeEvent(self, e):
        x = (self.plotter.size().width() - self.hintLabel.width()) // 2
        y = (self.plotter.size().height() - self.hintLabel.height()) // 2
        self.hintLabel.move(x, y)
        super().resizeEvent(e)

    # ^Menu
    def set_menu_bars(self):
        mainMenu = self.menuBar()
        
        # allow to add files
        fileMenu = mainMenu.addMenu('File')
        fileMenu.addAction('Add Workspace', functools.partial(self.add_workspace, prompt=True))
        fileMenu.addAction('Add Image', functools.partial(self.add_image_file, prompt=True))
        fileMenu.addAction('Add Mask', self.set_mask)
        fileMenu.addAction('Add Mesh', functools.partial(self.add_mesh_file, prompt=True))

        # allow to export files
        exportMenu = mainMenu.addMenu('Export')
        exportMenu.addAction('Workspace', self.export_workspace)
        exportMenu.addAction('Image', self.export_image)
        exportMenu.addAction('Mask', self.export_mask)
        exportMenu.addAction('Pose', self.export_pose)
        exportMenu.addAction('Mesh Render', self.export_mesh_render)
        # exportMenu.addAction('SegMesh Render', self.export_segmesh_render)
        exportMenu.addAction('Camera Info', self.export_camera_info)

    # ^Panel
    def set_panel_bar(self):
        # Create a left panel layout
        self.panel_widget = QtWidgets.QWidget()
        self.panel_layout = QtWidgets.QVBoxLayout(self.panel_widget)

        # Create a top panel bar with a toggle button
        self.panel_bar = QtWidgets.QMenuBar()
        self.toggle_action = QtWidgets.QAction("Panel", self)
        self.toggle_action.triggered.connect(self.toggle_panel)
        self.panel_bar.addAction(self.toggle_action)
        self.setMenuBar(self.panel_bar)

        self.panel_console()
        self.camera_control_console()
        self.panel_images_actors()
        self.panel_mesh_actors()
        self.panel_mask_actors()
        self.panel_output()

    def toggle_panel(self):
        if self.panel_widget.isVisible():
            # self.panel_widget width changes when the panel is visiable or hiden
            self.panel_widget_width = self.panel_widget.width()
            self.panel_widget.hide()
            x = (self.plotter.size().width() + self.panel_widget_width - self.hintLabel.width()) // 2
            y = (self.plotter.size().height() - self.hintLabel.height()) // 2
            self.hintLabel.move(x, y)
        else:
            self.panel_widget.show()
            x = (self.plotter.size().width() - self.panel_widget_width - self.hintLabel.width()) // 2
            y = (self.plotter.size().height() - self.hintLabel.height()) // 2
            self.hintLabel.move(x, y)

    def set_panel_row_column(self, row, column):
        column += 1
        if column % 3 == 0: 
            row += 1
            column = 0
        return row, column
    
    @utils.require_attributes([('scene.image_container.reference', 'Please load an image first!'), ('scene.mesh_container.reference', 'Please load a mesh first!')])
    def pnp_register(self):
        image = utils.get_image_actor_scalars(self.scene.image_container.images[self.scene.image_container.reference].actor)
        pnp_window = PnPWindow(image_source=image, 
                                    mesh_model=self.scene.mesh_container.meshes[self.scene.mesh_container.reference],
                                    camera_intrinsics=self.scene.camera_intrinsics.astype(np.float32))
        pnp_window.transformation_matrix_computed.connect(lambda transformation_matrix: self.handle_transformation_matrix(self.scene.mesh_container.reference, transformation_matrix))
    
    def on_pose_options_selection_change(self, option):
        if option == "Set Pose":
            self.set_pose()
        elif option == "PnP Register":
            self.pnp_register()
        elif option == "Reset GT Pose (k)":
            self.reset_gt_pose()
        elif option == "Update GT Pose (l)":
            self.update_gt_pose()
        elif option == "Undo Pose (s)":
            self.undo_actor_pose()

    def on_camera_options_selection_change(self, option):
        if option == "Set Camera":
            self.set_camera()
        elif option == "Reset Camera (c)":
            self.reset_camera()
        elif option == "Zoom In (x)":
            self.scene.zoom_in()
        elif option == "Zoom Out (z)":
            self.scene.zoom_out()
        elif option == "Calibrate":
            self.camera_calibrate()

    def panel_console(self):
        console_group = QtWidgets.QGroupBox("Console")       # self.display = QtWidgets.QGroupBox("Console")
        display_layout = QtWidgets.QVBoxLayout()
        display_layout.setContentsMargins(10, 15, 10, 5)

        #* Create the top widgets (layout)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Create Grid layout for function buttons
        top_grid_layout = QtWidgets.QGridLayout()

        row, column = 0, 0

        # Create a QPushButton that will act as a drop-down button and QMenu to act as the drop-down menu
        camera_options_button = QtWidgets.QPushButton("Set Camera")
        camera_options_menu = QtWidgets.QMenu()
        camera_options_menu.addAction("Set Camera", lambda: self.on_camera_options_selection_change("Set Camera"))
        camera_options_menu.addAction("Reset Camera (c)", lambda: self.on_camera_options_selection_change("Reset Camera (c)"))
        camera_options_menu.addAction("Zoom In (x)", lambda: self.on_camera_options_selection_change("Zoom In (x)"))
        camera_options_menu.addAction("Zoom Out (z)", lambda: self.on_camera_options_selection_change("Zoom Out (z)"))
        camera_options_menu.addAction("Calibrate", lambda: self.on_camera_options_selection_change("Calibrate"))
        camera_options_button.setMenu(camera_options_menu)
        top_grid_layout.addWidget(camera_options_button, row, column)

        pose_options_button = QtWidgets.QPushButton("Set Pose")
        pose_options_menu = QtWidgets.QMenu()
        pose_options_menu.addAction("Set Pose", lambda: self.on_pose_options_selection_change("Set Pose"))
        pose_options_menu.addAction("PnP Register", lambda: self.on_pose_options_selection_change("PnP Register"))
        pose_options_menu.addAction("Reset GT Pose (k)", lambda: self.on_pose_options_selection_change("Reset GT Pose (k)"))
        pose_options_menu.addAction("Update GT Pose (l)", lambda: self.on_pose_options_selection_change("Update GT Pose (l)"))
        pose_options_menu.addAction("Undo Pose (s)", lambda: self.on_pose_options_selection_change("Undo Pose (s)"))
        pose_options_button.setMenu(pose_options_menu)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(pose_options_button, row, column)

        # Other buttons
        clear_all_button = QtWidgets.QPushButton("Clear All")
        clear_all_button.clicked.connect(self.clear_plot)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(clear_all_button, row, column)
        
        top_grid_widget = QtWidgets.QWidget()
        top_grid_widget.setLayout(top_grid_layout)
        top_layout.addWidget(top_grid_widget)
        display_layout.addLayout(top_layout)
        console_group.setLayout(display_layout)
        self.panel_layout.addWidget(console_group)

    def set_mesh_pose(self):
        euler_angles = np.array([self.camera_rx_control.spin_box.value(), self.camera_ry_control.spin_box.value(), self.camera_rz_control.spin_box.value()])
        translation_vector = np.array([self.camera_tx_control.spin_box.value(), self.camera_ty_control.spin_box.value(), self.camera_tz_control.spin_box.value()])
        camera_control_matrix = utils.compose_transform(euler_angles, translation_vector)
        actor_actual_matrix = utils.get_actor_user_matrix(self.scene.mesh_container.meshes[self.scene.mesh_container.reference])
        offset_matrix = camera_control_matrix @ np.linalg.inv(actor_actual_matrix) # Compute the offset (change) matrix between the current camera control value and the true pose
        user_matrix = self.scene.mesh_container.meshes[self.scene.mesh_container.reference].actor.user_matrix
        new_matrix = offset_matrix @ user_matrix
        self.scene.mesh_container.meshes[self.scene.mesh_container.reference].actor.user_matrix = new_matrix
        self.scene.handle_mesh_click(name=self.scene.mesh_container.reference, output_text=True)
        self.reset_camera()

    def set_camera_spinbox(self, indicator):
        self.block_value_change_signal(self.camera_rx_control.spin_box, 0)
        self.block_value_change_signal(self.camera_ry_control.spin_box, 0)
        self.block_value_change_signal(self.camera_rz_control.spin_box, 0)
        self.block_value_change_signal(self.camera_tx_control.spin_box, 0)
        self.block_value_change_signal(self.camera_ty_control.spin_box, 0)
        self.block_value_change_signal(self.camera_tz_control.spin_box, 0)
        self.camera_rx_control.spin_box.setEnabled(indicator)
        self.camera_ry_control.spin_box.setEnabled(indicator)
        self.camera_rz_control.spin_box.setEnabled(indicator)
        self.camera_tx_control.spin_box.setEnabled(indicator)
        self.camera_ty_control.spin_box.setEnabled(indicator)
        self.camera_tz_control.spin_box.setEnabled(indicator)

    def camera_control_console(self):
        console_group = QtWidgets.QGroupBox("Camera Control")

        display_layout = QtWidgets.QVBoxLayout()
        display_layout.setContentsMargins(0, 15, 0, 5)
        top_layout = QtWidgets.QHBoxLayout()
        top_grid_layout = QtWidgets.QGridLayout()

        row, column = 0, 0
        self.camera_rx_control = CameraControlWidget("Rx", "(deg)", 180)
        self.camera_rx_control.spin_box.setSingleStep(1)
        self.camera_rx_control.spin_box.valueChanged.connect(self.set_mesh_pose)
        top_grid_layout.addWidget(self.camera_rx_control, row, column)

        row, column = self.set_panel_row_column(row, column)
        self.camera_ry_control = CameraControlWidget("Ry", "(deg)", 180)
        self.camera_ry_control.spin_box.setSingleStep(1)
        self.camera_ry_control.spin_box.valueChanged.connect(self.set_mesh_pose)
        top_grid_layout.addWidget(self.camera_ry_control, row, column)

        row, column = self.set_panel_row_column(row, column)
        self.camera_rz_control = CameraControlWidget("Rz", "(deg)", 180)
        self.camera_rz_control.spin_box.setSingleStep(1)
        self.camera_rz_control.spin_box.valueChanged.connect(self.set_mesh_pose)
        top_grid_layout.addWidget(self.camera_rz_control, row, column)

        row, column = self.set_panel_row_column(row, column)
        self.camera_tx_control = CameraControlWidget("Tx", "(mm)", 1e+4)
        self.camera_tx_control.spin_box.setSingleStep(0.1)
        self.camera_tx_control.spin_box.valueChanged.connect(self.set_mesh_pose)
        top_grid_layout.addWidget(self.camera_tx_control, row, column)

        row, column = self.set_panel_row_column(row, column)
        self.camera_ty_control = CameraControlWidget("Ty", "(mm)", 1e+4)
        self.camera_ty_control.spin_box.setSingleStep(0.1)
        self.camera_ty_control.spin_box.valueChanged.connect(self.set_mesh_pose)
        top_grid_layout.addWidget(self.camera_ty_control, row, column)

        row, column = self.set_panel_row_column(row, column)
        self.camera_tz_control = CameraControlWidget("Tz", "(mm)", 1e+4)
        self.camera_tz_control.spin_box.setSingleStep(10)
        self.camera_tz_control.spin_box.valueChanged.connect(self.set_mesh_pose)
        top_grid_layout.addWidget(self.camera_tz_control, row, column)

        # Disable the spinboxes before loading a mesh
        self.set_camera_spinbox(indicator=False)

        top_grid_widget = QtWidgets.QWidget()
        top_grid_widget.setLayout(top_grid_layout)
        top_layout.addWidget(top_grid_widget)
        display_layout.addLayout(top_layout)
        console_group.setLayout(display_layout)
        self.panel_layout.addWidget(console_group)

    # In your main class or wherever you're using panel_images_actors
    def panel_images_actors(self):
        func_options_button = QtWidgets.QPushButton("Func")
        func_options_button.setFixedSize(50, 20)
        func_options_menu = QtWidgets.QMenu()
        func_options_menu.addAction("Set Distance", self.set_distance2camera)
        mirror_menu = QtWidgets.QMenu("Mirror", func_options_menu)
        mirror_menu.addAction("x-axis", functools.partial(self.mirror_image, direction="x"))
        mirror_menu.addAction("y-axis", functools.partial(self.mirror_image, direction="y"))
        func_options_menu.addMenu(mirror_menu)
        func_options_menu.addAction("Clear", self.clear_image)
        func_options_button.setMenu(func_options_menu)

        self.images_actors_group = CustomGroupBox("Image", self)
        self.images_actors_group.addButtonClicked.connect(lambda image_path='', prompt=True: self.add_image_file(image_path, prompt))
        self.images_actors_group.add_button_to_header(func_options_button)
        self.panel_layout.addWidget(self.images_actors_group)

    def on_link_mesh_button_toggle(self, checked, clicked):
        if clicked and checked and self.scene.mesh_container.reference is not None:
            # First, compute the average translation and set an identify matrix for R
            ts = [mesh_model.actor.user_matrix[:3, 3] for mesh_model in self.scene.mesh_container.meshes.values()]
            average_t = np.mean(ts, axis=0)
            new_rt = np.eye(4)
            new_rt[:3, 3] = average_t
            for mesh_name, mesh_model in self.scene.mesh_container.meshes.items():
                # Compute the relative matrix for each mesh with respect to the new_rt
                matrix = mesh_model.actor.user_matrix
                relative_matrix = np.linalg.inv(new_rt) @ matrix
                vertices, faces = utils.get_mesh_actor_vertices_faces(mesh_model.actor)
                transformed_vertices = utils.transform_vertices(vertices, relative_matrix)
                mesh_model.pv_obj = pv.wrap(trimesh.Trimesh(transformed_vertices, faces, process=False))
                try:
                    mesh = self.plotter.add_mesh(mesh_model.pv_obj, color=mesh_model.color, opacity=mesh_model.opacity, pickable=True, name=mesh_name)
                except ValueError:
                    self.scene.mesh_container.set_color(mesh_name, mesh_model.color)
                mesh_model.actor = mesh
                mesh_model.actor.user_matrix = new_rt
                self.scene.mesh_container.meshes[mesh_name] = mesh_model
                mesh_model.undo_poses.clear()
                mesh_model.undo_poses.append(new_rt)
        elif checked and not clicked:
            for mesh_name, mesh_model in self.scene.mesh_container.meshes.items():
                if mesh_name == self.scene.mesh_container.reference: continue
                mesh_model.actor.user_matrix = self.scene.mesh_container.meshes[self.scene.mesh_container.reference].actor.user_matrix
            
    def panel_mesh_actors(self):
        func_options_button = QtWidgets.QPushButton("Func")
        func_options_button.setFixedSize(50, 20)
        func_options_menu = QtWidgets.QMenu()
        func_options_menu.addAction("Set Spacing", self.set_spacing)
        func_options_menu.addAction("Toggle Meshes", self.toggle_hide_meshes_button)
        mirror_menu = QtWidgets.QMenu("Mirror", func_options_menu)
        mirror_menu.addAction("x-axis", functools.partial(self.mirror_mesh, direction="x"))
        mirror_menu.addAction("y-axis", functools.partial(self.mirror_mesh, direction="y"))
        func_options_menu.addMenu(mirror_menu)
        func_options_menu.addAction("Clear", self.clear_mesh)
        func_options_button.setMenu(func_options_menu)

        self.link_mesh_button = QtWidgets.QPushButton("Link")
        self.link_mesh_button.setFixedSize(20, 20)
        self.link_mesh_button.setCheckable(True)
        self.link_mesh_button.setChecked(True)
        self.link_mesh_button.toggled.connect(lambda checked, clicked=True: self.on_link_mesh_button_toggle(checked, clicked))
        self.mesh_actors_group = CustomGroupBox("Mesh", self)
        self.mesh_actors_group.addButtonClicked.connect(lambda mesh_path='', prompt=True: self.add_mesh_file(mesh_path, prompt))
        self.mesh_actors_group.add_button_to_header(self.link_mesh_button)
        self.mesh_actors_group.add_button_to_header(func_options_button)
        self.panel_layout.addWidget(self.mesh_actors_group)

    def panel_mask_actors(self):
        draw_mask_button = QtWidgets.QPushButton("Draw")
        draw_mask_button.setFixedSize(20, 20)
        draw_options_menu = QtWidgets.QMenu()
        draw_options_menu.addAction("Set Mask", self.set_mask)
        draw_mask_menu = QtWidgets.QMenu("Draw Mask", draw_options_menu)
        draw_mask_menu.addAction("Free Hand", functools.partial(self.draw_mask, live_wire=False, sam=False))
        draw_mask_menu.addAction("Live Wire", functools.partial(self.draw_mask, live_wire=True, sam=False))
        draw_mask_menu.addAction("SAM", functools.partial(self.draw_mask, live_wire=False, sam=True))
        draw_options_menu.addMenu(draw_mask_menu)
        draw_options_menu.addAction("Reset Mask (t)", self.reset_mask)
        draw_mask_button.setMenu(draw_options_menu)

        func_options_button = QtWidgets.QPushButton("Func")
        func_options_button.setFixedSize(50, 20)
        func_options_menu = QtWidgets.QMenu()
        mirror_menu = QtWidgets.QMenu("Mirror", func_options_menu)
        mirror_menu.addAction("x-axis", functools.partial(self.mirror_mask, direction="x"))
        mirror_menu.addAction("y-axis", functools.partial(self.mirror_mask, direction="y"))
        func_options_menu.addMenu(mirror_menu)
        func_options_menu.addAction("Clear", self.clear_mask)
        func_options_button.setMenu(func_options_menu)

        self.mask_actors_group = CustomGroupBox("Mask", self)
        self.mask_actors_group.checkbox.setChecked(False)
        self.mask_actors_group.addButtonClicked.connect(lambda mask_path='', prompt=True: self.add_mask_file(mask_path, prompt))
        self.mask_actors_group.add_button_to_header(draw_mask_button)
        self.mask_actors_group.add_button_to_header(func_options_button)
        self.panel_layout.addWidget(self.mask_actors_group)

    #^ Panel Output
    def panel_output(self):
        # Add a spacer to the top of the main layout
        self.output = QtWidgets.QGroupBox("Output")
        output_layout = QtWidgets.QVBoxLayout()
        output_layout.setContentsMargins(10, 20, 10, 5)

        #* Create the top widgets (layout)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Create Grid layout for function buttons
        grid_layout = QtWidgets.QGridLayout()
        
        # Create a SearchBar for search bar
        self.search_bar = SearchBar()
        self.search_bar.setPlaceholderText("Search...")

        # Add a signal to the QLineEdit object to connect to a function
        self.search_bar.textChanged.connect(self.handle_search)
        self.search_bar.returnPressedSignal.connect(self.find_next)

        # Add the search bar to the layout
        grid_layout.addWidget(self.search_bar, 0, 0)
        
        # Create the set camera button
        copy_pixmap = QtGui.QPixmap(str(ICON_PATH / "copy.png"))
        copy_icon = QtGui.QIcon(copy_pixmap)
        copy_text_button = QtWidgets.QPushButton()
        copy_text_button.setIcon(copy_icon)  # Set copy icon
        copy_text_button.clicked.connect(self.copy_output_text)
        grid_layout.addWidget(copy_text_button, 0, 1)
        
        # Create the actor pose button
        reset_pixmap = QtGui.QPixmap(str(ICON_PATH / "reset.png"))
        reset_icon = QtGui.QIcon(reset_pixmap)
        reset_text_button = QtWidgets.QPushButton()
        reset_text_button.setIcon(reset_icon) # Set reset icon
        reset_text_button.clicked.connect(self.reset_output_text)
        grid_layout.addWidget(reset_text_button, 0, 2)

        grid_widget = QtWidgets.QWidget()
        grid_widget.setLayout(grid_layout)
        top_layout.addWidget(grid_widget)
        output_layout.addLayout(top_layout)

        # Access to the system clipboard
        self.clipboard = QtGui.QGuiApplication.clipboard()
        self.output_text.setReadOnly(True)
        output_layout.addWidget(self.output_text)
        self.output.setLayout(output_layout)
        self.panel_layout.addWidget(self.output)

    def toggle_group_content(self, group, checked):
        for child in group.findChildren(QtWidgets.QWidget):
            child.setVisible(checked)
        self.panel_layout.update()
        
    def handle_search(self, text):
        # If there's text in the search bar
        if text: self.highlight_keyword(text)
        # If the search bar is empty
        else: self.clear_highlight()
        
    def highlight_keyword(self, keyword):
        # Get QTextDocument from QTextEdit
        doc = self.output_text.document()

        # Default text format
        default_format = QtGui.QTextCharFormat()

        # Text format for highlighted words
        highlight_format = QtGui.QTextCharFormat()
        highlight_format.setBackground(QtGui.QBrush(QtGui.QColor("yellow")))
        highlight_format.setForeground(QtGui.QBrush(QtGui.QColor("black")))

        # Clear all previous highlights
        cursor = QtGui.QTextCursor(doc)
        cursor.beginEditBlock()
        block_format = cursor.blockFormat()
        cursor.select(QtGui.QTextCursor.Document)
        cursor.setBlockFormat(block_format)
        cursor.setCharFormat(default_format)
        cursor.clearSelection()
        cursor.endEditBlock()
        cursor.setPosition(0)

        # Loop through each occurrence of the keyword
        occurrence_found = False
        while not cursor.isNull() and not cursor.atEnd():
            cursor = doc.find(keyword, cursor)
            if not cursor.isNull():
                if not occurrence_found:
                    self.output_text.setTextCursor(cursor)
                    occurrence_found = True
                cursor.mergeCharFormat(highlight_format)
                
        if not occurrence_found:
            cursor.setPosition(0)
            self.output_text.setTextCursor(cursor)  # Set QTextEdit cursor to the beginning if no match found
    
    def find_next(self):
        keyword = self.search_bar.text()
        # Get the QTextCursor from the QTextEdit
        cursor = self.output_text.textCursor()
        # Move the cursor to the position after the current selection
        cursor.setPosition(cursor.position() + cursor.selectionEnd() - cursor.selectionStart())
        # Use the QTextDocument's find method to find the next occurrence
        found_cursor = self.output_text.document().find(keyword, cursor)
        if not found_cursor.isNull(): 
            self.output_text.setTextCursor(found_cursor)

    def clear_highlight(self):
        doc = self.output_text.document()
        default_format = QtGui.QTextCharFormat()
        cursor = QtGui.QTextCursor(doc)
        cursor.beginEditBlock()
        block_format = cursor.blockFormat()
        cursor.select(QtGui.QTextCursor.Document)
        cursor.setBlockFormat(block_format)
        cursor.setCharFormat(default_format)
        cursor.clearSelection()
        cursor.endEditBlock()
        cursor.setPosition(0)
        self.output_text.setTextCursor(cursor)  # Set QTextEdit cursor to the beginning

    #^ Plotter
    def create_plotter(self):
        self.frame = QtWidgets.QFrame()
        self.plotter = CustomQtInteractor(self.frame, self)
        self.signal_close.connect(self.plotter.close)

    def show_plot(self):
        self.plotter.enable_trackball_actor_style()
        self.plotter.add_axes(color='white') 
        self.plotter.add_camera_orientation_widget()
        self.plotter.show()
        self.show()
                    
    def check_image_button(self, name):
        button = next((btn for btn in self.image_button_group_actors.buttons() if btn.text() == name), None)
        if button: button.click()

    def block_value_change_signal(self, spinbox, value):
        spinbox.blockSignals(True)
        spinbox.setValue(value)
        spinbox.blockSignals(False)

    def set_camera_control_values(self, matrix):
        euler_angles, translation = utils.decompose_transform(matrix)
        self.block_value_change_signal(self.camera_rx_control.spin_box, euler_angles[0])
        self.block_value_change_signal(self.camera_ry_control.spin_box, euler_angles[1])
        self.block_value_change_signal(self.camera_rz_control.spin_box, euler_angles[2])
        self.block_value_change_signal(self.camera_tx_control.spin_box, translation[0])
        self.block_value_change_signal(self.camera_ty_control.spin_box, translation[1])
        self.block_value_change_signal(self.camera_tz_control.spin_box, translation[2])
                    
    def check_mesh_button(self, name, output_text):
        button = next((btn for btn in self.mesh_button_group_actors.buttons() if btn.text() == name), None)
        if button:
            button.setChecked(True)
            self.scene.handle_mesh_click(name=name, output_text=output_text)
            self.set_camera_control_values(utils.get_actor_user_matrix(self.scene.mesh_container.meshes[name]))
            self.on_link_mesh_button_toggle(checked=self.link_mesh_button.isChecked(), clicked=False)

    def check_mask_button(self, name):
        button = next((btn for btn in self.mask_button_group_actors.buttons() if btn.text() == name), None)
        if button: button.click()

    @utils.require_attributes([('scene.image_container.reference', "Please load an image first!")])  
    def mirror_image(self, direction):
        image_model = self.scene.image_container.images[self.scene.image_container.reference]
        if direction == 'x': 
            image_model.source_obj = image_model.source_obj[:, ::-1, :]
        elif direction == 'y': 
            image_model.source_obj = image_model.source_obj[::-1, :, :]
        self.check_image_button(image_model.name)

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])   
    def mirror_mesh(self, direction):
        mesh_model = self.scene.mesh_container.meshes[self.scene.mesh_container.reference]
        if (mesh_model.initial_pose != np.eye(4)).all(): 
            mesh_model.initial_pose = mesh_model.actor.user_matrix
        transformation_matrix = mesh_model.actor.user_matrix
        if direction == 'x': 
            transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        elif direction == 'y': 
            transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        mesh_model.actor.user_matrix = transformation_matrix
        self.check_mesh_button(name=mesh_model.name, output_text=True)

    @utils.require_attributes([('scene.mask_container.reference', "Please load a mask first!")])   
    def mirror_mask(self, direction):
        mask_model = self.scene.mask_container.masks[self.scene.mask_container.reference]
        transformation_matrix = mask_model.actor.user_matrix
        if direction == 'x': 
            transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        elif direction == 'y': 
            transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        mask_model.actor.user_matrix = transformation_matrix
        self.check_mask_button(name=mask_model.name)

    def add_mask_button(self, name):
        button_widget = CustomMaskButtonWidget(name)
        button_widget.colorChanged.connect(lambda color, name=name: self.scene.mask_color_value_change(name, color))
        button_widget.removeButtonClicked.connect(self.remove_mask_button)
        button = button_widget.button
        mask_model = self.scene.mask_container.masks[name]
        mask_model.opacity_spinbox = button_widget.double_spinbox
        mask_model.opacity_spinbox.setValue(mask_model.opacity)
        mask_model.opacity_spinbox.valueChanged.connect(lambda value, name=name: self.scene.mask_container.set_mask_opacity(name, value))
        mask_model.color_button = button_widget.square_button
        mask_model.color_button.setStyleSheet(f"background-color: {mask_model.color}")
        # check the button
        button.setCheckable(True)
        button.setChecked(False)
        button.clicked.connect(lambda _, name=name: self.handle_mask_click(name))
        self.mask_actors_group.widget_layout.insertWidget(0, button_widget)
        self.mask_button_group_actors.addButton(button)
        self.check_mask_button(name=name)

    def handle_mask_click(self, name):
        self.scene.mask_container.reference = name
        self.reset_camera()

    @utils.require_attributes([('scene.image_container.reference', "Please load an image first!")])  
    def draw_mask(self, live_wire=False, sam=False):
        def handle_output_path_change(output_path):
            if output_path:
                self.scene.mask_container.add_mask(mask_source=output_path,
                                                    fy = self.scene.fy,
                                                    cx = self.scene.cx,
                                                    cy = self.scene.cy)
                self.add_mask_button(self.scene.mask_container.reference)
        image = utils.get_image_actor_scalars(self.scene.image_container.images[self.scene.image_container.reference].actor)
        if sam: self.mask_window = SamWindow(image)
        elif live_wire: self.mask_window = LiveWireWindow(image)
        else: self.mask_window = MaskWindow(image)
        self.mask_window.mask_label.output_path_changed.connect(handle_output_path_change)

    def copy_output_text(self):
        self.clipboard.setText(self.output_text.toPlainText())
        
    def reset_output_text(self):
        self.output_text.clear()

    def add_workspace(self, workspace_path='', prompt=False):
        if prompt:
            workspace_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.json)")
        if workspace_path:
            self.clear_plot() # clear out everything before loading a workspace
            self.workspace_path = workspace_path
            self.hintLabel.hide()
            with open(str(self.workspace_path), 'r') as f: workspace = json.load(f)
            if 'image_path' in workspace and workspace['image_path'] is not None: self.add_image_file(image_path=SAVE_ROOT / pathlib.Path(*workspace['image_path'].split("\\")))
            if 'mask_path' in workspace and workspace['mask_path'] is not None: self.add_mask_file(mask_path=SAVE_ROOT / pathlib.Path(*workspace['mask_path'].split("\\")))
            if 'mesh_path' in workspace:
                meshes = workspace['mesh_path']
                for item in meshes: 
                    mesh_path, pose = meshes[item]
                    self.add_mesh_file(mesh_path = SAVE_ROOT / pathlib.Path(*mesh_path.split("\\")))
                    self.add_pose_file(pose)
            self.reset_camera()

    def export_workspace(self):
        workspace_dict = {"mesh_path": {}, "image_path": {}, "mask_path": {}}
        for mesh_model in self.scene.mesh_container.meshes.values():
            matrix = utils.get_actor_user_matrix(mesh_model)
            workspace_dict["mesh_path"][mesh_model.name] = (mesh_model.path, matrix.tolist())
        for image_model in self.scene.image_container.images.values():
            workspace_dict["image_path"][image_model.name] = image_model.path
        for mask_model in self.scene.mask_container.masks.values():
            workspace_dict["mask_path"][mask_model.name] = mask_model.path
        # write the dict to json file
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Mesh Files (*.json)")
        if output_path != "":
            with open(output_path, 'w') as f: json.dump(workspace_dict, f, indent=4)

    def remove_image_button(self, button):
        # Get the index of the button before removal
        buttons = self.image_button_group_actors.buttons()
        checked_button = self.image_button_group_actors.checkedButton()
        index = buttons.index(checked_button)
        
        # Remove the associated actor
        actor = self.scene.image_container.images[button.text()].actor
        self.plotter.remove_actor(actor)
        del self.scene.image_container.images[button.text()]
        self.image_button_group_actors.removeButton(button)
        self.remove_image_button_widget(button)
        self.scene.image_container.reference = None
        button.deleteLater()
        buttons = self.image_button_group_actors.buttons()
        if buttons:
            if index < len(buttons):
                next_button = buttons[index]
                next_button.click()
            else:
                buttons[-1].click()

    def remove_mask_button(self, button):
        # Get the index of the button before removal
        buttons = self.mask_button_group_actors.buttons()
        checked_button = self.mask_button_group_actors.checkedButton()
        index = buttons.index(checked_button)

        actor = self.scene.mask_container.masks[button.text()].actor
        self.plotter.remove_actor(actor)
        del self.scene.mask_container.masks[button.text()]
        self.mask_button_group_actors.removeButton(button)
        self.remove_mask_button_widget(button)
        self.scene.mask_container.reference = None
        button.deleteLater()
        buttons = self.mask_button_group_actors.buttons()
        if buttons:
            if index < len(buttons):
                next_button = buttons[index]
                next_button.click()
            else:
                buttons[-1].click()

    def remove_mesh_button(self, button):
        # Get the index of the button before removal
        buttons = self.mesh_button_group_actors.buttons()
        checked_button = self.mesh_button_group_actors.checkedButton()
        index = buttons.index(checked_button)

        # Remove the associated actor
        actor = self.scene.mesh_container.meshes[button.text()].actor
        self.plotter.remove_actor(actor)
        del self.scene.mesh_container.meshes[button.text()]
        self.mesh_button_group_actors.removeButton(button)
        self.remove_mesh_button_widget(button)
        self.scene.mesh_container.reference = None
        button.deleteLater()
        buttons = self.mesh_button_group_actors.buttons()
        if buttons:
            if index < len(buttons):
                next_button = buttons[index]
                next_button.click()
            else:
                buttons[-1].click()
        else:
            self.set_camera_spinbox(indicator=False)

    def clear_image(self):
        for button in self.image_button_group_actors.buttons(): self.remove_image_button(button)

    def clear_mesh(self):
        for button in self.mesh_button_group_actors.buttons(): self.remove_mesh_button(button)

    def clear_mask(self):
        for button in self.mask_button_group_actors.buttons(): self.remove_mask_button(button)

    def clear_plot(self):
        self.clear_image()
        self.clear_mesh()
        self.clear_mask()
        self.workspace_path = ''
        self.reset_output_text()
        self.hintLabel.show()
    
    def remove_image_button_widget(self, button):
        for i in range(self.images_actors_group.widget_layout.count()): 
            widget = self.images_actors_group.widget_layout.itemAt(i).widget()
            if widget is not None and hasattr(widget, 'button') and widget.button == button:
                self.images_actors_group.widget_layout.removeWidget(widget)
                widget.deleteLater()
                break
    
    def remove_mask_button_widget(self, button):
        for i in range(self.mask_actors_group.widget_layout.count()): 
            widget = self.mask_actors_group.widget_layout.itemAt(i).widget()
            if widget is not None and hasattr(widget, 'button') and widget.button == button:
                self.mask_actors_group.widget_layout.removeWidget(widget)
                widget.deleteLater()
                break

    def remove_mesh_button_widget(self, button):
        for i in range(self.mesh_actors_group.widget_layout.count()): 
            widget = self.mesh_actors_group.widget_layout.itemAt(i).widget()
            if widget is not None and hasattr(widget, 'button') and widget.button == button:
                self.mesh_actors_group.widget_layout.removeWidget(widget)
                widget.deleteLater()
                break
    
    @utils.require_attributes([('scene.image_container.reference', "Please load an image first!")])  
    def set_distance2camera(self):
        image_model = self.scene.image_container.images[self.scene.image_container.reference]
        dialog = DistanceInputDialog(title='Input', label='Set Objects distance to camera:', value=str(image_model.distance2camera), default_value=str(self.scene.fy))
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            distance = float(dialog.get_value())
            if self.scene.image_container.reference is not None:
                image_model = self.scene.image_container.images[self.scene.image_container.reference]
                image_model.distance2camera = distance
                pv_obj = pv.ImageData(dimensions=(image_model.width, image_model.height, 1), spacing=[1, 1, 1], origin=(0.0, 0.0, 0.0))
                pv_obj.point_data["values"] = image_model.source_obj.reshape((image_model.width * image_model.height, image_model.channel)) # order = 'C
                pv_obj = pv_obj.translate(-1 * np.array(pv_obj.center), inplace=False) # center the image at (0, 0)
                pv_obj = pv_obj.translate(-np.array([0, 0, pv_obj.center[-1]]), inplace=False) # very important, re-center it to [0, 0, 0]
                pv_obj = pv_obj.translate(np.array([0, 0, image_model.distance2camera]), inplace=False)
                if image_model.channel == 1: image_actor = self.plotter.add_mesh(pv_obj, cmap='gray', opacity=image_model.opacity, name=image_model.name)
                else: image_actor = self.plotter.add_mesh(pv_obj, rgb=True, opacity=image_model.opacity, pickable=False, name=image_model.name)
                image_model.actor = image_actor
                self.scene.image_container.images[image_model.name] = image_model
            if len(self.scene.mask_container.masks) > 0:
                for mask_name, mask_model in self.scene.mask_container.masks.items():
                    mask_model = self.scene.mask_container.masks[mask_name]
                    mask_model.pv_obj = mask_model.pv_obj.translate(-np.array([0, 0, mask_model.pv_obj.center[-1]]), inplace=False) # very important, re-center it to [0, 0, 0]
                    mask_model.pv_obj = mask_model.pv_obj.translate(np.array([0, 0, distance]), inplace=False)
                    mask_mesh = self.plotter.add_mesh(mask_model.pv_obj, color=mask_model.color, style='surface', opacity=mask_model.opacity, pickable=True, name=mask_name)
                    mask_model.actor = mask_mesh
                    self.scene.mask_container.masks[mask_name] = mask_model
            self.reset_camera()
        
    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])   
    def export_pose(self):
        os.makedirs(SAVE_ROOT / "export_pose", exist_ok=True)
        for mesh_name, mesh_model in self.scene.mesh_container.meshes.items():
            matrix = utils.get_actor_user_matrix(mesh_model)
            output_path = SAVE_ROOT / "export_pose" / (mesh_name + '.npy')
            np.save(output_path, matrix)
            self.output_text.append(f"Export {mesh_name} mesh pose to:\n {output_path}")

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])   
    def export_mesh_render(self, save_render=True):
        os.makedirs(SAVE_ROOT / "export_mesh_render", exist_ok=True)
        image = self.scene.mesh_container.render_mesh(name=self.scene.mesh_container.reference, camera=self.plotter.camera.copy(), width=self.scene.canvas_width, height=self.scene.canvas_height)
        if save_render:
            output_name = "export_" + self.scene.mesh_container.reference
            output_path = SAVE_ROOT / "export_mesh_render" / (output_name + '.png')
            while output_path.exists(): 
                output_name += "_copy"
                output_path = SAVE_ROOT / "export_mesh_render" / (output_name + ".png")
            rendered_image = PIL.Image.fromarray(image)
            rendered_image.save(output_path)
            self.output_text.append(f"-> Export mesh render to:\n {output_path}")
            return image
        
    @utils.require_attributes([('scene.mask_container.reference', "Please load a mask first!")])   
    def export_mask(self):
        os.makedirs(SAVE_ROOT / "export_mask", exist_ok=True)
        mask_model = self.scene.mask_container.masks[self.scene.mask_container.reference]
        output_name = "export_" + self.scene.mask_container.reference
        output_path = SAVE_ROOT / "export_mask" / (output_name + ".png")
        while output_path.exists(): 
            output_name += "_copy"
            output_path = SAVE_ROOT / "export_mask" / (output_name + ".png")
        # Update and store the transformed mask actor if there is any transformation
        self.scene.mask_container.update_mask(self.scene.mask_container.reference)
        image = self.scene.mask_container.render_mask(camera=self.plotter.camera.copy(), cx=self.scene.cx, cy=self.scene.cy)
        rendered_image = PIL.Image.fromarray(image)
        rendered_image.save(output_path)
        mask_model.path = output_path
        self.output_text.append(f"-> Export Mask render to:\n {output_path}")

    @utils.require_attributes([('scene.image_container.reference', "Please load an image first!")])   
    def export_image(self):
        os.makedirs(SAVE_ROOT / "export_image", exist_ok=True)
        image_rendered = self.scene.image_container.render_image(camera=self.plotter.camera.copy())
        rendered_image = PIL.Image.fromarray(image_rendered)
        output_name = "export_" + self.scene.image_container.reference
        output_path = SAVE_ROOT / "export_image" / (output_name + '.png')
        while output_path.exists():
            output_name += "_copy"
            output_path = SAVE_ROOT / "export_image" / (output_name + ".png")
        rendered_image.save(output_path)
        self.output_text.append(f"-> Export image render to:\n {output_path}")

    def export_camera_info(self):
        os.makedirs(SAVE_ROOT / "export_camera_info", exist_ok=True)
        output_path = SAVE_ROOT / "export_camera_info" / "camera_info.pkl"
        camera_intrinsics = np.array([[self.scene.fx, 0, self.scene.cx], [0, self.scene.fy, self.scene.cy], [0, 0, 1]], dtype=np.float32)
        camera_info = {'camera_intrinsics': camera_intrinsics, 'canvas_height': self.scene.canvas_height, 'canvas_width': self.scene.canvas_width}
        with open(output_path,"wb") as f: pickle.dump(camera_info, f)
        self.output_text.append(f"-> Export camera info to:\n {output_path}")
