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
os.environ["QT_API"] = "pyqt5" # Setting the Qt bindings for QtPy
import json
import math
import pickle
import trimesh
import pathlib
import functools

import PIL.Image
import numpy as np

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
from ..widgets import CustomBboxButtonWidget
from ..widgets import CustomMaskButtonWidget
from ..widgets import GetPoseDialog
from ..widgets import GetMaskDialog

from ..tools import utils
from ..tools import exception
from ..containers import Scene

from ..path import ICON_PATH, PKG_ROOT

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
        self.mask_button_group_actors = QtWidgets.QButtonGroup(self)
        self.mesh_button_group_actors = QtWidgets.QButtonGroup(self)
        self.bbox_button_group_actors = QtWidgets.QButtonGroup(self)
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
        self.splitter.setStretchFactor(1, 3) # for self.plotter

        self.main_layout.addWidget(self.splitter)

        # Show the plotter
        self.show_plot()

        # Shortcut key bindings
        self.key_bindings()

    def key_bindings(self):
        # Camera related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("c"), self).activated.connect(self.scene.camera_store.reset_camera)
        QtWidgets.QShortcut(QtGui.QKeySequence("z"), self).activated.connect(self.scene.camera_store.zoom_out)
        QtWidgets.QShortcut(QtGui.QKeySequence("x"), self).activated.connect(self.scene.camera_store.zoom_in)

        # Mask related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("t"), self).activated.connect(self.scene.mask_container.reset_mask)

        # Bbox related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("f"), self).activated.connect(self.scene.bbox_container.reset_bbox)

        # Mesh related key bindings 
        QtWidgets.QShortcut(QtGui.QKeySequence("k"), self).activated.connect(self.scene.reset_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("l"), self).activated.connect(self.scene.update_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("s"), self).activated.connect(self.scene.mesh_store.undo_actor_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("y"), self).activated.connect(lambda up=True: self.toggle_surface_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("u"), self).activated.connect(lambda up=False: self.toggle_surface_opacity(up))

        # todo: create the swith button for mesh and ct "ctrl + tap"
        QtWidgets.QShortcut(QtGui.QKeySequence("Tab"), self).activated.connect(self.scene.tap_toggle_opacity)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Tab"), self).activated.connect(self.scene.ctrl_tap_opacity)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+w"), self).activated.connect(self.clear_plot)

    def showMaximized(self):
        super(MyMainWindow, self).showMaximized()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
            self.hintLabel.hide()  # Hide hint when dragging
        else:
            e.ignore()

    def add_image_file(self, image_path='', prompt=False):
        if prompt:
            image_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)")
        if image_path:
            self.hintLabel.hide()
            image_data = self.scene.image_container.add_image(image_path)
            # add remove current image to removeMenu
            if image_data.name not in self.scene.track_image_actors:
                self.scene.track_image_actors.append(image_data.name)
                self.add_image_button(image_data.name)
            self.scene.camera_store.reset_camera()

    def add_mask_file(self, mask_path='', prompt=False):
        if prompt:
            mask_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.npy *.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)") 
        if mask_path:
            self.hintLabel.hide()
            mask_data = self.scene.mask_container.add_mask(mask_path)
            # Add remove current image to removeMenu
            if mask_data.name not in self.scene.track_mask_actors:
                self.scene.track_mask_actors.append('mask')
                self.add_mask_button(mask_data.name)

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
                        os.makedirs(PKG_ROOT.parent / "output", exist_ok=True)
                        os.makedirs(PKG_ROOT.parent / "output" / "mask_points", exist_ok=True)
                        if self.scene.image_store.image_path: mask_path = PKG_ROOT.parent / "output" / "mask_points" / f"{pathlib.Path(self.scene.image_store.image_path).stem}.npy"
                        else: mask_path = PKG_ROOT.parent / "output" / "mask_points" / "mask_points.npy"
                        np.save(mask_path, points)
                        self.add_mask_file(mask_path)
                    else:
                        utils.display_warning("It needs to be a n by 2 matrix")

    def add_bbox_file(self, bbox_path='', prompt=False):
        if prompt:
            bbox_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.npy)") 
        if bbox_path:
            self.hintLabel.hide()
            bbox_data = self.scene.bbox_container.add_bbox(bbox_path)
            # Add remove current image to removeMenu
            if bbox_data.name not in self.scene.track_bbox_actors:
                self.scene.track_bbox_actors.append(bbox_data.name)
                self.add_bbox_button(bbox_data.name)

    def add_mesh_file(self, mesh_path='', prompt=False):
        if prompt: 
            mesh_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.mesh *.ply *.stl *.obj *.off *.dae *.fbx *.3ds *.x3d)") 
        if mesh_path:
            self.hintLabel.hide()
            mesh_data = self.scene.mesh_store.add_mesh(mesh_source=mesh_path)
            if mesh_data: 
                if self.scene.mesh_store.reference is not None:
                    reference_matrix = self.scene.mesh_store.meshes[self.scene.mesh_store.reference].actor.user_matrix
                    mesh_data = self.scene.mesh_container.add_mesh(mesh_data, reference_matrix)
                else:
                    mesh_data = self.scene.mesh_container.add_mesh(mesh_data, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1e+3], [0, 0, 0, 1]])) # set the initial pose, r_x, r_y, t_z includes the scaling too
            else: utils.display_warning("The mesh format is not supported!")
            # add remove current mesh to removeMenu
            if mesh_data.name not in self.scene.track_mesh_actors:
                self.scene.track_mesh_actors.append(mesh_data.name)
                self.add_mesh_button(mesh_data.name, self.output_text)
            #* very important for mirroring
            self.check_mesh_button(name=mesh_data.name, output_text=False) 
            self.scene.camera_store.reset_camera()

    def mirror_mesh(self, name, direction):
        mesh_data = self.scene.mesh_store.meshes[name]
        if direction == 'x': mesh_data.mirror_x = not mesh_data.mirror_x
        elif direction == 'y': mesh_data.mirror_y = not mesh_data.mirror_y
        if (mesh_data.initial_pose != np.eye(4)).all(): mesh_data.initial_pose = mesh_data.actor.user_matrix
        transformation_matrix = mesh_data.actor.user_matrix
        if mesh_data.mirror_x: 
            transformation_matrix[0,0] *= -1
            mesh_data.mirror_x = False # very important
        if mesh_data.mirror_y: 
            transformation_matrix[2,2] *= -1
            mesh_data.mirror_y = False # very important
        mesh_data.actor.user_matrix = transformation_matrix
        self.check_mesh_button(name=mesh_data.name, output_text=False)

    def set_spacing(self):
        checked_button = self.mesh_button_group_actors.checkedButton()
        if checked_button:
            name = checked_button.text()
            if name in self.scene.mesh_store.meshes:
                mesh_data = self.scene.mesh_store.meshes[name]
                spacing, ok = QtWidgets.QInputDialog().getText(QtWidgets.QMainWindow(), 'Input', "Set Spacing", text=str(mesh_data.spacing))
                if ok:
                    mesh_data.spacing = exception.set_spacing(spacing)
                    # Calculate the centroid
                    centroid = np.mean(mesh_data.source_mesh.vertices, axis=0)
                    offset = mesh_data.source_mesh.vertices - centroid
                    scaled_offset = offset * mesh_data.spacing
                    vertices = centroid + scaled_offset
                    mesh_data.pv_mesh.points = vertices
            else: utils.display_warning("Need to select a mesh object instead")
        else: utils.display_warning("Need to select a mesh actor first")

    def toggle_surface_opacity(self, up):
        checked_button = self.mesh_button_group_actors.checkedButton()
        if checked_button:
            name = checked_button.text()
            if name in self.scene.mesh_store.meshes: 
                change = 0.05
                if not up: change *= -1
                current_opacity = self.scene.mesh_store.meshes[name].opacity_spinbox.value()
                current_opacity += change
                current_opacity = np.clip(current_opacity, 0, 1)
                self.scene.mesh_store.meshes[name].opacity_spinbox.setValue(current_opacity)
                
    def handle_hide_meshes_opacity(self, flag):
        checked_button = self.mesh_button_group_actors.checkedButton()
        checked_name = checked_button.text() if checked_button else None
        for button in self.mesh_button_group_actors.buttons():
            name = button.text()
            if name not in self.scene.mesh_store.meshes: continue
            if len(self.scene.mesh_store.meshes) != 1 and name == checked_name: continue
            mesh_data = self.scene.mesh_store.meshes[name]
            if flag: self.scene.mesh_container.set_mesh_opacity(name, 0)
            else: self.scene.mesh_container.set_mesh_opacity(name, mesh_data.previous_opacity)
            
    def toggle_hide_meshes_button(self):
        self.toggle_hide_meshes_flag = not self.toggle_hide_meshes_flag
        self.handle_hide_meshes_opacity(self.toggle_hide_meshes_flag)

    def add_pose_file(self, pose_path):
        if pose_path:
            self.hintLabel.hide()
            if isinstance(pose_path, list): transformation_matrix = np.array(pose_path)
            else: transformation_matrix = np.load(pose_path)
            mesh_data = self.scene.mesh_store.meshes[self.scene.mesh_store.reference]
            if mesh_data.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            if mesh_data.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            self.scene.add_pose(matrix=transformation_matrix)

    def set_pose(self):
        if self.scene.mesh_store.reference:
            mesh_data = self.scene.mesh_store.meshes[self.scene.mesh_store.reference]
            get_pose_dialog = GetPoseDialog(mesh_data.actor.user_matrix)
            res = get_pose_dialog.exec_()
            if res == QtWidgets.QDialog.Accepted:
                user_text = get_pose_dialog.get_text()
                if "," not in user_text:
                    user_text = user_text.replace(" ", ",")
                    user_text =user_text.strip().replace("[,", "[")
                gt_pose = exception.set_data_format(user_text, mesh_data.actor.user_matrix)
                if gt_pose.shape == (4, 4):
                    self.hintLabel.hide()
                    transformation_matrix = gt_pose
                    if mesh_data.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                    if mesh_data.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                    self.scene.add_pose(matrix=transformation_matrix)
                else: utils.display_warning("It needs to be a 4 by 4 matrix")
        else: utils.display_warning("Needs to select a mesh first")

    def undo_actor_pose(self):
        if self.button_group_actors.checkedButton():
            name = self.button_group_actors.checkedButton().text()
            self.scene.mesh_store.undo_actor_pose(name)
            self.check_mesh_button(name=name) # very important, donnot change this line to "toggle_register"
        else: utils.display_warning("Choose a mesh actor first")

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isdir(file_path): self.add_folder(folder_path=file_path)
            else:
                # Load workspace json file
                if file_path.endswith(('.json')): self.add_workspace(workspace_path=file_path)
                # Load mesh file
                elif file_path.endswith(('.mesh', '.ply', '.stl', '.obj', '.off', '.dae', '.fbx', '.3ds', '.x3d')):
                    self.scene.mesh_container.add_mesh_file(mesh_path=file_path)
                # Load video file
                elif file_path.endswith(('.avi', '.mp4', '.mkv', '.mov', '.fly', '.wmv', '.mpeg', '.asf', '.webm')):
                    self.scene.video_container.add_video_file(video_path=file_path)
                # Load image/mask file
                elif file_path.endswith(('.png', '.jpg', 'jpeg', '.tiff', '.bmp', '.webp', '.ico')):  # add image/mask
                    file_data = np.array(PIL.Image.open(file_path).convert('L'), dtype='uint8')
                    unique, _ = np.unique(file_data, return_counts=True)
                    if len(unique) == 2: self.scene.mask_container.add_mask_file(mask_path=file_path)
                    else: self.add_image_file(image_path=file_path) 
                        
                elif file_path.endswith('.npy'): self.scene.mesh_container.add_pose_file(pose_path=file_path)
                else: utils.display_warning("File format is not supported!")

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
        fileMenu.addAction('Add Bbox', functools.partial(self.add_bbox_file, prompt=True))
        fileMenu.addAction('Add Mesh', functools.partial(self.add_mesh_file, prompt=True))
        fileMenu.addAction('Clear', self.clear_plot)

        # allow to export files
        exportMenu = mainMenu.addMenu('Export')
        exportMenu.addAction('Workspace', self.export_workspace)
        exportMenu.addAction('Image', self.export_image)
        exportMenu.addAction('Mask', self.export_mask)
        exportMenu.addAction('Bbox', self.export_bbox)
        exportMenu.addAction('Mesh/Pose', self.export_mesh_pose)
        exportMenu.addAction('Mesh Render', self.export_mesh_render)
        exportMenu.addAction('SegMesh Render', self.export_segmesh_render)
        exportMenu.addAction('Camera Info', self.export_camera_info)

    def draw_menu(self, event):
        context_menu = QtWidgets.QMenu(self)

        set_distance = QtWidgets.QAction('Set Distance', self)
        set_distance.triggered.connect(self.set_distance2camera)

        set_mask_act = QtWidgets.QAction('Set Mask', self)
        set_mask_act.triggered.connect(self.set_mask)

        draw_mask_menu = QtWidgets.QMenu('Draw Mask', self)  # Create a submenu for 'Draw Mask'
        free_hand = QtWidgets.QAction('Free Hand', self)
        free_hand.triggered.connect(functools.partial(self.scene.mask_container.draw_mask, live_wire=False, sam=False))  # Connect to a slot
        live_wire = QtWidgets.QAction('Live Wire', self)
        live_wire.triggered.connect(functools.partial(self.scene.mask_container.draw_mask, live_wire=True, sam=False))  # Connect to a slot
        sam = QtWidgets.QAction('SAM', self)
        sam.triggered.connect(functools.partial(self.scene.mask_container.draw_mask, live_wire=False, sam=True))  # Connect to another slot
        draw_mask_menu.addAction(free_hand)
        draw_mask_menu.addAction(live_wire)
        draw_mask_menu.addAction(sam)
        
        draw_bbox = QtWidgets.QAction('Draw BBox', self)
        draw_bbox.triggered.connect(self.scene.bbox_container.draw_bbox)
        
        reset_mask = QtWidgets.QAction('Reset Mask (t)', self)
        reset_mask.triggered.connect(self.scene.mask_container.reset_mask)
        reset_bbox = QtWidgets.QAction('Reset Bbox (f)', self)
        reset_bbox.triggered.connect(self.scene.bbox_container.reset_bbox)
        
        context_menu.addAction(set_distance)
        context_menu.addAction(set_mask_act)
        context_menu.addMenu(draw_mask_menu)
        context_menu.addAction(draw_bbox)
        context_menu.addAction(reset_mask)
        context_menu.addAction(reset_bbox)

        # Popup the menu
        context_menu.popup(QtGui.QCursor.pos())

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
        self.panel_images_actors()
        self.panel_mesh_actors()
        self.panel_mask_actors()
        self.panel_bbox_actors()
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
        if column % 4 == 0: 
            row += 1
            column = 0
        return row, column
    
    def pnp_register(self):
        if not self.scene.image_store.reference: utils.display_warning("Need to load an image first!"); return
        if self.scene.mesh_store.reference is None: utils.display_warning("Need to select a mesh first!"); return
        image = utils.get_image_actor_scalars(self.scene.image_store.reference)
        self.pnp_window = PnPWindow(image_source=image, 
                                    mesh_data=self.scene.mesh_store.meshes[self.scene.mesh_store.reference],
                                    camera_intrinsics=self.scene.image_store.camera_intrinsics.astype(np.float32))
        self.pnp_window.transformation_matrix_computed.connect(self.scene.handle_transformation_matrix)
    
    def on_pose_options_selection_change(self, option):
        if option == "Set Pose":
            self.scene.mesh_container.set_pose()
        elif option == "PnP Register":
            self.pnp_register()
        elif option == "Reset GT Pose (k)":
            self.scene.mesh_container.reset_gt_pose()
        elif option == "Update GT Pose (l)":
            self.scene.mesh_container.update_gt_pose()
        elif option == "Undo Pose (s)":
            self.scene.mesh_container.undo_actor_pose()

    def panel_console(self):
        self.console_group = QtWidgets.QGroupBox("Console")       # self.display = QtWidgets.QGroupBox("Console")
        self.display_layout = QtWidgets.QVBoxLayout()
        self.display_layout.setContentsMargins(10, 20, 10, 5)

        #* Create the top widgets (layout)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Create Grid layout for function buttons
        top_grid_layout = QtWidgets.QGridLayout()

        row, column = 0, 0

        # Create a QPushButton that will act as a drop-down button and QMenu to act as the drop-down menu
        self.camera_options_button = QtWidgets.QPushButton("Set Camera")
        self.camera_options_menu = QtWidgets.QMenu()
        self.camera_options_menu.addAction("Set Camera", lambda: self.scene.on_camera_options_selection_change("Set Camera"))
        self.camera_options_menu.addAction("Reset Camera (c)", lambda: self.scene.on_camera_options_selection_change("Reset Camera (c)"))
        self.camera_options_menu.addAction("Zoom In (x)", lambda: self.scene.on_camera_options_selection_change("Zoom In (x)"))
        self.camera_options_menu.addAction("Zoom Out (z)", lambda: self.scene.on_camera_options_selection_change("Zoom Out (z)"))
        self.camera_options_menu.addAction("Calibrate", lambda: self.scene.on_camera_options_selection_change("Calibrate"))
        self.camera_options_button.setMenu(self.camera_options_menu)
        top_grid_layout.addWidget(self.camera_options_button, row, column)

        self.pose_options_button = QtWidgets.QPushButton("Set Pose")
        self.pose_options_menu = QtWidgets.QMenu()
        self.pose_options_menu.addAction("Set Pose", lambda: self.scene.on_pose_options_selection_change("Set Pose"))
        self.pose_options_menu.addAction("PnP Register", lambda: self.scene.on_pose_options_selection_change("PnP Register"))
        self.pose_options_menu.addAction("Reset GT Pose (k)", lambda: self.scene.on_pose_options_selection_change("Reset GT Pose (k)"))
        self.pose_options_menu.addAction("Update GT Pose (l)", lambda: self.scene.on_pose_options_selection_change("Update GT Pose (l)"))
        self.pose_options_menu.addAction("Undo Pose (s)", lambda: self.scene.on_pose_options_selection_change("Undo Pose (s)"))
        self.pose_options_button.setMenu(self.pose_options_menu)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(self.pose_options_button, row, column)

        # Draw buttons
        self.draw_options_button = QtWidgets.QPushButton("Draw")
        self.draw_options_menu = QtWidgets.QMenu()
        self.draw_options_menu.addAction("Set Mask", self.set_mask)
        draw_mask_menu = QtWidgets.QMenu("Draw Mask", self.draw_options_menu)
        draw_mask_menu.addAction("Free Hand", functools.partial(self.scene.mask_container.draw_mask, live_wire=False, sam=False))
        draw_mask_menu.addAction("Live Wire", functools.partial(self.scene.mask_container.draw_mask, live_wire=True, sam=False))
        draw_mask_menu.addAction("SAM", functools.partial(self.scene.mask_container.draw_mask, live_wire=False, sam=True))
        self.draw_options_menu.addMenu(draw_mask_menu)
        self.draw_options_menu.addAction("Draw Bbox", self.scene.bbox_container.draw_bbox)
        self.draw_options_menu.addAction("Reset Mask (t)", self.scene.mask_container.reset_mask)
        self.draw_options_menu.addAction("Reset Bbox (f)", self.scene.bbox_container.reset_bbox)
        self.draw_options_button.setMenu(self.draw_options_menu)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(self.draw_options_button, row, column)

        # Other buttons
        self.other_options_button = QtWidgets.QPushButton("Other")
        self.other_options_menu = QtWidgets.QMenu()
        self.other_options_menu.addAction("Flip Left/Right", lambda direction="x": self.mirror_actors(direction))
        self.other_options_menu.addAction("Flip Up/Down", lambda direction="y": self.mirror_actors(direction))
        self.other_options_menu.addAction("Set Mesh Spacing", self.set_spacing)
        self.other_options_menu.addAction("Set Image Distance", self.set_distance2camera)
        self.other_options_menu.addAction("Toggle Meshes", self.toggle_hide_meshes_button)
        self.other_options_button.setMenu(self.other_options_menu)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(self.other_options_button, row, column)
        
        top_grid_widget = QtWidgets.QWidget()
        top_grid_widget.setLayout(top_grid_layout)
        top_layout.addWidget(top_grid_widget)
        self.display_layout.addLayout(top_layout)
        self.console_group.setLayout(self.display_layout)
        self.panel_layout.addWidget(self.console_group)

    def panel_images_actors(self):
        self.images_actors_group = QtWidgets.QGroupBox("Image")
        self.images_actors_group.setCheckable(True)
        self.images_actors_group.setChecked(True)
        self.images_actors_group.setStyleSheet("""
            QGroupBox::indicator:checked {
                background-color: green;
                border: 2px solid black;
                border-radius: 6px;
                padding: 2px;
            }
            QGroupBox::indicator:unchecked {
                background-color: red;
                border: 2px solid black;
                border-radius: 6px;
                padding: 2px;
            }
        """)
        self.images_actors_group.toggled.connect(lambda checked: self.toggle_group_content(self.images_actors_group, checked))
        self.display_layout = QtWidgets.QVBoxLayout()
        self.display_layout.setContentsMargins(10, 20, 10, 5)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.display_layout.addWidget(scroll_area)
        custom_widget_container = QtWidgets.QWidget()
        self.image_widget_layout = QtWidgets.QVBoxLayout()
        self.image_widget_layout.setSpacing(0)
        custom_widget_container.setLayout(self.image_widget_layout)
        self.image_widget_layout.addStretch()
        scroll_area.setWidget(custom_widget_container)
        self.images_actors_group.setLayout(self.display_layout)
        self.panel_layout.addWidget(self.images_actors_group)

    def panel_mesh_actors(self):
        self.mesh_actors_group = QtWidgets.QGroupBox("Mesh")
        self.mesh_actors_group.setCheckable(True)
        self.mesh_actors_group.setChecked(True)
        self.mesh_actors_group.setStyleSheet("""
            QGroupBox::indicator:checked {
                background-color: green;
                border: 2px solid black;
                border-radius: 6px;
                padding: 2px;
            }
            QGroupBox::indicator:unchecked {
                background-color: red;
                border: 2px solid black;
                border-radius: 6px;
                padding: 2px;
            }
        """)
        self.mesh_actors_group.toggled.connect(lambda checked: self.toggle_group_content(self.mesh_actors_group, checked))
        self.display_layout = QtWidgets.QVBoxLayout()
        self.display_layout.setContentsMargins(10, 20, 10, 5)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.display_layout.addWidget(scroll_area)
        custom_widget_container = QtWidgets.QWidget()
        self.mesh_widget_layout = QtWidgets.QVBoxLayout()
        self.mesh_widget_layout.setSpacing(0)
        custom_widget_container.setLayout(self.mesh_widget_layout)
        self.mesh_widget_layout.addStretch()
        scroll_area.setWidget(custom_widget_container)
        self.mesh_actors_group.setLayout(self.display_layout)
        self.panel_layout.addWidget(self.mesh_actors_group)

    def panel_mask_actors(self):
        self.mask_actors_group = QtWidgets.QGroupBox("Mask")
        self.mask_actors_group.setCheckable(True)
        self.mask_actors_group.setStyleSheet("""
            QGroupBox::indicator:checked {
                background-color: green;
                border: 2px solid black;
                border-radius: 6px;
                padding: 2px;
            }
            QGroupBox::indicator:unchecked {
                background-color: red;
                border: 2px solid black;
                border-radius: 6px;
                padding: 2px;
            }
        """)
        self.mask_actors_group.toggled.connect(lambda checked: self.toggle_group_content(self.mask_actors_group, checked))
        self.display_layout = QtWidgets.QVBoxLayout()
        self.display_layout.setContentsMargins(10, 20, 10, 5)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.display_layout.addWidget(scroll_area)
        custom_widget_container = QtWidgets.QWidget()
        self.mask_widget_layout = QtWidgets.QVBoxLayout()
        self.mask_widget_layout.setSpacing(0)
        custom_widget_container.setLayout(self.mask_widget_layout)
        self.mask_widget_layout.addStretch()
        scroll_area.setWidget(custom_widget_container)
        self.mask_actors_group.setLayout(self.display_layout)
        self.panel_layout.addWidget(self.mask_actors_group)
        # set the mask group to be unchecked
        self.mask_actors_group.setChecked(False)
        self.panel_layout.update()

    def panel_bbox_actors(self):
        self.bbox_actors_group = QtWidgets.QGroupBox("Bbox")
        self.bbox_actors_group.setCheckable(True)
        self.bbox_actors_group.setChecked(True)
        self.bbox_actors_group.setStyleSheet("""
            QGroupBox::indicator:checked {
                background-color: green;
                border: 2px solid black;
                border-radius: 6px;
                padding: 2px;
            }
            QGroupBox::indicator:unchecked {
                background-color: red;
                border: 2px solid black;
                border-radius: 6px;
                padding: 2px;
            }
        """)
        self.bbox_actors_group.toggled.connect(lambda checked: self.toggle_group_content(self.bbox_actors_group, checked))
        self.display_layout = QtWidgets.QVBoxLayout()
        self.display_layout.setContentsMargins(10, 20, 10, 5)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.display_layout.addWidget(scroll_area)
        custom_widget_container = QtWidgets.QWidget()
        self.bbox_widget_layout = QtWidgets.QVBoxLayout()
        self.bbox_widget_layout.setSpacing(0)
        custom_widget_container.setLayout(self.bbox_widget_layout)
        self.bbox_widget_layout.addStretch()
        scroll_area.setWidget(custom_widget_container)
        self.bbox_actors_group.setLayout(self.display_layout)
        self.panel_layout.addWidget(self.bbox_actors_group)
        # set the bbox group to be unchecked
        self.bbox_actors_group.setChecked(False)
        self.panel_layout.update()

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
                    
    def check_mesh_button(self, name, output_text):
        button = next((btn for btn in self.mesh_button_group_actors.buttons() if btn.text() == name), None)
        if button:
            button.setChecked(True)
            self.scene.handle_mesh_click(name=name, output_text=output_text)

    def check_image_button(self, name):
        button = next((btn for btn in self.image_button_group_actors.buttons() if btn.text() == name), None)
        if button:
            button.setChecked(True)
            self.scene.handle_image_click(name=name)

    def check_mask_button(self, name):
        button = next((btn for btn in self.mask_button_group_actors.buttons() if btn.text() == name), None)
        if button:
            button.setChecked(True)
            self.scene.handle_mask_click(name=name)

    def check_bbox_button(self, name):
        button = next((btn for btn in self.bbox_button_group_actors.buttons() if btn.text() == name), None)
        if button:
            button.setChecked(True)
            self.scene.handle_bbox_click(name=name)

    def add_image_button(self, name):
        button_widget = CustomImageButtonWidget(name, image_path=self.scene.image_store.images[name].image_path)
        button = button_widget.button
        self.scene.image_store.images[name].opacity_spinbox = button_widget.double_spinbox
        self.scene.image_store.images[name].opacity_spinbox.setValue(self.scene.image_store.images[name].opacity)
        self.scene.image_store.images[name].opacity_spinbox.valueChanged.connect(lambda value, name=name: self.scene.image_container.set_image_opacity(name, value))
        # check the button
        button.setCheckable(True)
        button.setChecked(True)
        button.clicked.connect(lambda _, name=name: self.scene.handle_image_click(name))
        self.image_widget_layout.insertWidget(0, button_widget)
        self.image_button_group_actors.addButton(button_widget.button)
        self.scene.handle_image_click(name=name)

    def add_mask_button(self, name):
        button_widget = CustomMaskButtonWidget(name)
        button_widget.colorChanged.connect(lambda color, name=name: self.scene.color_value_change(color, name))
        button = button_widget.button
        self.scene.mask_store.opacity_spinbox = button_widget.double_spinbox
        self.scene.mask_store.opacity_spinbox.setValue(self.scene.mask_store.mask_opacity)
        self.scene.mask_store.opacity_spinbox.valueChanged.connect(lambda value, name=name: self.scene.mask_container.set_mask_opacity(value))
        self.scene.mask_store.color_button = button_widget.square_button
        self.scene.mask_store.color_button.setStyleSheet(f"background-color: {self.scene.mask_store.color}")
        # check the button
        button.setCheckable(True)
        button.setChecked(True)
        button.clicked.connect(lambda _, name=name: self.scene.handle_mask_click(name))
        self.mask_widget_layout.insertWidget(0, button_widget)
        self.mask_button_group_actors.addButton(button)
        self.scene.handle_mask_click(name=name)

    def add_bbox_button(self, name):
        button_widget = CustomBboxButtonWidget(name)
        button_widget.colorChanged.connect(lambda color, name=name: self.scene.color_value_change(color, name))
        button = button_widget.button
        self.scene.bbox_store.opacity_spinbox = button_widget.double_spinbox
        self.scene.bbox_store.opacity_spinbox.setValue(self.scene.bbox_store.bbox_opacity)
        self.scene.bbox_store.opacity_spinbox.valueChanged.connect(lambda value, name=name: self.scene.bbox_container.set_bbox_opacity(value))
        self.scene.bbox_store.color_button = button_widget.square_button
        self.scene.bbox_store.color_button.setStyleSheet(f"background-color: {self.scene.bbox_store.color}")
        # check the button
        button.setCheckable(True)
        button.setChecked(True)
        button.clicked.connect(lambda _, name=name: self.scene.handle_bbox_click(name))
        self.bbox_widget_layout.insertWidget(0, button_widget)
        self.bbox_button_group_actors.addButton(button)
        self.scene.handle_bbox_click(name=name)

    def add_mesh_button(self, name, output_text):
        button_widget = CustomMeshButtonWidget(name)
        button_widget.colorChanged.connect(lambda color, name=name: self.scene.color_value_change(color, name))
        button = button_widget.button
        self.scene.mesh_store.meshes[name].opacity_spinbox = button_widget.double_spinbox
        self.scene.mesh_store.meshes[name].opacity_spinbox.setValue(self.scene.mesh_store.meshes[name].opacity)
        self.scene.mesh_store.meshes[name].opacity_spinbox.valueChanged.connect(lambda value, name=name: self.scene.mesh_container.set_mesh_opacity(name, value))
        self.scene.mesh_store.meshes[name].color_button = button_widget.square_button
        self.scene.mesh_store.meshes[name].color_button.setStyleSheet(f"background-color: {self.scene.mesh_store.meshes[name].color}")
        # check the button
        button.setCheckable(True)
        button.setChecked(True)
        button.clicked.connect(lambda _, name=name, output_text=output_text: self.scene.handle_mesh_click(name, output_text))
        self.mesh_widget_layout.insertWidget(0, button_widget)
        self.mesh_button_group_actors.addButton(button)
        self.scene.handle_mesh_click(name=name, output_text=output_text)

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
            root = PKG_ROOT.parent.parent.parent
            if 'image_path' in workspace and workspace['image_path'] is not None: self.add_image_file(image_path=root / pathlib.Path(*workspace['image_path'].split("\\")))
            if 'mask_path' in workspace and workspace['mask_path'] is not None: self.add_mask_file(mask_path=root / pathlib.Path(*workspace['mask_path'].split("\\")))
            if 'bbox_path' in workspace and workspace['bbox_path'] is not None: self.add_bbox_file(bbox_path=root / pathlib.Path(*workspace['bbox_path'].split("\\")))
            if 'mesh_path' in workspace:
                meshes = workspace['mesh_path']
                for item in meshes: 
                    mesh_path, pose = meshes[item]
                    self.scene.mesh_container.add_mesh_file(mesh_path=root / pathlib.Path(*mesh_path.split("\\")))
                    self.scene.mesh_container.add_pose_file(pose)
            self.scene.camera_store.reset_camera()

    def export_workspace(self):
        workspace_dict = {"mesh_path": {}}
        workspace_dict["image_path"] = self.scene.image_store.image_path
        workspace_dict["mask_path"] = self.scene.mask_store.mask_path
        workspace_dict["bbox_path"] = self.scene.bbox_store.bbox_path
        for mesh_data in self.scene.mesh_store.meshes.values(): 
            workspace_dict["mesh_path"][mesh_data.name] = (mesh_data.mesh_path, mesh_data.actor.user_matrix.tolist())
        # write the dict to json file
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Mesh Files (*.json)")
        if output_path != "":
            with open(output_path, 'w') as f: json.dump(workspace_dict, f, indent=4)

    def add_folder(self, folder_path='', prompt=False):
        if prompt: 
            folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            if self.scene.video_store.video_path or self.workspace_path: self.clear_plot() # main goal is to set video_path to None
            image_path, mask_path, pose_path, mesh_path = self.scene.folder_store.add_folder(folder_path=folder_path, meshes=self.scene.mesh_store.meshes)
            if image_path or mask_path or pose_path or mesh_path:
                if image_path: self.scene.image_container.add_image_file(image_path=image_path)
                if mask_path: self.scene.mask_container.add_mask_file(mask_path=mask_path)
                if mesh_path: 
                    with open(mesh_path, 'r') as f: mesh_path = f.read().splitlines()
                    for path in mesh_path: self.scene.mesh_container.add_mesh_file(path)
                if pose_path: self.scene.mesh_container.add_pose_file(pose_path=pose_path)
                self.scene.camera_store.reset_camera()
            else:
                self.scene.folder_store.reset()
                utils.display_warning("Not a valid folder, please reload a folder")

    def mirror_actors(self, direction):
        # checked_button = self.button_group_actors.checkedButton()
        # if checked_button:
        #     name = checked_button.text()
        #     if name in self.image_store.images: self.image_container.mirror_image(name, direction)
        #     elif name == 'mask': self.mask_container.mirror_mask(direction)
        #     elif name == 'bbox': self.bbox_container.mirror_bbox(direction)
        #     elif name in self.mesh_store.meshes: self.mesh_container.mirror_mesh(name, direction)
        # else: utils.display_warning("Need to select an actor first")
        pass

    def remove_actor(self, button):
        name = button.text()
        if name in self.scene.image_store.images: 
            actor = self.scene.image_store.images[name].actor
            self.track_image_actors.remove(name)
            self.scene.image_store.reset()
            # remove the button from the button group
            self.image_button_group_actors.removeButton(button)
            self.remove_image_button_widget(button)
            button.deleteLater()
        elif name == 'mask':
            actor = self.scene.mask_store.mask_actor
            self.track_mask_actors.remove(name)
            self.scene.mask_store.reset()
            self.mask_button_group_actors.removeButton(button)
            self.remove_mask_button_widget(button)
            button.deleteLater()
        elif name == 'bbox':
            actor = self.scene.bbox_store.bbox_actor
            self.track_bbox_actors.remove(name)
            self.scene.bbox_store.reset()
            self.bbox_button_group_actors.removeButton(button)
            self.remove_bbox_button_widget(button)
            button.deleteLater()
        elif name in self.scene.mesh_store.meshes: 
            actor = self.scene.mesh_store.meshes[name].actor
            self.track_mesh_actors.remove(name)
            self.scene.mesh_store.remove_mesh(name)
            self.mesh_button_group_actors.removeButton(button)
            self.remove_mesh_button_widget(button)
            button.deleteLater()

        self.plotter.remove_actor(actor)

        # clear out the plot if there is no actor
        if (self.scene.image_store.images is None) and (self.scene.mask_store.mask_actor is None) and (len(self.scene.mesh_store.meshes) == 0): 
            self.clear_plot()

    def remove_image_button(self, button):
        actor = self.scene.image_store.images[button.text()].actor
        self.scene.image_store.images[button.text()].reset()
        self.scene.image_store.images[button.text()].mirror_x = False
        self.scene.image_store.images[button.text()].mirror_y = False
        self.plotter.remove_actor(actor)
        self.image_button_group_actors.removeButton(button)
        self.remove_image_button_widget(button)
        button.deleteLater()

    def remove_mask_button(self, button):
        actor = self.scene.mask_store.mask_actor
        self.scene.mask_store.reset()
        self.scene.mask_store.mirror_x = False
        self.scene.mask_store.mirror_y = False
        self.plotter.remove_actor(actor)
        self.mask_button_group_actors.removeButton(button)
        self.remove_mask_button_widget(button)
        button.deleteLater()

    def remove_bbox_button(self, button):
        actor = self.scene.bbox_store.bbox_actor
        self.scene.bbox_store.reset()
        self.scene.bbox_store.mirror_x = False
        self.scene.bbox_store.mirror_y = False
        self.plotter.remove_actor(actor)
        self.bbox_button_group_actors.removeButton(button)
        self.remove_bbox_button_widget(button)
        button.deleteLater()

    def remove_mesh_button(self, button):
        actor = self.scene.mesh_store.meshes[button.text()].actor
        self.scene.mesh_store.remove_mesh(button.text())
        self.plotter.remove_actor(actor)
        self.mesh_button_group_actors.removeButton(button)
        self.remove_mesh_button_widget(button)
        button.deleteLater()

    def clear_plot(self):
        # Clear out everything in the remove menu
        for button in self.image_button_group_actors.buttons():
            actor = self.scene.image_store.images[button.text()].actor
            self.scene.image_store.images[button.text()].reset()
            self.scene.image_store.images[button.text()].mirror_x = False
            self.scene.image_store.images[button.text()].mirror_y = False
            self.plotter.remove_actor(actor)
            self.image_button_group_actors.removeButton(button)
            self.remove_image_button_widget(button)
            button.deleteLater()
        for button in self.mask_button_group_actors.buttons():
            actor = self.scene.mask_store.mask_actor
            self.scene.mask_store.reset()
            self.scene.mask_store.mirror_x = False
            self.scene.mask_store.mirror_y = False
            self.plotter.remove_actor(actor)
            self.image_button_group_actors.removeButton(button)
            self.remove_mask_button_widget(button)
            button.deleteLater()
        for button in self.bbox_button_group_actors.buttons():
            actor = self.scene.bbox_store.bbox_actor
            self.scene.bbox_store.reset()
            self.scene.bbox_store.mirror_x = False
            self.scene.bbox_store.mirror_y = False
            self.plotter.remove_actor(actor)
            self.image_button_group_actors.removeButton(button)
            self.remove_bbox_button_widget(button)
            button.deleteLater()
        for button in self.mesh_button_group_actors.buttons():
            actor = self.scene.mesh_store.meshes[button.text()].actor
            self.scene.mesh_store.remove_mesh(button.text())
            self.plotter.remove_actor(actor)
            self.image_button_group_actors.removeButton(button)
            self.remove_mesh_button_widget(button)
            button.deleteLater()

        self.scene.mesh_store.reset()
        self.scene.video_store.reset()
        self.scene.folder_store.reset()
        self.workspace_path = ''
        self.track_image_actors.clear()
        self.track_mask_actors.clear()
        self.track_bbox_actors.clear()
        self.track_mesh_actors.clear()
        self.reset_output_text()
        self.hintLabel.show()
    
    def remove_image_button_widget(self, button):
        for i in range(self.image_widget_layout.count()): 
            widget = self.image_widget_layout.itemAt(i).widget()
            if widget is not None and hasattr(widget, 'button') and widget.button == button:
                self.image_widget_layout.removeWidget(widget)
                widget.deleteLater()
                break
    
    def remove_mask_button_widget(self, button):
        for i in range(self.mask_widget_layout.count()): 
            widget = self.mask_widget_layout.itemAt(i).widget()
            if widget is not None and hasattr(widget, 'button') and widget.button == button:
                self.mask_widget_layout.removeWidget(widget)
                widget.deleteLater()
                break
    
    def remove_bbox_button_widget(self, button):
        for i in range(self.bbox_widget_layout.count()): 
            widget = self.bbox_widget_layout.itemAt(i).widget()
            if widget is not None and hasattr(widget, 'button') and widget.button == button:
                self.bbox_widget_layout.removeWidget(widget)
                widget.deleteLater()
                break

    def remove_mesh_button_widget(self, button):
        for i in range(self.mesh_widget_layout.count()): 
            widget = self.mesh_widget_layout.itemAt(i).widget()
            if widget is not None and hasattr(widget, 'button') and widget.button == button:
                self.mesh_widget_layout.removeWidget(widget)
                widget.deleteLater()
                break

    def set_distance2camera(self):
        distance, ok = QtWidgets.QInputDialog().getText(QtWidgets.QMainWindow(), 'Input', "Set Objects distance to camera", text=str(self.image_store.images[self.image_store.reference].distance2camera))
        if ok:
            self.scene.set_distance2camera(distance)
        
    def export_mesh_pose(self):
        for mesh_data in self.scene.mesh_store.meshes.values():
            verts, faces = utils.get_mesh_actor_vertices_faces(mesh_data.actor)
            vertices = utils.transform_vertices(verts, mesh_data.actor.user_matrix)
            os.makedirs(PKG_ROOT.parent / "output" / "export_mesh", exist_ok=True)
            output_path = PKG_ROOT.parent / "output" / "export_mesh" / (mesh_data.name + '.ply')
            mesh = trimesh.Trimesh(vertices, faces, process=False)
            ply_file = trimesh.exchange.ply.export_ply(mesh)
            with open(output_path, "wb") as fid: fid.write(ply_file)
            self.output_text.append(f"Export {mesh_data.name} mesh to:\n {output_path}")
                
    def export_mesh_render(self, save_render=True):
        image = None
        if self.scene.mesh_store.reference:
            image = self.render_mesh(camera=self.plotter.camera.copy())
            if save_render:
                output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Mesh Files (*.png)")
                if output_path:
                    if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.png')
                    rendered_image = PIL.Image.fromarray(image)
                    rendered_image.save(output_path)
                    self.output_text.append(f"-> Export mesh render to:\n {output_path}")
        else: utils.display_warning("Need to load a mesh first")
        return image

    def export_segmesh_render(self):
        if self.scene.mesh_store.reference and self.scene.mask_store.mask_actor:
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "SegMesh Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.png')
                mask_surface = self.scene.mask_store.update_mask()
                self.scene.mask_container.load_mask(mask_surface)
                segmask = self.scene.mask_store.render_mask(camera=self.plotter.camera.copy())
                if np.max(segmask) > 1: segmask = segmask / 255
                image = self.scene.mesh_container.render_mesh(camera=self.plotter.camera.copy())
                image = (image * segmask).astype(np.uint8)
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.output_text.append(f"-> Export segmask render:\n to {output_path}")
        else: utils.display_warning("Need to load a mesh or mask first")

    def export_bbox(self):
        if self.scene.bbox_store.actor:
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Bbox Files (*.npy)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.png')
                # Store the transformed bbox actor if there is any transformation
                points = utils.get_bbox_actor_points(self.scene.bbox_store.bbox_actor, self.scene.bbox_store.image_center)
                np.save(output_path, points)
                self.output_text.append(f"-> Export Bbox points to:\n {output_path}")
            self.scene.bbox_store.bbox_data.bbox_path = output_path
        else: utils.display_warning("Need to load a bounding box first!")

    def export_mask(self):
        if self.scene.mask_store.mask_actor:
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Mask Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.png')
                # Store the transformed mask actor if there is any transformation
                mask_surface = self.scene.mask_store.update_mask()
                self.scene.mask_container.load_mask(mask_surface)
                image = self.scene.mask_store.render_mask(camera=self.plotter.camera.copy())
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                # self.output_text.append(f"-> Export mask render to:\n {output_path}")
            self.scene.mask_store.mask_path = output_path
        else: utils.display_warning("Need to load a mask first!")

    def export_image(self):
        if self.scene.image_store.reference:
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Image Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.png')
                image_rendered = self.scene.image_store.render_image(camera=self.plotter.camera.copy())
                rendered_image = PIL.Image.fromarray(image_rendered)
                rendered_image.save(output_path)
                # self.output_text.append(f"-> Export image render to:\n {output_path}")
        else: utils.display_warning("Need to load an image first!")

    def export_camera_info(self):
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Camera Info Files (*.pkl)")
        if output_path:
            if pathlib.Path(output_path).suffix == '': output_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + '.pkl')
            camera_intrinsics = self.scene.image_store.camera_intrinsics.astype('float32')
            if self.scene.image_store.height:
                focal_length = (self.scene.image_store.height / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
                camera_intrinsics[0, 0] = focal_length
                camera_intrinsics[1, 1] = focal_length
            camera_info = {'camera_intrinsics': camera_intrinsics}
            with open(output_path,"wb") as f: pickle.dump(camera_info, f)
            # self.output_text.append(f"-> Export camera info to:\n {output_path}")
