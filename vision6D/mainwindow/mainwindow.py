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
import pathlib
import functools

import PIL.Image
import numpy as np

# Qt5 import
from PyQt5 import QtWidgets, QtGui
from pyvistaqt import MainWindow
from PyQt5.QtCore import Qt, QPoint, pyqtSignal

# self defined package import
from ..widgets import CustomQtInteractor
from ..widgets import PopUpDialog
from ..widgets import SearchBar
from ..widgets import PnPWindow

from ..components import ImageStore
from ..components import MaskStore
from ..components import BboxStore
from ..components import MeshStore
from ..components import VideoStore
from ..components import FolderStore

from ..containers import ImageContainer
from ..containers import MaskContainer
from ..containers import BboxContainer
from ..containers import MeshContainer
from ..containers import PnPContainer
from ..containers import VideoContainer
from ..containers import FolderContainer

from ..tools import utils

from ..path import ICON_PATH, PKG_ROOT

np.set_printoptions(suppress=True)

class SquareButton(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)

    def resizeEvent(self, event):
        self.setFixedSize(25, 25)

class CustomButtonWidget(QtWidgets.QWidget):
    colorChanged = pyqtSignal(str, str) 
    def __init__(self, button_name, parent=None):
        super(CustomButtonWidget, self).__init__(parent)
        self.setFixedHeight(50)

        # Main layout for the widget
        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Container widget for the buttons
        button_container = QtWidgets.QWidget()
        button_layout = QtWidgets.QGridLayout(button_container)
        button_layout.setContentsMargins(0, 0, 10, 0)
        button_layout.setSpacing(0)

        # Create the main button
        self.button = QtWidgets.QPushButton(button_name)
        self.button.setFixedHeight(50)
        button_layout.addWidget(self.button, 0, 0, 1, 1)

        # Create the square button
        self.square_button = SquareButton()
        self.square_button.clicked.connect(self.show_color_popup)
        button_layout.addWidget(self.square_button, 0, 0, 0, 0, Qt.AlignRight | Qt.AlignVCenter)

        # Add the button container to the main layout
        layout.addWidget(button_container)

        # Create the double spin box and add it to the layout
        self.double_spinbox = QtWidgets.QDoubleSpinBox()
        self.double_spinbox.setFixedHeight(45)
        self.double_spinbox.setMinimum(0.0)
        self.double_spinbox.setMaximum(1.0)
        self.double_spinbox.setDecimals(2)
        self.double_spinbox.setSingleStep(0.05)
        layout.addWidget(self.double_spinbox)

        # Set the stretch factors
        layout.setStretch(0, 20)  # Main Button stretch factor
        layout.setStretch(1, 1)   # Square Button stretch factor
        layout.setStretch(2, 1)   # SpinBox stretch factor

        # Set the layout for the widget
        self.setLayout(layout)

    def update_square_button_color(self, text, popup):
        self.square_button.setObjectName(text)
        if text == 'nocs' or text == 'texture':
            gradient_str = """
            background-color: qlineargradient(
                spread:pad, x1:0, y1:0, x2:1, y2:1,
                stop:0 red, stop:0.17 orange, stop:0.33 yellow,
                stop:0.5 green, stop:0.67 blue, stop:0.83 indigo, stop:1 violet);
            """
            self.square_button.setStyleSheet(gradient_str)
        else:
            self.square_button.setStyleSheet(f"background-color: {text}")
        self.colorChanged.emit(text, self.button.text()) # the order is important (color, name)
        popup.close() # automatically close the popup window

    def show_color_popup(self):
        button_name = self.button.text()
        if button_name != 'image':
            popup = PopUpDialog(self, on_button_click=lambda text: self.update_square_button_color(text, popup))
            button_position = self.square_button.mapToGlobal(QPoint(0, 0))
            popup.move(button_position + QPoint(self.square_button.width(), 0))
            popup.exec_()

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
        self.track_actors_names = []
        self.button_group_actors_names = QtWidgets.QButtonGroup(self)

        # Create the plotter
        self.create_plotter()

        self.image_store = ImageStore(self.plotter)
        self.mask_store = MaskStore()
        self.bbox_store = BboxStore()
        self.mesh_store = MeshStore()
        self.video_store = VideoStore()
        self.folder_store = FolderStore()

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

        # create containers
        self.initial_containers()
                
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

    def initial_containers(self):

        # set up the camera props
        self.image_container = ImageContainer(plotter=self.plotter, 
                                            hintLabel=self.hintLabel,
                                            track_actors_names=self.track_actors_names, 
                                            add_button_actor_name=self.add_button_actor_name,
                                            output_text=self.output_text)
        
        self.mask_container = MaskContainer(plotter=self.plotter,
                                            hintLabel=self.hintLabel,
                                            track_actors_names=self.track_actors_names, 
                                            add_button_actor_name=self.add_button_actor_name,
                                            output_text=self.output_text)
         
        self.mesh_container = MeshContainer(plotter=self.plotter,
                                            hintLabel=self.hintLabel,
                                            track_actors_names=self.track_actors_names,
                                            add_button_actor_name=self.add_button_actor_name,
                                            button_group_actors_names=self.button_group_actors_names,
                                            check_button=self.check_button,
                                            reset_camera=self.image_store.reset_camera,
                                            toggle_register=self.toggle_register,
                                            load_mask = self.mask_container.load_mask,
                                            output_text=self.output_text)
        
        self.pnp_container = PnPContainer(plotter=self.plotter,
                                        export_mesh_render=self.mesh_container.export_mesh_render,
                                        output_text=self.output_text)
        
        self.video_container = VideoContainer(plotter=self.plotter,
                                            hintLabel=self.hintLabel, 
                                            toggle_register=self.toggle_register,
                                            add_image=self.image_container.add_image,
                                            load_mask=self.mask_container.load_mask,
                                            clear_plot=self.clear_plot,
                                            output_text=self.output_text)
        
        self.folder_container = FolderContainer(plotter=self.plotter,
                                                toggle_register=self.toggle_register,
                                                add_folder=self.add_folder,
                                                load_mask=self.mask_container.load_mask,
                                                output_text=self.output_text)
        
        self.bbox_container = BboxContainer(plotter=self.plotter,
                                            hintLabel=self.hintLabel,
                                            track_actors_names=self.track_actors_names, 
                                            add_button_actor_name=self.add_button_actor_name,
                                            output_text=self.output_text)

    def key_bindings(self):
        # Camera related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("c"), self).activated.connect(self.image_store.reset_camera)
        QtWidgets.QShortcut(QtGui.QKeySequence("z"), self).activated.connect(self.image_store.zoom_out)
        QtWidgets.QShortcut(QtGui.QKeySequence("x"), self).activated.connect(self.image_store.zoom_in)

        # Mask related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("t"), self).activated.connect(self.mask_container.reset_mask)

        # Bbox related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("f"), self).activated.connect(self.bbox_container.reset_bbox)

        # Mesh related key bindings 
        QtWidgets.QShortcut(QtGui.QKeySequence("k"), self).activated.connect(self.mesh_container.reset_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("l"), self).activated.connect(self.mesh_container.update_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("s"), self).activated.connect(self.mesh_container.undo_actor_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("y"), self).activated.connect(lambda up=True: self.mesh_container.toggle_surface_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("u"), self).activated.connect(lambda up=False: self.mesh_container.toggle_surface_opacity(up))

        # Video related key bindings 
        QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self).activated.connect(self.video_container.next_info)
        QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self).activated.connect(self.video_container.prev_info)
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self).activated.connect(self.video_container.play_video)

        # Folder related key bindings 
        QtWidgets.QShortcut(QtGui.QKeySequence("a"), self).activated.connect(self.folder_container.prev_info)
        QtWidgets.QShortcut(QtGui.QKeySequence("d"), self).activated.connect(self.folder_container.next_info)

        # todo: create the swith button for mesh and ct "ctrl + tap"
        QtWidgets.QShortcut(QtGui.QKeySequence("Tab"), self).activated.connect(self.tap_toggle_opacity)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Tab"), self).activated.connect(self.ctrl_tap_opacity)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+w"), self).activated.connect(self.clear_plot)

    def showMaximized(self):
        super(MyMainWindow, self).showMaximized()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
            self.hintLabel.hide()  # Hide hint when dragging
        else:
            e.ignore()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isdir(file_path): self.add_folder(folder_path=file_path)
            else:
                # Load workspace json file
                if file_path.endswith(('.json')): self.add_workspace(workspace_path=file_path)
                # Load mesh file
                elif file_path.endswith(('.mesh', '.ply', '.stl', '.obj', '.off', '.dae', '.fbx', '.3ds', '.x3d')):
                    self.mesh_container.add_mesh_file(mesh_path=file_path)
                # Load video file
                elif file_path.endswith(('.avi', '.mp4', '.mkv', '.mov', '.fly', '.wmv', '.mpeg', '.asf', '.webm')):
                    self.video_container.add_video_file(video_path=file_path)
                # Load image/mask file
                elif file_path.endswith(('.png', '.jpg', 'jpeg', '.tiff', '.bmp', '.webp', '.ico')):  # add image/mask
                    file_data = np.array(PIL.Image.open(file_path).convert('L'), dtype='uint8')
                    unique, _ = np.unique(file_data, return_counts=True)
                    if len(unique) == 2: self.mask_container.add_mask_file(mask_path=file_path)
                    else: self.image_container.add_image_file(image_path=file_path) 
                        
                elif file_path.endswith('.npy'): self.mesh_container.add_pose_file(pose_path=file_path)
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
        fileMenu.addAction('Add Folder', functools.partial(self.add_folder, prompt=True))
        fileMenu.addAction('Add Video', functools.partial(self.video_container.add_video_file, prompt=True))
        fileMenu.addAction('Add Image', functools.partial(self.image_container.add_image_file, prompt=True))
        fileMenu.addAction('Add Mask', self.mask_container.set_mask)
        fileMenu.addAction('Add Bbox', functools.partial(self.bbox_container.add_bbox_file, prompt=True))
        fileMenu.addAction('Add Mesh', functools.partial(self.mesh_container.add_mesh_file, prompt=True))
        fileMenu.addAction('Clear', self.clear_plot)

        # allow to export files
        exportMenu = mainMenu.addMenu('Export')
        exportMenu.addAction('Workspace', self.export_workspace)
        exportMenu.addAction('Image', self.image_container.export_image)
        exportMenu.addAction('Mask', self.mask_container.export_mask)
        exportMenu.addAction('Bbox', self.bbox_container.export_bbox)
        exportMenu.addAction('Mesh/Pose', self.mesh_container.export_mesh_pose)
        exportMenu.addAction('Mesh Render', self.mesh_container.export_mesh_render)
        exportMenu.addAction('SegMesh Render', self.mesh_container.export_segmesh_render)
        exportMenu.addAction('Camera Info', self.image_container.export_camera_info)
        
        # Add video related actions
        VideoMenu = mainMenu.addMenu('Video')
        VideoMenu.addAction('Play', self.video_container.play_video)
        VideoMenu.addAction('Sample', self.video_container.sample_video)
        VideoMenu.addAction('Save', self.video_container.save_info)
        VideoMenu.addAction('Prev', self.video_container.prev_info)
        VideoMenu.addAction('Next', self.video_container.next_info)

        # Add folder related actions
        FolderMenu = mainMenu.addMenu('Folder')
        FolderMenu.addAction('Save', self.folder_container.save_info)
        FolderMenu.addAction('Prev', self.folder_container.prev_info)
        FolderMenu.addAction('Next', self.folder_container.next_info)

        # Add pnp algorithm related actions
        PnPMenu = mainMenu.addMenu('PnP')
        PnPMenu.addAction('EPnP with mesh', self.pnp_container.epnp_mesh)
        PnPMenu.addAction('EPnP with nocs mask', functools.partial(self.pnp_container.epnp_mask, True))
        PnPMenu.addAction('EPnP with latlon mask', functools.partial(self.pnp_container.epnp_mask, False))

    # create draw menu when right click on the image
    def draw_menu(self, event):
        context_menu = QtWidgets.QMenu(self)

        set_distance = QtWidgets.QAction('Set Distance', self)
        set_distance.triggered.connect(self.set_object_distance)

        set_mask = QtWidgets.QAction('Set Mask', self)
        set_mask.triggered.connect(self.mask_container.set_mask)

        draw_mask_menu = QtWidgets.QMenu('Draw Mask', self)  # Create a submenu for 'Draw Mask'
        live_wire = QtWidgets.QAction('Live Wire', self)
        live_wire.triggered.connect(self.mask_container.draw_mask)  # Connect to a slot
        # sam = QtWidgets.QAction('SAM', self)
        # sam.triggered.connect(functools.partial(self.mask_container.draw_mask, sam=True))  # Connect to another slot
        
        draw_bbox = QtWidgets.QAction('Draw BBox', self)
        draw_bbox.triggered.connect(self.bbox_container.draw_bbox)
        draw_mask_menu.addAction(live_wire)
        # draw_mask_menu.addAction(sam)

        reset_mask = QtWidgets.QAction('Reset Mask (t)', self)
        reset_mask.triggered.connect(self.mask_container.reset_mask)
        reset_bbox = QtWidgets.QAction('Reset Bbox (f)', self)
        reset_bbox.triggered.connect(self.bbox_container.reset_bbox)
        
        context_menu.addAction(set_distance)
        context_menu.addAction(set_mask)
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

        self.panel_display()
        self.panel_output()
        
        # Set the stretch factor for each section to be equal
        self.panel_layout.setStretchFactor(self.display, 1)
        self.panel_layout.setStretchFactor(self.output, 1)

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
    
    def on_camera_options_selection_change(self, option):
        if option == "Set Camera":
            self.image_container.set_camera()
        elif option == "Reset Camera (c)":
            self.image_store.reset_camera()
        elif option == "Zoom In (x)":
            self.image_store.zoom_in()
        elif option == "Zoom Out (z)":
            self.image_store.zoom_out()
        elif option == "Calibrate":
            self.image_container.camera_calibrate()
    
    def on_pose_options_selection_change(self, option):
        if option == "Set Pose":
            self.mesh_container.set_pose()
        elif option == "PnP Register":
            self.pnp_register()
        elif option == "Reset GT Pose (k)":
            self.mesh_container.reset_gt_pose()
        elif option == "Update GT Pose (l)":
            self.mesh_container.update_gt_pose()
        elif option == "Undo Pose (s)":
            self.mesh_container.undo_actor_pose()

    def handle_transformation_matrix(self, transformation_matrix):
        self.toggle_register(transformation_matrix)
        self.mesh_container.update_gt_pose()

    def pnp_register(self):
        if not self.image_store.image_actor: utils.display_warning("Need to load an image first!"); return
        if self.mesh_store.reference is None: utils.display_warning("Need to select a mesh first!"); return
        image = utils.get_image_actor_scalars(self.image_store.image_actor)
        self.pnp_window = PnPWindow(image_source=image, 
                                    mesh_data=self.mesh_store.meshes[self.mesh_store.reference],
                                    camera_intrinsics=self.image_store.camera_intrinsics.astype(np.float32))
        self.pnp_window.transformation_matrix_computed.connect(self.handle_transformation_matrix)
        
    #^ Panel Display
    def panel_display(self):
        self.display = QtWidgets.QGroupBox("Console")
        display_layout = QtWidgets.QVBoxLayout()
        display_layout.setContentsMargins(10, 20, 10, 5)

        #* Create the top widgets (layout)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Create Grid layout for function buttons
        top_grid_layout = QtWidgets.QGridLayout()

        row, column = 0, 0

        # Create a QPushButton that will act as a drop-down button and QMenu to act as the drop-down menu
        self.camera_options_button = QtWidgets.QPushButton("Set Camera")
        self.camera_options_menu = QtWidgets.QMenu()
        self.camera_options_menu.addAction("Set Camera", lambda: self.on_camera_options_selection_change("Set Camera"))
        self.camera_options_menu.addAction("Reset Camera (c)", lambda: self.on_camera_options_selection_change("Reset Camera (c)"))
        self.camera_options_menu.addAction("Zoom In (x)", lambda: self.on_camera_options_selection_change("Zoom In (x)"))
        self.camera_options_menu.addAction("Zoom Out (z)", lambda: self.on_camera_options_selection_change("Zoom Out (z)"))
        self.camera_options_menu.addAction("Calibrate", lambda: self.on_camera_options_selection_change("Calibrate"))
        self.camera_options_button.setMenu(self.camera_options_menu)
        top_grid_layout.addWidget(self.camera_options_button, row, column)

        self.pose_options_button = QtWidgets.QPushButton("Set Pose")
        self.pose_options_menu = QtWidgets.QMenu()
        self.pose_options_menu.addAction("Set Pose", lambda: self.on_pose_options_selection_change("Set Pose"))
        self.pose_options_menu.addAction("PnP Register", lambda: self.on_pose_options_selection_change("PnP Register"))
        self.pose_options_menu.addAction("Reset GT Pose (k)", lambda: self.on_pose_options_selection_change("Reset GT Pose (k)"))
        self.pose_options_menu.addAction("Update GT Pose (l)", lambda: self.on_pose_options_selection_change("Update GT Pose (l)"))
        self.pose_options_menu.addAction("Undo Pose (s)", lambda: self.on_pose_options_selection_change("Undo Pose (s)"))
        self.pose_options_button.setMenu(self.pose_options_menu)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(self.pose_options_button, row, column)

        # Draw buttons
        self.draw_options_button = QtWidgets.QPushButton("Draw")
        self.draw_options_menu = QtWidgets.QMenu()
        self.draw_options_menu.addAction("Set Mask", self.mask_container.set_mask)
        self.draw_options_menu.addAction("Draw Mask", self.mask_container.draw_mask)
        self.draw_options_menu.addAction("Draw Bbox", self.bbox_container.draw_bbox)
        self.draw_options_menu.addAction("Reset Mask (t)", self.mask_container.reset_mask)
        self.draw_options_menu.addAction("Reset Bbox (f)", self.bbox_container.reset_bbox)
        self.draw_options_button.setMenu(self.draw_options_menu)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(self.draw_options_button, row, column)

        # Other buttons
        self.other_options_button = QtWidgets.QPushButton("Other")
        self.other_options_menu = QtWidgets.QMenu()
        self.other_options_menu.addAction("Flip Left/Right", lambda direction="x": self.mirror_actors(direction))
        self.other_options_menu.addAction("Flip Up/Down", lambda direction="y": self.mirror_actors(direction))
        self.other_options_menu.addAction("Set Mesh Spacing", self.mesh_container.set_spacing)
        self.other_options_menu.addAction("Set Image Distance", self.set_object_distance)
        self.other_options_menu.addAction("Toggle Meshes", self.mesh_container.toggle_hide_meshes_button)
        self.other_options_menu.addAction("Remove Actor", self.remove_actors_button)
        self.other_options_button.setMenu(self.other_options_menu)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(self.other_options_button, row, column)
        
        top_grid_widget = QtWidgets.QWidget()
        top_grid_widget.setLayout(top_grid_layout)
        top_layout.addWidget(top_grid_widget)
        display_layout.addLayout(top_layout)

        #* Create the bottom widgets
        actor_widget = QtWidgets.QLabel("Actors")
        display_layout.addWidget(actor_widget)

        # Create a scroll area for the buttons
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        display_layout.addWidget(scroll_area)

        # Create a container widget for the custom widgets
        custom_widget_container = QtWidgets.QWidget()
        self.custom_widget_layout = QtWidgets.QVBoxLayout()
        self.custom_widget_layout.setSpacing(0)  # Remove spacing between custom widgets
        custom_widget_container.setLayout(self.custom_widget_layout)

        self.custom_widget_layout.addStretch()

        # Set the container widget as the scroll area's widget
        scroll_area.setWidget(custom_widget_container)

        self.display.setLayout(display_layout)
        self.panel_layout.addWidget(self.display)

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
        
    def toggle_register(self, pose):
        self.mesh_store.meshes[self.mesh_store.reference].actor.user_matrix = pose
        self.mesh_store.meshes[self.mesh_store.reference].undo_poses.append(pose)
        self.mesh_store.meshes[self.mesh_store.reference].undo_poses = self.mesh_store.meshes[self.mesh_store.reference].undo_poses[-20:]
            
    def check_button(self, name, output_text=True):  
        button = next((btn for btn in self.button_group_actors_names.buttons() if btn.text() == name), None)
        if button:
            button.setChecked(True)
            self.button_actor_name_clicked(name=name, output_text=output_text)
                
    def button_actor_name_clicked(self, name, output_text=True):
        if name in self.mesh_store.meshes:
            self.mesh_store.reference = name
            mesh_data = self.mesh_store.meshes[name]
            text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
            mesh_data.actor.user_matrix[0, 0], mesh_data.actor.user_matrix[0, 1], mesh_data.actor.user_matrix[0, 2], mesh_data.actor.user_matrix[0, 3], 
            mesh_data.actor.user_matrix[1, 0], mesh_data.actor.user_matrix[1, 1], mesh_data.actor.user_matrix[1, 2], mesh_data.actor.user_matrix[1, 3], 
            mesh_data.actor.user_matrix[2, 0], mesh_data.actor.user_matrix[2, 1], mesh_data.actor.user_matrix[2, 2], mesh_data.actor.user_matrix[2, 3],
            mesh_data.actor.user_matrix[3, 0], mesh_data.actor.user_matrix[3, 1], mesh_data.actor.user_matrix[3, 2], mesh_data.actor.user_matrix[3, 3])
            if output_text: self.output_text.append(f"--> Mesh {name} pose is:"); self.output_text.append(text)
            self.mesh_store.meshes[self.mesh_store.reference].undo_poses.append(mesh_data.actor.user_matrix)
            self.mesh_store.meshes[self.mesh_store.reference].undo_poses = self.mesh_store.meshes[self.mesh_store.reference].undo_poses[-20:]
        # else:
        #     self.mesh_store.reference = None #* For fixing some bugs in segmesh render function

    def add_button_actor_name(self, name):
        button_widget = CustomButtonWidget(name)
        # create the color button for each instance, and connect the button to the colorChanged signal
        button_widget.colorChanged.connect(lambda color, name=name: self.color_value_change(color, name))
        button = button_widget.button

        if name == 'image': 
            self.image_store.opacity_spinbox = button_widget.double_spinbox
            self.image_store.opacity_spinbox.setValue(self.image_store.image_opacity)
            self.image_store.opacity_spinbox.valueChanged.connect(lambda value, name=name: self.opacity_value_change(value, name))
        elif name == 'mask': 
            self.mask_store.opacity_spinbox = button_widget.double_spinbox
            self.mask_store.opacity_spinbox.setValue(self.mask_store.mask_opacity)
            self.mask_store.opacity_spinbox.valueChanged.connect(lambda value, name=name: self.opacity_value_change(value, name))
            self.mask_store.color_button = button_widget.square_button
            self.mask_store.color_button.setStyleSheet(f"background-color: {self.mask_store.color}")
        elif name == 'bbox':
            self.bbox_store.opacity_spinbox = button_widget.double_spinbox
            self.bbox_store.opacity_spinbox.setValue(self.bbox_store.bbox_opacity)
            self.bbox_store.opacity_spinbox.valueChanged.connect(lambda value, name=name: self.opacity_value_change(value, name))
            self.bbox_store.color_button = button_widget.square_button
            self.bbox_store.color_button.setStyleSheet(f"background-color: {self.bbox_store.color}")
        elif name in self.mesh_store.meshes: 
            self.mesh_store.meshes[name].opacity_spinbox = button_widget.double_spinbox
            self.mesh_store.meshes[name].opacity_spinbox.setValue(self.mesh_store.meshes[name].opacity)
            self.mesh_store.meshes[name].opacity_spinbox.valueChanged.connect(lambda value, name=name: self.opacity_value_change(value, name))
            self.mesh_store.meshes[name].color_button = button_widget.square_button
            self.mesh_store.meshes[name].color_button.setStyleSheet(f"background-color: {self.mesh_store.meshes[name].color}")

        button.setCheckable(True) # Set the button to be checkable
        button.clicked.connect(lambda _, name=name: self.button_actor_name_clicked(name))
        button.setChecked(True)
        self.custom_widget_layout.insertWidget(0, button_widget)
        self.button_group_actors_names.addButton(button)
        self.button_actor_name_clicked(name=name)
    
    def opacity_value_change(self, value, name):
        if name == 'image': self.image_container.set_image_opacity(value)
        elif name == 'mask': self.mask_container.set_mask_opacity(value)
        elif name == 'bbox': self.bbox_container.set_bbox_opacity(value)
        elif name in self.mesh_store.meshes: self.mesh_container.set_mesh_opacity(name, value)

    def color_value_change(self, color, name):
        if name == 'mask': 
            try:
                self.mask_container.set_mask_color(color)
                self.mask_store.color = color
            except ValueError: 
                utils.display_warning(f"Cannot set color ({color}) to mask")
                self.mask_store.color_button.setStyleSheet(f"background-color: {self.mask_store.color}")
        elif name == 'bbox':
            try:
                self.bbox_container.set_bbox_color(color)
                self.bbox_store.color = color
            except ValueError: 
                utils.display_warning(f"Cannot set color ({color}) to bbox")
                self.bbox_store.color_button.setStyleSheet(f"background-color: {self.bbox_store.color}")
        elif name in self.mesh_store.meshes:
            try:
                color = self.mesh_container.set_color(color, name)
                self.mesh_store.meshes[name].color = color
                if color != "nocs" and color != "texture": 
                    self.mesh_store.meshes[name].color_button.setStyleSheet(f"background-color: {self.mesh_store.meshes[name].color}")
            except ValueError:
                utils.display_warning(f"Cannot set color ({color}) to {name}")

    def tap_toggle_opacity(self):
        if self.mesh_store.meshes[self.mesh_store.reference].opacity == 1.0: 
            self.mesh_store.meshes[self.mesh_store.reference].opacity = 0.0
            self.image_store.image_opacity = 1.0
        elif self.mesh_store.meshes[self.mesh_store.reference].opacity == 0.9:
            self.mesh_store.meshes[self.mesh_store.reference].opacity = 1.0
            self.image_store.image_opacity = 0.0
        else:
            self.mesh_store.meshes[self.mesh_store.reference].opacity = 0.9
            self.image_store.image_opacity = 0.9
        self.image_store.image_actor.GetProperty().opacity = self.image_store.image_opacity
        self.image_store.opacity_spinbox.setValue(self.image_store.image_opacity)
        self.mesh_store.meshes[self.mesh_store.reference].opacity_spinbox.setValue(self.mesh_store.meshes[self.mesh_store.reference].opacity)

    def ctrl_tap_opacity(self):
        if self.mesh_store.reference is not None:
            for mesh_data in self.mesh_store.meshes.values():
                if mesh_data.opacity != 0: mesh_data.opacity_spinbox.setValue(0)
                else: mesh_data.opacity_spinbox.setValue(mesh_data.previous_opacity)
        else:
            if self.image_store.image_actor is not None:
                if self.image_store.image_opacity != 0: self.image_store.opacity_spinbox.setValue(0)
                else: self.image_store.opacity_spinbox.setValue(self.image_store.previous_opacity)
            if self.mask_store.mask_actor is not None:
                if self.mask_store.mask_opacity != 0: self.mask_store.opacity_spinbox.setValue(0)
                else: self.mask_store.opacity_spinbox.setValue(self.mask_store.previous_opacity)
            if self.bbox_store.bbox_actor is not None:
                if self.bbox_store.bbox_opacity != 0: self.bbox_store.opacity_spinbox.setValue(0)
                else: self.bbox_store.opacity_spinbox.setValue(self.bbox_store.previous_opacity)

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
            if 'image_path' in workspace and workspace['image_path'] is not None: self.image_container.add_image_file(image_path=root / pathlib.Path(*workspace['image_path'].split("\\")))
            if 'video_path' in workspace and workspace['video_path'] is not None: self.video_container.add_video_file(video_path=root / pathlib.Path(*workspace['video_path'].split("\\")))
            if 'mask_path' in workspace and workspace['mask_path'] is not None: self.mask_container.add_mask_file(mask_path=root / pathlib.Path(*workspace['mask_path'].split("\\")))
            if 'bbox_path' in workspace and workspace['bbox_path'] is not None: self.bbox_container.add_bbox_file(bbox_path=root / pathlib.Path(*workspace['bbox_path'].split("\\")))
            if 'mesh_path' in workspace:
                meshes = workspace['mesh_path']
                for item in meshes: 
                    mesh_path, pose = meshes[item]
                    self.mesh_container.add_mesh_file(mesh_path=root / pathlib.Path(*mesh_path.split("\\")))
                    self.mesh_container.add_pose_file(pose)
            self.image_store.reset_camera()

    def export_workspace(self):
        workspace_dict = {"mesh_path": {}}
        workspace_dict["image_path"] = self.image_store.image_path
        workspace_dict["mask_path"] = self.mask_store.mask_path
        workspace_dict["bbox_path"] = self.bbox_store.bbox_path
        for mesh_data in self.mesh_store.meshes.values(): 
            workspace_dict["mesh_path"][mesh_data.name] = (mesh_data.mesh_path, mesh_data.actor.user_matrix.tolist())
        # write the dict to json file
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Mesh Files (*.json)")
        if output_path != "":
            with open(output_path, 'w') as f: json.dump(workspace_dict, f, indent=4)

    def add_folder(self, folder_path='', prompt=False):
        if prompt: 
            folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            if self.video_store.video_path or self.workspace_path: self.clear_plot() # main goal is to set video_path to None
            image_path, mask_path, pose_path, mesh_path = self.folder_store.add_folder(folder_path=folder_path, meshes=self.mesh_store.meshes)
            if image_path or mask_path or pose_path or mesh_path:
                if image_path: self.image_container.add_image_file(image_path=image_path)
                if mask_path: self.mask_container.add_mask_file(mask_path=mask_path)
                if mesh_path: 
                    with open(mesh_path, 'r') as f: mesh_path = f.read().splitlines()
                    for path in mesh_path: self.mesh_container.add_mesh_file(path)
                if pose_path: self.mesh_container.add_pose_file(pose_path=pose_path)
                self.image_store.reset_camera()
            else:
                self.folder_store.reset()
                utils.display_warning("Not a valid folder, please reload a folder")

    def mirror_actors(self, direction):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button:
            name = checked_button.text()
            if name == 'image': self.image_container.mirror_image(direction)
            elif name == 'mask': self.mask_container.mirror_mask(direction)
            elif name == 'bbox': self.bbox_container.mirror_bbox(direction)
            elif name in self.mesh_store.meshes: self.mesh_container.mirror_mesh(name, direction)
        else: utils.display_warning("Need to select an actor first")

    def remove_actor(self, button):
        name = button.text()
        if name == 'image': 
            actor = self.image_store.image_actor
            self.image_store.reset()
        elif name == 'mask':
            actor = self.mask_store.mask_actor
            self.mask_store.reset()
        elif name == 'bbox':
            actor = self.bbox_store.bbox_actor
            self.bbox_store.reset()
        elif name in self.mesh_store.meshes: 
            actor = self.mesh_store.meshes[name].actor
            self.mesh_store.remove_mesh(name)

        self.plotter.remove_actor(actor)
        self.track_actors_names.remove(name)
        # remove the button from the button group
        self.button_group_actors_names.removeButton(button)
        # remove the button from the custom button widget
        self.remove_custom_button_widget(button)
        # offically delete the button
        button.deleteLater()

        # clear out the plot if there is no actor
        if (self.image_store.image_actor is None) and (self.mask_store.mask_actor is None) and (len(self.mesh_store.meshes) == 0): 
            self.clear_plot()

    def remove_actors_button(self):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button: self.remove_actor(checked_button)
        else: utils.display_warning("Need to select an actor first")

    def clear_plot(self):
        # Clear out everything in the remove menu
        for button in self.button_group_actors_names.buttons():
            name = button.text()
            if name == 'image': 
                actor = self.image_store.image_actor
                self.image_store.reset()
                self.image_store.mirror_x = False
                self.image_store.mirror_y = False
            elif name == 'mask': 
                actor = self.mask_store.mask_actor
                self.mask_store.reset()
                self.mask_store.mirror_x = False
                self.mask_store.mirror_y = False
            elif name == 'bbox':
                actor = self.bbox_store.bbox_actor
                self.bbox_store.reset()
                self.bbox_store.mirror_x = False
                self.bbox_store.mirror_y = False
            elif name in self.mesh_store.meshes: 
                actor = self.mesh_store.meshes[name].actor
                self.mesh_store.remove_mesh(name)

            self.plotter.remove_actor(actor)
            # remove the button from the button group
            self.button_group_actors_names.removeButton(button)
            self.remove_custom_button_widget(button)
            # offically delete the button
            button.deleteLater()

        self.mesh_store.reset()
        self.video_store.reset()
        self.folder_store.reset()
        self.workspace_path = ''
        self.track_actors_names.clear()
        self.reset_output_text()
        self.hintLabel.show()

    def remove_custom_button_widget(self, button):
        for i in range(self.custom_widget_layout.count()): 
            widget = self.custom_widget_layout.itemAt(i).widget()
            if widget is not None and hasattr(widget, 'button') and widget.button == button:
                self.custom_widget_layout.removeWidget(widget)
                widget.deleteLater()
                if i <= self.custom_widget_layout.count() and self.custom_widget_layout.count() > 1:
                    if i == self.custom_widget_layout.count()-1:
                        next_widget = self.custom_widget_layout.itemAt(i-1).widget()
                    else:
                        next_widget = self.custom_widget_layout.itemAt(i).widget()
                    if next_widget is not None and hasattr(next_widget, 'button'):
                        next_widget.button.setChecked(True)
                break

    def set_object_distance(self):
        distance, ok = QtWidgets.QInputDialog().getText(QtWidgets.QMainWindow(), 'Input', "Set Objects distance to camera", text=str(self.image_store.object_distance))
        if ok:
            distance = float(distance)
            if self.image_store.image_actor is not None:
                self.image_store.image_pv.translate(-np.array([0, 0, self.image_store.image_pv.center[-1]]), inplace=True) # very important, re-center it to [0, 0, 0]
                self.image_store.image_pv.translate(np.array([0, 0, distance]), inplace=True)
            if self.mask_store.mask_actor is not None:
                self.mask_store.mask_pv.translate(-np.array([0, 0, self.mask_store.mask_pv.center[-1]]), inplace=True) # very important, re-center it to [0, 0, 0]
                self.mask_store.mask_pv.translate(np.array([0, 0, distance]), inplace=True)
            if self.bbox_store.bbox_actor is not None:
                self.bbox_store.bbox_pv.translate(-np.array([0, 0, self.bbox_store.bbox_pv.center[-1]]), inplace=True) # very important, re-center it to [0, 0, 0]
                self.bbox_store.bbox_pv.translate(np.array([0, 0, distance]), inplace=True)
            #! do not modify the object_distance for meshes, because it will mess up the pose
            self.image_store.object_distance = distance
            self.image_store.reset_camera()