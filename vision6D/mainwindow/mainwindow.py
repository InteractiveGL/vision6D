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
from PyQt5.QtCore import Qt, QPoint

# self defined package import
from ..widgets import CustomQtInteractor
from ..widgets import PopUpDialog
from ..widgets import SearchBar

from ..components import CameraStore
from ..components import ImageStore
from ..components import MaskStore
from ..components import BboxStore
from ..components import MeshStore
from ..components import PointStore
from ..components import VideoStore
from ..components import FolderStore

from ..containers import CameraContainer
from ..containers import ImageContainer
from ..containers import MaskContainer
from ..containers import BboxContainer
from ..containers import MeshContainer
from ..containers import PointContainer
from ..containers import PnPContainer
from ..containers import VideoContainer
from ..containers import FolderContainer

from ..tools import utils

from ..path import ICON_PATH
from ..path import PKG_ROOT

np.set_printoptions(suppress=True)

class MyMainWindow(MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        
        # Set up the main window layout
        self.setWindowTitle("Vision6D")
        self.window_size = (1920, 1080)
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        self.setAcceptDrops(True)

        # Initialize
        self.workspace_path = ''
        self.track_actors_names = []
        self.button_group_actors_names = QtWidgets.QButtonGroup(self)

        self.camera_store = CameraStore(self.window_size) # center camera at the origin in world coordinate to match with pytorch3d
        self.object_distance = 100.0 # set the object distance to the camera in world coordinate
        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.bbox_store = BboxStore()
        self.mesh_store = MeshStore(self.window_size)
        self.point_store = PointStore()
        self.video_store = VideoStore()
        self.folder_store = FolderStore()

        self.anchor_button = QtWidgets.QPushButton("Anchor")
        self.color_button = QtWidgets.QPushButton("Color")
        self.opacity_spinbox = QtWidgets.QDoubleSpinBox()
        self.play_video_button = QtWidgets.QPushButton("Play Video")
        self.output_text = QtWidgets.QTextEdit()

        # Create the plotter
        self.create_plotter()

        # Add a QLabel as an overlay hint label
        self.hintLabel = QtWidgets.QLabel(self.plotter)
        self.hintLabel.setText("Drag and drop a file here...")
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
        self.camera_container = CameraContainer(plotter=self.plotter)
        self.camera_container.set_camera_props()
        
        self.image_container = ImageContainer(plotter=self.plotter, 
                                            hintLabel=self.hintLabel, 
                                            object_distance=self.object_distance,
                                            track_actors_names=self.track_actors_names, 
                                            add_button_actor_name=self.add_button_actor_name, 
                                            check_button=self.check_button,
                                            output_text=self.output_text)
        
        self.mask_container = MaskContainer(plotter=self.plotter,
                                            hintLabel=self.hintLabel, 
                                            object_distance=self.object_distance,
                                            track_actors_names=self.track_actors_names, 
                                            add_button_actor_name=self.add_button_actor_name, 
                                            check_button=self.check_button, 
                                            output_text=self.output_text)
        
        self.point_container = PointContainer(plotter=self.plotter,
                                            hintLabel=self.hintLabel,
                                            track_actors_names=self.track_actors_names, 
                                            add_button_actor_name=self.add_button_actor_name,
                                            output_text=self.output_text,
                                            )
        
        self.mesh_container = MeshContainer(color_button=self.color_button,
                                            plotter=self.plotter,
                                            hintLabel=self.hintLabel,
                                            track_actors_names=self.track_actors_names,
                                            add_button_actor_name=self.add_button_actor_name,
                                            button_group_actors_names=self.button_group_actors_names,
                                            check_button=self.check_button,
                                            opacity_spinbox=self.opacity_spinbox,
                                            opacity_value_change=self.opacity_value_change,
                                            reset_camera=self.camera_container.reset_camera,
                                            toggle_register=self.toggle_register,
                                            load_mask = self.mask_container.load_mask,
                                            output_text=self.output_text)
        
        self.pnp_container = PnPContainer(plotter=self.plotter,
                                        export_mesh_render=self.mesh_container.export_mesh_render,
                                        output_text=self.output_text)
        
        self.video_container = VideoContainer(plotter=self.plotter,
                                            anchor_button=self.anchor_button,
                                            play_video_button=self.play_video_button, 
                                            hintLabel=self.hintLabel, 
                                            toggle_register=self.toggle_register,
                                            add_image=self.image_container.add_image,
                                            load_mask=self.mask_container.load_mask,
                                            clear_plot=self.clear_plot,
                                            output_text=self.output_text)
        
        self.folder_container = FolderContainer(plotter=self.plotter,
                                                play_video_button=self.play_video_button, 
                                                toggle_register=self.toggle_register,
                                                add_folder=self.add_folder,
                                                load_mask=self.mask_container.load_mask,
                                                output_text=self.output_text)
        
        self.bbox_container = BboxContainer(plotter=self.plotter,
                                            hintLabel=self.hintLabel, 
                                            object_distance=self.object_distance,
                                            track_actors_names=self.track_actors_names, 
                                            add_button_actor_name=self.add_button_actor_name, 
                                            check_button=self.check_button, 
                                            output_text=self.output_text)

    def key_bindings(self):
        # Camera related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("c"), self).activated.connect(self.camera_container.reset_camera)
        QtWidgets.QShortcut(QtGui.QKeySequence("z"), self).activated.connect(self.camera_container.zoom_out)
        QtWidgets.QShortcut(QtGui.QKeySequence("x"), self).activated.connect(self.camera_container.zoom_in)
        
        # Image related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("b"), self).activated.connect(lambda up=True: self.image_container.toggle_image_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("n"), self).activated.connect(lambda up=False: self.image_container.toggle_image_opacity(up))

        # Mask related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("t"), self).activated.connect(self.mask_container.reset_mask)
        QtWidgets.QShortcut(QtGui.QKeySequence("g"), self).activated.connect(lambda up=True: self.mask_container.toggle_mask_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("h"), self).activated.connect(lambda up=False: self.mask_container.toggle_mask_opacity(up))

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
        fileMenu.addAction('Add Bbox', functools.partial(self.bbox_container.add_bbox_file, prompt=True))
        fileMenu.addAction('Add Mesh', functools.partial(self.mesh_container.add_mesh_file, prompt=True))
        fileMenu.addAction('Add Points', functools.partial(self.point_container.load_points_file, prompt=True))
        fileMenu.addAction('Clear', self.clear_plot)

        # allow to export files
        exportMenu = mainMenu.addMenu('Export')
        exportMenu.addAction('Image', self.image_container.export_image)
        exportMenu.addAction('Mask', self.mask_container.export_mask)
        exportMenu.addAction('Bbox', self.bbox_container.export_bbox)
        exportMenu.addAction('Mesh/Pose', self.mesh_container.export_mesh_pose)
        exportMenu.addAction('Mesh Render', self.mesh_container.export_mesh_render)
        exportMenu.addAction('SegMesh Render', self.mesh_container.export_segmesh_render)
        exportMenu.addAction('Camera Info', self.camera_container.export_camera_info)

        # Add draw related actions
        DrawMenu = mainMenu.addMenu('Draw')
        DrawMenu.addAction('Mask', self.mask_container.draw_mask)
        DrawMenu.addAction('BBox', self.bbox_container.draw_bbox)
        # DrawMenu.addAction('Points', self.point_container.draw_point)
        DrawMenu.addAction('Reset Mask (t)', self.mask_container.reset_mask)
        DrawMenu.addAction('Reset Bbox (f)', self.bbox_container.reset_bbox)
        
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

        # Add camera related actions
        CameraMenu = mainMenu.addMenu('Camera')
        CameraMenu.addAction('Set Camera', self.camera_container.set_camera)
        CameraMenu.addAction('Reset Camera (d)', self.camera_container.reset_camera)
        CameraMenu.addAction('Zoom In (x)', self.camera_container.zoom_in)
        CameraMenu.addAction('Zoom Out (z)', self.camera_container.zoom_out)
        CameraMenu.addAction('Calibrate', self.camera_container.camera_calibrate)
        
        # Add pose related actions
        PoseMenu = mainMenu.addMenu('Pose')
        PoseMenu.addAction('Set Pose', self.mesh_container.set_pose)
        PoseMenu.addAction('Reset GT Pose (k)', self.mesh_container.reset_gt_pose)
        PoseMenu.addAction('Update GT Pose (l)', self.mesh_container.update_gt_pose)
        PoseMenu.addAction('Undo Pose (s)', self.mesh_container.undo_actor_pose)

        # Add pnp algorithm related actions
        PnPMenu = mainMenu.addMenu('PnP')
        PnPMenu.addAction('EPnP with mesh', self.pnp_container.epnp_mesh)
        PnPMenu.addAction('EPnP with nocs mask', functools.partial(self.pnp_container.epnp_mask, True))
        PnPMenu.addAction('EPnP with latlon mask', functools.partial(self.pnp_container.epnp_mask, False))

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
        
    #^ Panel Display
    def panel_display(self):
        self.display = QtWidgets.QGroupBox("Console")
        display_layout = QtWidgets.QVBoxLayout()
        display_layout.setContentsMargins(10, 15, 10, 5)

        #* Create the top widgets (layout)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Create Grid layout for function buttons
        top_grid_layout = QtWidgets.QGridLayout()

        row, column = 0, 0
        
        # Anchor butto
        self.anchor_button.setCheckable(True) # Set the button to be checkable so it is highlighted, very important
        self.anchor_button.setChecked(True)
        self.anchor_button.clicked.connect(self.mesh_container.anchor_mesh)
        top_grid_layout.addWidget(self.anchor_button, row, column)
        
        # Color button
        self.color_button.clicked.connect(self.show_color_popup)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(self.color_button, row, column)
        
        # Opacity box
        self.opacity_spinbox.setMinimum(0.0)
        self.opacity_spinbox.setMaximum(1.0)
        self.opacity_spinbox.setDecimals(2)
        self.opacity_spinbox.setSingleStep(0.05)
        self.opacity_spinbox.setValue(0.3)
        self.opacity_spinbox.valueChanged.connect(self.opacity_value_change)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(self.opacity_spinbox, row, column)

        # Set the mask
        set_mask_button = QtWidgets.QPushButton("Set Mask")
        set_mask_button.clicked.connect(self.mask_container.set_mask)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(set_mask_button, row, column)
        
        # Create the actor pose button
        actor_pose_button = QtWidgets.QPushButton("Set Pose")
        actor_pose_button.clicked.connect(self.mesh_container.set_pose)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(actor_pose_button, row, column)

        set_distance_button = QtWidgets.QPushButton("Set Distance")
        set_distance_button.clicked.connect(self.set_object_distance)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(set_distance_button, row, column)

        # Create the spacing button
        self.spacing_button = QtWidgets.QPushButton("Spacing")
        self.spacing_button.clicked.connect(self.mesh_container.set_spacing)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(self.spacing_button, row, column)

        # Create the hide button
        hide_button = QtWidgets.QPushButton("Toggle Meshes")
        hide_button.clicked.connect(self.mesh_container.toggle_hide_meshes_button)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(hide_button, row, column)

        # Mirror buttons
        # mirror_x mean mirror left/right
        # mirror_y mean mirror up/down
        self.mirror_x_button = QtWidgets.QPushButton("Flip Left/Right")
        self.mirror_x_button.clicked.connect(lambda _, direction="x": self.mirror_actors(direction))
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(self.mirror_x_button, row, column)
        self.mirror_y_button = QtWidgets.QPushButton("Flip Up/Down")
        self.mirror_y_button.clicked.connect(lambda _, direction="y": self.mirror_actors(direction))
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(self.mirror_y_button, row, column)

        # Create the remove button
        remove_button = QtWidgets.QPushButton("Remove Actor")
        remove_button.clicked.connect(self.remove_actors_button)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(remove_button, row, column)

        # Create the video related button
        self.play_video_button.clicked.connect(self.video_container.play_video)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(self.play_video_button, row, column)

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

        # Create a container widget for the buttons
        button_container = QtWidgets.QWidget()
        self.button_layout = QtWidgets.QVBoxLayout()
        self.button_layout.setSpacing(0)  # Remove spacing between buttons
        button_container.setLayout(self.button_layout)

        self.button_layout.addStretch()

        # Set the container widget as the scroll area's widget
        scroll_area.setWidget(button_container)

        self.display.setLayout(display_layout)
        self.panel_layout.addWidget(self.display)

    #^ Panel Output
    def panel_output(self):
        # Add a spacer to the top of the main layout
        self.output = QtWidgets.QGroupBox("Output")
        output_layout = QtWidgets.QVBoxLayout()
        output_layout.setContentsMargins(10, 15, 10, 5)

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
        self.frame.setFixedSize(*self.window_size)
        self.plotter = CustomQtInteractor(self.frame, self)
        # self.plotter.setFixedSize(*self.window_size)
        self.signal_close.connect(self.plotter.close)

    def show_plot(self):
        self.plotter.enable_joystick_actor_style()
        self.plotter.enable_trackball_actor_style()
        
        self.plotter.add_axes(color='white')
        self.plotter.add_camera_orientation_widget()

        self.plotter.show()
        self.show()

    def register_pose(self, pose):
        for mesh_data in self.mesh_store.meshes.values(): 
            mesh_data.actor.user_matrix = pose
            mesh_data.undo_poses.append(pose)
            mesh_data.undo_poses = mesh_data.undo_poses[-20:]
        
    def toggle_register(self, pose):
        if self.mesh_store.toggle_anchor_mesh: self.register_pose(pose)
        else: self.mesh_store.meshes[self.mesh_store.reference].actor.user_matrix = pose
            
    #* when there is no anchor, the base vertices changes when we move around the objects
    def toggle_anchor(self, pose):
        if self.mesh_store.toggle_anchor_mesh: self.register_pose(pose)
        else:
            for mesh_data in self.mesh_store.meshes.values():
                verts, _ = utils.get_mesh_actor_vertices_faces(mesh_data.actor)
                mesh_data.undo_vertices.append(verts)
                mesh_data.undo_vertices = mesh_data.undo_vertices[-20:]
                vertices = utils.transform_vertices(verts, mesh_data.actor.user_matrix)
                mesh_data.pv_mesh.points = vertices
                mesh_data.actor.user_matrix = np.eye(4)
                mesh_data.initial_pose = np.eye(4) # if meshes are not anchored, then there the initial pose will always be np.eye(4)
                
    def button_actor_name_clicked(self, name, output_text=True):
        if name in self.mesh_store.meshes:
            self.mesh_store.reference = name
            mesh_data = self.mesh_store.meshes[name]
            curr_opacity = mesh_data.actor.GetProperty().opacity
            self.opacity_spinbox.setValue(curr_opacity)
            self.toggle_anchor(mesh_data.actor.user_matrix)
            self.color_button.setText(mesh_data.color)
            self.mesh_container.set_color(mesh_data.color, name)
            text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
            mesh_data.actor.user_matrix[0, 0], mesh_data.actor.user_matrix[0, 1], mesh_data.actor.user_matrix[0, 2], mesh_data.actor.user_matrix[0, 3], 
            mesh_data.actor.user_matrix[1, 0], mesh_data.actor.user_matrix[1, 1], mesh_data.actor.user_matrix[1, 2], mesh_data.actor.user_matrix[1, 3], 
            mesh_data.actor.user_matrix[2, 0], mesh_data.actor.user_matrix[2, 1], mesh_data.actor.user_matrix[2, 2], mesh_data.actor.user_matrix[2, 3],
            mesh_data.actor.user_matrix[3, 0], mesh_data.actor.user_matrix[3, 1], mesh_data.actor.user_matrix[3, 2], mesh_data.actor.user_matrix[3, 3])
            if output_text: self.output_text.append(f"--> Mesh {name} pose is:"); self.output_text.append(text)
        else:
            self.color_button.setText("Color")
            if name == 'image': curr_opacity = self.image_store.image_opacity
            elif name == 'mask': curr_opacity = self.mask_store.mask_opacity
            else: curr_opacity = self.opacity_spinbox.value()
            self.mesh_store.reference = None #* For fixing some bugs in segmesh render function
            self.opacity_spinbox.setValue(curr_opacity)
            
    def check_button(self, name, output_text=True):  
        button = next((btn for btn in self.button_group_actors_names.buttons() if btn.text() == name), None)
        if button:
            button.setChecked(True)
            self.button_actor_name_clicked(name=name, output_text=output_text)

    def add_button_actor_name(self, name):
        button = QtWidgets.QPushButton(name)
        button.setCheckable(True)  # Set the button to be checkable so it is highlighted, very important
        button.clicked.connect(lambda _, name=name: self.button_actor_name_clicked(name))
        button.setChecked(True)
        button.setFixedSize(self.display.size().width(), 50)
        self.button_layout.insertWidget(0, button) # insert from the top # self.button_layout.addWidget(button)
        self.button_group_actors_names.addButton(button)
        self.button_actor_name_clicked(name=name)
    
    def opacity_value_change(self, value):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button:
            name = checked_button.text()
            if name == 'image': self.image_container.set_image_opacity(value)
            elif name == 'mask': self.mask_container.set_mask_opacity(value)
            elif name in self.mesh_store.meshes:
                mesh_data = self.mesh_store.meshes[name]
                mesh_data.opacity = value
                mesh_data.previous_opacity = value
                self.mesh_container.set_mesh_opacity(name, mesh_data.opacity)
    
    def update_color_button_text(self, text, popup):
        self.color_button.setText(text)
        popup.close() # automatically close the popup window

    def show_color_popup(self):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button:
            name = checked_button.text()
            if name in self.mesh_store.meshes:
                popup = PopUpDialog(self, on_button_click=lambda text: self.update_color_button_text(text, popup), for_mesh=True)
                button_position = self.color_button.mapToGlobal(QPoint(0, 0))
                popup.move(button_position + QPoint(self.color_button.width(), 0))
                popup.exec_()
                color = self.color_button.text()
                self.mesh_store.meshes[name].color = color
                try: self.mesh_container.set_color(color, name)
                except ValueError: utils.display_warning(f"Cannot set color ({color}) to {name}")
            elif name == 'mask':
                popup = PopUpDialog(self, on_button_click=lambda text: self.update_color_button_text(text, popup), for_mesh=False)
                button_position = self.color_button.mapToGlobal(QPoint(0, 0))
                popup.move(button_position + QPoint(self.color_button.width(), 0))
                popup.exec_()
                color = self.color_button.text()
                self.mask_store.color = color
                self.mask_container.set_mask_color(color)
            else: utils.display_warning("Only be able to color mesh or mask objects")
        else: utils.display_warning("Need to select an actor first")
 
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
            if 'image_path' in workspace: self.image_container.add_image_file(image_path=root / pathlib.Path(*workspace['image_path'].split("\\")))
            if 'video_path' in workspace: self.video_container.add_video_file(video_path=root / pathlib.Path(*workspace['video_path'].split("\\")))
            if 'mask_path' in workspace: self.mask_container.add_mask_file(mask_path=root / pathlib.Path(*workspace['mask_path'].split("\\")))
            if 'bbox_path' in workspace: self.bbox_container.add_bbox_file(bbox_path=root / pathlib.Path(*workspace['bbox_path'].split("\\")))
            if 'mesh_path' in workspace:
                mesh_paths = workspace['mesh_path']
                for path in mesh_paths: self.mesh_container.add_mesh_file(mesh_path=root / pathlib.Path(*path.split("\\")))
            if 'pose_path' in workspace: self.mesh_container.add_pose_file(pose_path=root / pathlib.Path(*workspace['pose_path'].split("\\")))
            # reset camera
            self.camera_container.reset_camera()

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
                self.anchor_button.setCheckable(False)
                self.anchor_button.setEnabled(False)
                self.play_video_button.setEnabled(False)
                self.play_video_button.setText(f"Image ({self.folder_store.current_image}/{self.folder_store.total_image})")
                self.camera_container.reset_camera()
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
            self.color_button.setText("Color")
        elif name in self.point_store.point_actors:
            actor = self.point_store.point_actors[name]
            self.point_store.remove_point(name)

        self.plotter.remove_actor(actor)
        self.track_actors_names.remove(name)
        # remove the button from the button group
        self.button_group_actors_names.removeButton(button)
        # remove the button from the self.button_layout widget
        self.button_layout.removeWidget(button)
        # offically delete the button
        button.deleteLater()

        # clear out the plot if there is no actor
        if (self.image_store.image_actor is None) and (self.mask_store.mask_actor is None) and (len(self.mesh_store.meshes) == 0) and (len(self.point_store.point_actors) == 0): 
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
                self.color_button.setText("Color")
            elif name in self.point_store.point_actors:
                actor = self.point_store.point_actors[name]
                self.point_store.remove_point(name)

            self.plotter.remove_actor(actor)
            # remove the button from the button group
            self.button_group_actors_names.removeButton(button)
            # remove the button from the self.button_layout widget
            self.button_layout.removeWidget(button)
            # offically delete the button
            button.deleteLater()

        self.mesh_store.reset()
        self.point_store.reset()
        self.video_store.reset()
        self.folder_store.reset()
        self.workspace_path = ''
        self.track_actors_names.clear()
        self.reset_output_text()

        self.color_button.setText("Color")
        self.anchor_button.setCheckable(True)
        self.anchor_button.setChecked(True)
        self.anchor_button.setEnabled(True)
        self.play_video_button.setEnabled(True)
        self.play_video_button.setText("Play Video")
        self.opacity_spinbox.setValue(0.3)
        
        self.hintLabel.show()

    def set_object_distance(self):
        distance, ok = QtWidgets.QInputDialog().getText(QtWidgets.QMainWindow(), 'Input', "Set Objects distance to camera", text=str(self.object_distance))
        if ok:
            distance = float(distance)
            if self.image_store.image_actor is not None:
                self.image_store.image_pv.translate(-np.array([0, 0, self.image_store.image_pv.center[-1]]), inplace=True)
                self.image_store.image_pv.translate(np.array([0, 0, distance]), inplace=True)
                self.image_container.set_object_distance(distance)
            if self.mask_store.mask_actor is not None:
                self.mask_store.mask_pv.translate(-np.array([0, 0, self.mask_store.mask_pv.center[-1]]), inplace=True)
                self.mask_store.mask_pv.translate(np.array([0, 0, distance]), inplace=True)
                self.mask_container.set_object_distance(distance)
            if self.bbox_store.bbox_actor is not None:
                self.bbox_store.bbox_pv.translate(-np.array([0, 0, self.bbox_store.bbox_pv.center[-1]]), inplace=True)
                self.bbox_store.bbox_pv.translate(np.array([0, 0, distance]), inplace=True)
                self.bbox_container.set_object_distance(distance)
            if len(self.mesh_store.meshes) > 0:
                for mesh_data in self.mesh_store.meshes.values(): 
                    user_matrix = mesh_data.actor.user_matrix
                    user_matrix[2, 3] -= self.object_distance
                    user_matrix[2, 3] += distance
                    mesh_data.actor.user_matrix = user_matrix

            self.object_distance = distance
            self.camera_container.reset_camera()