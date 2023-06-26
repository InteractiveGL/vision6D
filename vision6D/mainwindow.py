# General import
import os
os.environ["QT_API"] = "pyqt5" # Setting the Qt bindings for QtPy
import ast
import json
import copy
import pathlib
import functools

import trimesh
import PIL.Image
import numpy as np
import pyvista as pv

# Qt5 import
from PyQt5 import QtWidgets, QtGui
from pyvistaqt import MainWindow
from PyQt5.QtCore import Qt, QPoint

# self defined package import
from . import utils

from .widgets import GetTextDialog
from .widgets import CustomQtInteractor
from .widgets import CalibrationPopWindow
from .widgets import CameraPropsInputDialog
from .widgets import PopUpDialog
from .widgets import LabelWindow

from .components import ImageStore
from .components import MaskStore
from .components import CameraStore
from .components import MeshStore
from .components import VideoStore
from .components import FolderStore

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

        # initialize
        self.mirror_x = False
        self.mirror_y = False
        self.track_actors_names = []
        self.button_group_actors_names = QtWidgets.QButtonGroup(self)
        self.toggle_hide_meshes_flag = False

        # initial stores
        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.camera_store = CameraStore(self.window_size)
        self.mesh_store = MeshStore(self.window_size)
        self.video_store = VideoStore()
        self.folder_store = FolderStore()
        
        # Dialogs to record users input info
        self.input_dialog = QtWidgets.QInputDialog()
        self.file_dialog = QtWidgets.QFileDialog()
        self.get_text_dialog = GetTextDialog()

        # Set panel bar
        self.set_panel_bar()
        
        # Set menu bar
        self.set_menu_bars()

        # Create the plotter
        self.create_plotter()

        # Set up camera after create plotter
        self.set_camera_props()

        # Set up the main layout with the left panel and the render window using QSplitter
        self.main_layout = QtWidgets.QHBoxLayout(self.main_widget)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.panel_widget)
        self.splitter.addWidget(self.plotter)
        self.splitter.setStretchFactor(0, 1) # for self.panel_widget
        self.splitter.setStretchFactor(1, 3) # for self.plotter

        self.main_layout.addWidget(self.splitter)

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

        # Show the plotter
        self.show_plot()

        # Shortcut key bindings
        self.key_bindings()

    def key_bindings(self):
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
                    self.add_mesh_file(mesh_path=file_path)
                # Load video file
                elif file_path.endswith(('.avi', '.mp4', '.mkv', '.mov', '.fly', '.wmv', '.mpeg', '.asf', '.webm')):
                    self.add_video_file(video_path=file_path)
                # Load image/mask file
                elif file_path.endswith(('.png', '.jpg', 'jpeg', '.tiff', '.bmp', '.webp', '.ico')):  # add image/mask
                    file_data = np.array(PIL.Image.open(file_path).convert('L'), dtype='uint8')
                    unique, _ = np.unique(file_data, return_counts=True)
                    if len(unique) > 2: self.add_image_file(image_path=file_path)
                    else: self.add_mask_file(mask_path=file_path)
                        
                elif file_path.endswith('.npy'): self.add_pose_file(pose_path=file_path)
                else:
                    QtWidgets.QMessageBox.warning(self, 'vision6D', "File format is not supported!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    return 0

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
        fileMenu.addAction('Add Video', functools.partial(self.add_video_file, prompt=True))
        fileMenu.addAction('Add Image', functools.partial(self.add_image_file, prompt=True))
        fileMenu.addAction('Add Mask', functools.partial(self.add_mask_file, prompt=True))
        fileMenu.addAction('Add Mesh', functools.partial(self.add_mesh_file, prompt=True))
        fileMenu.addAction('Draw Mask', self.draw_mask)
        fileMenu.addAction('Clear', self.clear_plot)

        # allow to export files
        exportMenu = mainMenu.addMenu('Export')
        exportMenu.addAction('Image', self.export_image)
        exportMenu.addAction('Mask', self.export_mask)
        exportMenu.addAction('Pose', self.export_pose)
        exportMenu.addAction('Mesh Render', self.export_mesh_render)
        exportMenu.addAction('SegMesh Render', self.export_segmesh_render)
        
        # Add video related actions
        VideoFolderMenu = mainMenu.addMenu('Video/Folder')
        VideoFolderMenu.addAction('Play', self.play_video)
        VideoFolderMenu.addAction('Sample', self.sample_video)
        VideoFolderMenu.addAction('Save Frame', self.save_frame)
        VideoFolderMenu.addAction('Prev Frame', self.prev_frame)
        VideoFolderMenu.addAction('Next Frame', self.next_frame)

        # Add camera related actions
        CameraMenu = mainMenu.addMenu('Camera')
        CameraMenu.addAction('Calibrate', self.camera_calibrate)
        CameraMenu.addAction('Reset Camera (d)', self.reset_camera)
        CameraMenu.addAction('Zoom In (x)', self.zoom_in)
        CameraMenu.addAction('Zoom Out (z)', self.zoom_out)

        # add mirror actors related actions
        mirrorMenu = mainMenu.addMenu('Mirror')
        mirrorMenu.addAction('Mirror X axis', functools.partial(self.mirror_actors, direction='x'))
        mirrorMenu.addAction('Mirror Y axis', functools.partial(self.mirror_actors, direction='y'))
        
        # Add register related actions
        RegisterMenu = mainMenu.addMenu('Register')
        RegisterMenu.addAction('Reset GT Pose (k)', self.reset_gt_pose)
        RegisterMenu.addAction('Reset Mask (t)', self.reset_mask)
        RegisterMenu.addAction('Update GT Pose (l)', self.update_gt_pose)
        RegisterMenu.addAction('Undo Pose (s)', self.undo_pose)

        # Add pnp algorithm related actions
        PnPMenu = mainMenu.addMenu('PnP')
        PnPMenu.addAction('EPnP with mesh', self.epnp_mesh)
        PnPMenu.addAction('EPnP with nocs mask', functools.partial(self.epnp_mask, True))
        PnPMenu.addAction('EPnP with latlon mask', functools.partial(self.epnp_mask, False))

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
        
    #^ Panel Display
    def panel_display(self):
        self.display = QtWidgets.QGroupBox("Console")
        display_layout = QtWidgets.QVBoxLayout()
        display_layout.setContentsMargins(10, 15, 10, 0)

        #* Create the top widgets (layout)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Create Grid layout for function buttons
        top_grid_layout = QtWidgets.QGridLayout()

        # Create the set camera button
        set_camera_button = QtWidgets.QPushButton("Set Camera")
        set_camera_button.clicked.connect(self.set_camera)
        top_grid_layout.addWidget(set_camera_button, 0, 0)

        # Create the actor pose button
        actor_pose_button = QtWidgets.QPushButton("Set Pose")
        actor_pose_button.clicked.connect(self.set_pose)
        top_grid_layout.addWidget(actor_pose_button, 0, 1)

        # Create the draw mask button
        draw_mask_button = QtWidgets.QPushButton("Draw Mask")
        draw_mask_button.clicked.connect(self.draw_mask)
        top_grid_layout.addWidget(draw_mask_button, 0, 2)

        # Create the video related button
        self.play_video_button = QtWidgets.QPushButton("Play Video")
        self.play_video_button.clicked.connect(self.play_video)
        top_grid_layout.addWidget(self.play_video_button, 0, 3)

        self.sample_video_button = QtWidgets.QPushButton("Sample Video")
        self.sample_video_button.clicked.connect(self.sample_video)
        top_grid_layout.addWidget(self.sample_video_button, 1, 0)

        self.save_frame_button = QtWidgets.QPushButton("Save Frame")
        self.save_frame_button.clicked.connect(self.save_frame)
        top_grid_layout.addWidget(self.save_frame_button, 1, 1)

        self.prev_frame_button = QtWidgets.QPushButton("Prev Frame")
        self.prev_frame_button.clicked.connect(self.prev_frame)
        top_grid_layout.addWidget(self.prev_frame_button, 1, 2)

        self.next_frame_button = QtWidgets.QPushButton("Next Frame")
        self.next_frame_button.clicked.connect(self.next_frame)
        top_grid_layout.addWidget(self.next_frame_button, 1, 3)

        top_grid_widget = QtWidgets.QWidget()
        top_grid_widget.setLayout(top_grid_layout)
        top_layout.addWidget(top_grid_widget)
        display_layout.addLayout(top_layout)

        #* Create the bottom widgets
        actor_widget = QtWidgets.QLabel("Actors")
        display_layout.addWidget(actor_widget)

        actor_grid_layout = QtWidgets.QGridLayout()
        actor_grid_layout.setContentsMargins(10, 15, 10, 0)

        # Create the color dropdown menu
        self.color_button = QtWidgets.QPushButton("Color")
        self.color_button.clicked.connect(self.show_color_popup)
        actor_grid_layout.addWidget(self.color_button, 0, 0)
        
        # Create the opacity spinbox
        self.opacity_spinbox = QtWidgets.QDoubleSpinBox()
        self.opacity_spinbox.setMinimum(0.0)
        self.opacity_spinbox.setMaximum(1.0)
        self.opacity_spinbox.setDecimals(2)
        self.opacity_spinbox.setSingleStep(0.05)
        self.ignore_spinbox_value_change = True
        self.opacity_spinbox.setValue(0.3)
        self.ignore_spinbox_value_change = False 
        self.opacity_spinbox.valueChanged.connect(self.opacity_value_change)
        actor_grid_layout.addWidget(self.opacity_spinbox, 0, 1)

        # Create the hide button
        hide_button = QtWidgets.QPushButton("toggle meshes")
        hide_button.clicked.connect(self.toggle_hide_meshes_button)
        actor_grid_layout.addWidget(hide_button, 0, 2)

        # Create the remove button
        remove_button = QtWidgets.QPushButton("Remove Actor")
        remove_button.clicked.connect(self.remove_actors_button)
        actor_grid_layout.addWidget(remove_button, 0, 3)
        display_layout.addLayout(actor_grid_layout)

        # Create the spacing button
        self.spacing_button = QtWidgets.QPushButton("Spacing")
        self.spacing_button.clicked.connect(self.set_spacing)
        actor_grid_layout.addWidget(self.spacing_button, 1, 0)

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
        output_layout.setContentsMargins(10, 15, 10, 0)

        #* Create the top widgets (layout)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Create Grid layout for function buttons
        grid_layout = QtWidgets.QGridLayout()

        # Create the set camera button
        copy_text_button = QtWidgets.QPushButton("Copy")
        copy_text_button.clicked.connect(self.copy_output_text)
        grid_layout.addWidget(copy_text_button, 0, 2, 1, 1)

        # Create the actor pose button
        clear_text_button = QtWidgets.QPushButton("Clear")
        clear_text_button.clicked.connect(self.clear_output_text)
        grid_layout.addWidget(clear_text_button, 0, 3, 1, 1)

        grid_widget = QtWidgets.QWidget()
        grid_widget.setLayout(grid_layout)
        top_layout.addWidget(grid_widget)
        output_layout.addLayout(top_layout)

        self.output_text = QtWidgets.QTextEdit()
        self.output_text.setReadOnly(True)
        # Access to the system clipboard
        self.clipboard = QtGui.QGuiApplication.clipboard()
        output_layout.addWidget(self.output_text)
        self.output.setLayout(output_layout)
        self.panel_layout.addWidget(self.output)

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
        
        self.plotter.add_axes()
        self.plotter.add_camera_orientation_widget()

        self.plotter.show()
        self.show()

    #* Interface  
    def button_actor_name_clicked(self, text):
        if text in self.mesh_store.mesh_actors:
            self.color_button.setText(self.mesh_store.mesh_colors[text])
            self.mesh_store.reference = text
            self.current_pose()
            curr_opacity = self.mesh_store.mesh_actors[self.mesh_store.reference].GetProperty().opacity
            self.opacity_spinbox.setValue(curr_opacity)
        else:
            self.color_button.setText("Color")
            if text == 'image': curr_opacity = self.image_store.image_opacity
            elif text == 'mask': curr_opacity = self.mask_store.mask_opacity
            else: curr_opacity = self.opacity_spinbox.value()
            self.opacity_spinbox.setValue(curr_opacity)
            # self.mesh_store.reference = None
        
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
        self.mesh_store.mesh_opacity[name] = surface_opacity
        self.mesh_store.mesh_actors[name].user_matrix = pv.array_from_vtkmatrix(self.mesh_store.mesh_actors[name].GetMatrix())
        self.mesh_store.mesh_actors[name].GetProperty().opacity = surface_opacity
        self.plotter.add_actor(self.mesh_store.mesh_actors[name], pickable=True, name=name)

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

    def add_mesh(self, mesh_source, transformation_matrix=None):
        """ add a mesh to the pyqt frame """
        mesh_data, source_verts, source_faces = self.mesh_store.add_mesh(mesh_source)

        if mesh_data:      
            mesh = self.plotter.add_mesh(mesh_data, color=self.mesh_store.mesh_colors[self.mesh_store.mesh_name], opacity=self.mesh_store.mesh_opacity[self.mesh_store.mesh_name], name=self.mesh_store.mesh_name)
            mesh.user_matrix = self.mesh_store.transformation_matrix if transformation_matrix is None else transformation_matrix
            actor, _ = self.plotter.add_actor(mesh, pickable=True, name=self.mesh_store.mesh_name)

            actor_vertices, actor_faces = utils.get_mesh_actor_vertices_faces(actor)
            assert (actor_vertices == source_verts).all(), "vertices should be the same"
            assert (actor_faces == source_faces).all(), "faces should be the same"
            assert actor.name == self.mesh_store.mesh_name, "actor's name should equal to mesh_name"
            
            self.mesh_store.mesh_actors[self.mesh_store.mesh_name] = actor
            self.color_button.setText(self.mesh_store.mesh_colors[self.mesh_store.mesh_name])

            # add remove current mesh to removeMenu
            if self.mesh_store.mesh_name not in self.track_actors_names:
                self.track_actors_names.append(self.mesh_store.mesh_name)
                self.add_button_actor_name(self.mesh_store.mesh_name)

            self.check_button(self.mesh_store.mesh_name)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "The mesh format is not supported!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def reset_camera(self):
        self.plotter.camera = self.camera_store.camera.copy()

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
        change = 0.05
        if not up: change *= -1
        current_opacity = self.opacity_spinbox.value()
        current_opacity += change
        current_opacity = np.clip(current_opacity, 0, 1)
        self.opacity_spinbox.setValue(current_opacity)

    def opacity_value_change(self, value):
        if self.ignore_spinbox_value_change: return 0
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button:
            actor_name = checked_button.text()
            if actor_name == 'image': self.set_image_opacity(value)
            elif actor_name == 'mask': self.set_mask_opacity(value)
            elif actor_name in self.mesh_store.mesh_actors: 
                self.mesh_store.store_mesh_opacity[actor_name] = copy.deepcopy(self.mesh_store.mesh_opacity[actor_name])
                self.mesh_store.mesh_opacity[actor_name] = value
                self.set_mesh_opacity(actor_name, self.mesh_store.mesh_opacity[actor_name])
        else:
            self.ignore_spinbox_value_change = True
            self.opacity_spinbox.setValue(value)
            self.ignore_spinbox_value_change = False
            return 0
        
    def toggle_hide_meshes_button(self):
        self.toggle_hide_meshes_flag = not self.toggle_hide_meshes_flag
        
        if self.toggle_hide_meshes_flag:
            for button in self.button_group_actors_names.buttons():
                if button.text() in self.mesh_store.mesh_actors:
                    button.setChecked(True); self.opacity_value_change(0)
    
            checked_button = self.button_group_actors_names.checkedButton()
            if checked_button: 
                self.ignore_spinbox_value_change = True
                self.opacity_spinbox.setValue(0.0)
                self.ignore_spinbox_value_change = False
            else: QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        
        else:
            for button in self.button_group_actors_names.buttons():
                if button.text() in self.mesh_store.mesh_actors:
                    button.setChecked(True)
                    self.opacity_value_change(self.mesh_store.store_mesh_opacity[button.text()])

            checked_button = self.button_group_actors_names.checkedButton()
            if checked_button:
                self.ignore_spinbox_value_change = True
                if checked_button.text() in self.mesh_store.mesh_actors: self.opacity_spinbox.setValue(self.mesh_store.mesh_opacity[checked_button.text()])
                self.ignore_spinbox_value_change = False
            else: QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
 
    def set_camera_props(self):
        self.camera_store.set_camera_intrinsics()
        self.camera_store.set_camera_extrinsics()
        self.plotter.camera = self.camera_store.camera.copy()

    def camera_calibrate(self):
        if self.image_store.image_path:
            original_image = np.array(PIL.Image.open(self.image_store.image_path), dtype='uint8')
            # make the the original image shape is [h, w, 3] to match with the rendered calibrated_image
            original_image = original_image[..., :3]
            if len(original_image.shape) == 2: original_image = original_image[..., None]
            if original_image.shape[-1] == 1: original_image = np.dstack((original_image, original_image, original_image))
            calibrated_image = np.array(self.image_store.render_image(self.plotter.camera.copy()), dtype='uint8')
            if original_image.shape != calibrated_image.shape:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Original image shape is not equal to calibrated image shape!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return 0
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
        calibrate_pop = CalibrationPopWindow(calibrated_image, original_image)
        calibrate_pop.exec_()

    def set_camera(self):
        dialog = CameraPropsInputDialog(
            line1=("Fx", self.camera_store.fx), 
            line2=("Fy", self.camera_store.fy), 
            line3=("Cx", self.camera_store.cx), 
            line4=("Cy", self.camera_store.cy), 
            line5=("View Up", self.camera_store.cam_viewup), 
            line6=("Cam Position", self.camera_store.cam_position))
        if dialog.exec():
            fx, fy, cx, cy, cam_viewup, cam_position = dialog.getInputs()
            pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup, pre_cam_position = self.camera_store.fx, self.camera_store.fy, self.camera_store.cx, self.camera_store.cy, self.camera_store.cam_viewup, self.camera_store.cam_position
            if not (fx == '' or fy == '' or cx == '' or cy == '' or cam_viewup == '' or cam_position == ''):
                try:
                    self.camera_store.fx, self.camera_store.fy, self.camera_store.cx, self.camera_store.cy, self.camera_store.cam_viewup, self.camera_store.cam_position = ast.literal_eval(fx), ast.literal_eval(fy), ast.literal_eval(cx), ast.literal_eval(cy), ast.literal_eval(cam_viewup), ast.literal_eval(cam_position)
                    self.set_camera_props()
                except:
                    self.camera_store.fx, self.camera_store.fy, self.camera_store.cx, self.camera_store.cy, self.camera_store.cam_viewup, self.camera_store.cam_position = pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup, pre_cam_position
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
                    self.mesh_store.transformation_matrix = transformation_matrix
                    if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                    if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                    self.add_pose(matrix=transformation_matrix)
                    return 0
            except: 
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return None
        else: 
            return None

    def add_workspace(self, workspace_path='', prompt=False):
        if prompt:
            workspace_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.json)")
        if workspace_path:
            self.hintLabel.hide()
            with open(str(workspace_path), 'r') as f: workspace = json.load(f)
            if 'image_path' in workspace: self.add_image_file(image_path=workspace['image_path'])
            if 'video_path' in workspace: self.add_video_file(video_path=workspace['video_path'])
            if 'mask_path' in workspace: self.add_mask_file(mask_path=workspace['mask_path'])
            # need to load pose before loading meshes
            if 'pose_path' in workspace: self.add_pose_file(pose_path=workspace['pose_path'])
            if 'mesh_path' in workspace:
                mesh_paths = workspace['mesh_path']
                for path in mesh_paths: self.add_mesh_file(mesh_path=path)
            
            # reset camera
            self.reset_camera()
   
    def add_folder(self, folder_path='', prompt=False):
        if prompt: 
            folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            if self.video_store.video_path: self.clear_plot() # main goal is to set video_path to None
            image_path, mask_path, pose_path, mesh_path = self.folder_store.add_folder(folder_path=folder_path)
            if image_path or mask_path or pose_path or mesh_path:
                if image_path: self.add_image_file(image_path=image_path)
                if mask_path: self.add_mask_file(mask_path=mask_path)
                if pose_path: self.add_pose_file(pose_path=pose_path)
                if mesh_path: 
                    with open(mesh_path, 'r') as f: mesh_path = f.read().splitlines()
                    for path in mesh_path: self.add_mesh_file(path)
                self.play_video_button.setEnabled(False)
                self.sample_video_button.setEnabled(False)
                self.play_video_button.setText(f"Frame ({self.folder_store.current_frame}/{self.folder_store.total_frame})")
                self.output_text.append(f"-> Current frame is ({self.folder_store.current_frame}/{self.folder_store.total_frame})")
                self.reset_camera()
            else:
                self.folder_store.reset()
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Not a valid folder, please reload a folder", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def add_video_file(self, video_path='', prompt=False):
        if prompt:
            video_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.avi *.mp4 *.mkv *.mov *.fly *.wmv *.mpeg *.asf *.webm)")
        if video_path:
            if self.folder_store.folder_path: self.clear_plot() # main goal is to set folder_path to None
            self.hintLabel.hide()
            # self.folder_store.reset()
            self.video_store.add_video(video_path)
            self.play_video_button.setEnabled(True)
            self.sample_video_button.setEnabled(True)
            self.play_video_button.setText(f"Play ({self.video_store.current_frame}/{self.video_store.total_frame})")
            self.output_text.append(f"-> Load video {self.video_store.video_path} into vision6D")
            self.output_text.append(f"-> Current frame is ({self.video_store.current_frame}/{self.video_store.total_frame})")
            self.load_per_frame_info()
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

    def add_mesh_file(self, mesh_path='', prompt=False):
        if prompt: 
            mesh_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.mesh *.ply *.stl *.obj *.off *.dae *.fbx *.3ds *.x3d)") 
        if mesh_path:
            self.hintLabel.hide()
            transformation_matrix = self.mesh_store.transformation_matrix
            if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix                   
            self.add_mesh(mesh_path, transformation_matrix)

    def add_pose(self, matrix:np.ndarray=None, rot:np.ndarray=None, trans:np.ndarray=None):
        if matrix is not None: 
            self.mesh_store.initial_pose = matrix
            self.reset_gt_pose()
        else:
            if (rot and trans): matrix = np.vstack((np.hstack((rot, trans)), [0, 0, 0, 1]))
        self.reset_camera()

    def add_pose_file(self, pose_path):
        if pose_path:
            self.hintLabel.hide()
            self.mesh_store.pose_path = pose_path
            transformation_matrix = np.load(self.mesh_store.pose_path)
            self.mesh_store.transformation_matrix = transformation_matrix
            if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            self.add_pose(matrix=transformation_matrix)
    
    def register_pose(self, pose):
        for actor_name, actor in self.mesh_store.mesh_actors.items():
            actor.user_matrix = pose
            self.plotter.add_actor(actor, pickable=True, name=actor_name)

    def reset_gt_pose(self):
        self.output_text.append(f"-> Reset the GT pose to: \n{self.mesh_store.initial_pose}")
        self.register_pose(self.mesh_store.initial_pose)

    def update_gt_pose(self):
        self.mesh_store.initial_pose = self.mesh_store.transformation_matrix
        self.current_pose()
        self.output_text.append(f"Update the GT pose to: \n{self.mesh_store.initial_pose}")
            
    def current_pose(self):
        self.mesh_store.current_pose()
        self.output_text.append(f"-> Current reference mesh is: <span style='background-color:yellow; color:black;'>{self.mesh_store.reference}</span>")
        self.output_text.append(f"Current pose is: \n{self.mesh_store.transformation_matrix}")
        self.register_pose(self.mesh_store.transformation_matrix)

    def undo_pose(self):
        if self.button_group_actors_names.checkedButton():
            actor_name = self.button_group_actors_names.checkedButton().text()
            if self.mesh_store.undo_poses and len(self.mesh_store.undo_poses[actor_name]) != 0: 
                self.mesh_store.undo_pose(actor_name)
                # register the rest meshes' pose to current pose
                self.check_button(actor_name)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Choose a mesh actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def mirror_actors(self, direction):

        if direction == 'x': self.mirror_x = not self.mirror_x
        elif direction == 'y': self.mirror_y = not self.mirror_y

        #^ mirror the image actor
        if self.image_store.image_path: self.add_image(self.image_store.image_path)

        #^ mirror the mask actor
        if self.mask_store.mask_path: self.add_mask(self.mask_store.mask_path)

        #^ mirror the mesh actors
        if self.mesh_store.reference:
            transformation_matrix = self.mesh_store.initial_pose
            if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            self.add_mesh(self.mesh_store.meshdict[self.mesh_store.reference], transformation_matrix)
            self.mesh_store.undo_poses = {}
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
        elif name in self.mesh_store.mesh_actors: 
            actor = self.mesh_store.mesh_actors[name]
            self.mesh_store.remove_mesh(name)
            self.color_button.setText("Color")

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
        if self.image_store.image_actor is None and self.mask_store.mask_actor is None and len(self.mesh_store.mesh_actors) == 0: self.clear_plot()

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
            elif name in self.mesh_store.mesh_actors: actor = self.mesh_store.mesh_actors[name]
            self.plotter.remove_actor(actor)
            # remove the button from the button group
            self.button_group_actors_names.removeButton(button)
            # remove the button from the self.button_layout widget
            self.button_layout.removeWidget(button)
            # offically delete the button
            button.deleteLater()

        self.image_store.reset()
        self.mask_store.reset()
        self.mesh_store.reset()
        self.video_store.reset()
        self.folder_store.reset()

        # Re-initial the dictionaries
        self.mirror_x = False
        self.mirror_y = False

        self.track_actors_names = []
        self.color_button.setText("Color")
        self.play_video_button.setText("Play Video")
        self.clear_output_text()

        self.hintLabel.show()

    def export_image(self):
        if self.image_store.image_actor:
            image = self.image_store.render_image(camera=self.camera_store.camera.copy())
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
            image = self.mask_store.render_mask(camera=self.camera_store.camera.copy())
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
        if self.mesh_store.reference: 
            self.update_gt_pose()
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Pose Files (*.npy)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.npy')
                np.save(output_path, self.mesh_store.transformation_matrix)
                self.output_text.append(f"-> Saved:\n{self.mesh_store.transformation_matrix}\nExport to:\n {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
    
    def export_mesh_render(self, save_render=True):

        if self.mesh_store.reference:
            image = self.mesh_store.render_mesh(camera=self.camera_store.camera.copy())
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

        if self.mesh_store.reference and self.mask_store.mask_actor:
            segmask = self.mask_store.render_mask(camera=self.camera_store.camera.copy())
            if np.max(segmask) > 1: segmask = segmask / 255
            image = self.mesh_store.render_mesh(camera=self.camera_store.camera.copy())
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
            if actor_name in self.mesh_store.mesh_actors:
                spacing, ok = self.input_dialog.getText(self, 'Input', "Set Spacing", text=str(self.mesh_store.mesh_spacing))
                if ok:
                    try: self.mesh_store.mesh_spacing = ast.literal_eval(spacing)
                    except: QtWidgets.QMessageBox.warning(self, 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    self.add_mesh(self.mesh_store.meshdict[actor_name])
            else:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select a mesh object instead", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select a mesh actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def set_scalar(self, nocs, actor_name):
        mesh_data, colors = self.mesh_store.set_scalar(nocs, actor_name, self.mirror_x, self.mirror_y)
        if mesh_data:
            mesh = self.plotter.add_mesh(mesh_data, scalars=colors, rgb=True, opacity=self.mesh_store.mesh_opacity[actor_name], name=actor_name)
            transformation_matrix = pv.array_from_vtkmatrix(self.mesh_store.mesh_actors[actor_name].GetMatrix())
            mesh.user_matrix = transformation_matrix
            actor, _ = self.plotter.add_actor(mesh, pickable=True, name=actor_name)
            actor_colors = utils.get_mesh_actor_scalars(actor)
            assert (actor_colors == colors).all(), "actor_colors should be the same as colors"
            assert actor.name == actor_name, "actor's name should equal to actor_name"
            self.mesh_store.mesh_actors[actor_name] = actor
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Cannot set the selected color", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def set_color(self, color, actor_name):
        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_store.mesh_actors[actor_name])
        mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        mesh = self.plotter.add_mesh(mesh_data, color=color, opacity=self.mesh_store.mesh_opacity[actor_name], name=actor_name)
        transformation_matrix = pv.array_from_vtkmatrix(self.mesh_store.mesh_actors[actor_name].GetMatrix())
        mesh.user_matrix = transformation_matrix
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=actor_name)
        assert actor.name == actor_name, "actor's name should equal to actor_name"
        self.mesh_store.mesh_actors[actor_name] = actor
        
    def nocs_epnp(self, color_mask, mesh):
        vertices = mesh.vertices
        pts3d, pts2d = utils.create_2d_3d_pairs(color_mask, vertices)
        pts2d = pts2d.astype('float32')
        pts3d = pts3d.astype('float32')
        camera_intrinsics = self.camera_store.camera_intrinsics.astype('float32')
        predicted_pose = utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics, self.camera_store.camera.position)
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

        lat = np.array(self.mesh_store.latlon[..., 0])
        lon = np.array(self.mesh_store.latlon[..., 1])
        lonf = lon[mesh.faces]
        msk = (np.sum(lonf>=0, axis=1)==3) & (np.sum(lat[mesh.faces]>=0, axis=1)==3)
        for i in range(len(pts2d)):
            pt = utils.latLon2xyz(mesh, lat, lonf, msk, gx[i], gy[i])
            pts3d.append(pt)
       
        pts3d = np.array(pts3d).reshape((len(pts3d), 3))

        pts2d = pts2d.astype('float32')
        pts3d = pts3d.astype('float32')
        camera_intrinsics = self.camera_store.camera_intrinsics.astype('float32')
        
        predicted_pose = utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics, self.camera_store.camera.position)

        return predicted_pose

    def epnp_mesh(self):
        if len(self.mesh_store.mesh_actors) == 1: self.mesh_store.reference = list(self.mesh_store.mesh_actors.keys())[0]
        if self.mesh_store.reference:
            colors = utils.get_mesh_actor_scalars(self.mesh_store.mesh_actors[self.mesh_store.reference])
            if colors is not None and (not np.all(colors == colors[0])):
                color_mask = self.export_mesh_render(save_render=False)
                gt_pose = self.mesh_store.mesh_actors[self.mesh_store.reference].user_matrix
                if self.mirror_x: gt_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                if self.mirror_y: gt_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose

                if np.sum(color_mask):
                    if self.mesh_store.mesh_colors[self.mesh_store.reference] == 'nocs':
                        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_store.mesh_actors[self.mesh_store.reference])
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
            mask_data = self.mask_store.render_mask(camera=self.camera_store.camera.copy())
            if np.max(mask_data) > 1: mask_data = mask_data / 255

            # current shown mask is binary mask
            if np.all(np.logical_or(mask_data == 0, mask_data == 1)):
                if len(self.mesh_store.mesh_actors) == 1: 
                    self.mesh_store.reference = list(self.mesh_store.mesh_actors.keys())[0]
                if self.mesh_store.reference:
                    colors = utils.get_mesh_actor_scalars(self.mesh_store.mesh_actors[self.mesh_store.reference])
                    if colors is not None and (not np.all(colors == colors[0])):
                        color_mask = self.export_mesh_render(save_render=False)
                        nocs_color = (self.mesh_store.mesh_colors[self.mesh_store.reference] == 'nocs')
                        gt_pose = self.mesh_store.mesh_actors[self.mesh_store.reference].user_matrix
                        if self.mirror_x: gt_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                        if self.mirror_y: gt_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_store.mesh_actors[self.mesh_store.reference])
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
            if actor_name in self.mesh_store.mesh_actors:
                popup = PopUpDialog(self, on_button_click=lambda text: self.update_color_button_text(text, popup))
                button_position = self.color_button.mapToGlobal(QPoint(0, 0))
                popup.move(button_position + QPoint(self.color_button.width(), 0))
                popup.exec_()

                text = self.color_button.text()
                self.mesh_store.mesh_colors[actor_name] = text
                if text == 'nocs': self.set_scalar(True, actor_name)
                elif text == 'latlon': self.set_scalar(False, actor_name)
                else: self.set_color(text, actor_name)
            else:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Only be able to color mesh actors", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def load_per_frame_info(self):
        video_frame = self.video_store.load_per_frame_info()
        if video_frame is not None: 
            self.add_image(video_frame)
            return video_frame
        else: return None
                
    def save_frame(self):
        if self.video_store.video_path:
            video_frame = self.load_per_frame_info()
            if video_frame is not None:
                os.makedirs(pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D", exist_ok=True)
                os.makedirs(pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "frames", exist_ok=True)
                output_frame_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "frames" / f"frame_{self.video_store.current_frame}.png"
                save_frame = PIL.Image.fromarray(video_frame)
                
                # save each frame
                save_frame.save(output_frame_path)
                self.output_text.append(f"-> Save frame {self.video_store.current_frame}: ({self.video_store.current_frame}/{self.video_store.total_frame}) to <span style='background-color:yellow; color:black;'>{str(output_frame_path)}</span>")
                self.image_store.image_path = str(output_frame_path)

                # save gt_pose for each frame
                os.makedirs(pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses", exist_ok=True)
                output_pose_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses" / f"pose_{self.video_store.current_frame}.npy"
                self.current_pose()
                np.save(output_pose_path, self.mesh_store.transformation_matrix)
                self.output_text.append(f"-> Save frame {self.video_store.current_frame} pose to <span style='background-color:yellow; color:black;'>{str(output_pose_path)}</span>:")
                self.output_text.append(f"{self.mesh_store.transformation_matrix}")
        elif self.folder_store.folder_path:
            # save gt_pose for specific frame
            os.makedirs(pathlib.Path(self.folder_store.folder_path) / "vision6D", exist_ok=True)
            os.makedirs(pathlib.Path(self.folder_store.folder_path) / "vision6D" / "poses", exist_ok=True)
            output_pose_path = pathlib.Path(self.folder_store.folder_path) / "vision6D" / "poses" / f"{pathlib.Path(self.mesh_store.pose_path).stem}.npy"
            self.current_pose()
            np.save(output_pose_path, self.mesh_store.transformation_matrix)
            self.output_text.append(f"-> Save frame {pathlib.Path(self.mesh_store.pose_path).stem} pose to <span style='background-color:yellow; color:black;'>{str(output_pose_path)}</span>:")
            self.output_text.append(f"{self.mesh_store.transformation_matrix}")
        else: 
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or a folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def sample_video(self):
        if self.video_store.video_path: 
            self.video_store.sample_video()
        else: 
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def play_video(self):
        if self.video_store.video_path:
            self.video_store.play_video()
            self.output_text.append(f"-> Current frame is ({self.video_store.current_frame}/{self.video_store.total_frame})")
            self.play_video_button.setText(f"Play ({self.video_store.current_frame}/{self.video_store.total_frame})")
            self.load_per_frame_info()
        else: QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
    
    def prev_frame(self):
        if self.video_store.video_path:
            self.video_store.prev_frame()
            self.output_text.append(f"-> Current frame is ({self.video_store.current_frame}/{self.video_store.total_frame})")
            pose_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses" / f"pose_{self.video_store.current_frame}.npy"
            if os.path.isfile(pose_path): 
                self.mesh_store.transformation_matrix = np.load(pose_path)
                self.register_pose(self.mesh_store.transformation_matrix)
                self.output_text.append(f"-> Load saved frame {self.video_store.current_frame} pose: \n{self.mesh_store.transformation_matrix}")
            else: 
                self.output_text.append(f"-> No saved pose for frame {self.video_store.current_frame}")
            self.play_video_button.setText(f"Play ({self.video_store.current_frame}/{self.video_store.total_frame})")
            self.load_per_frame_info()
        elif self.folder_store.folder_path:
            self.folder_store.prev_frame()
            self.play_video_button.setText(f"Frame ({self.folder_store.current_frame}/{self.folder_store.total_frame})")
            self.add_folder(self.folder_store.folder_path)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def next_frame(self):
        if self.video_store.video_path:
            self.save_frame()
            self.video_store.next_frame()
            self.output_text.append(f"-> Current frame is ({self.video_store.current_frame}/{self.video_store.total_frame})")
            # load pose for the current frame if the pose exist
            pose_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses" / f"pose_{self.video_store.current_frame}.npy"
            if os.path.isfile(pose_path): 
                self.mesh_store.transformation_matrix = np.load(pose_path)
                self.register_pose(self.mesh_store.transformation_matrix)
                self.output_text.append(f"-> Load saved frame {self.video_store.current_frame} pose: \n{self.mesh_store.transformation_matrix}")
            self.play_video_button.setText(f"Play ({self.video_store.current_frame}/{self.video_store.total_frame})")
            self.load_per_frame_info()
        elif self.folder_store.folder_path:
            self.folder_store.next_frame()
            self.play_video_button.setText(f"Frame ({self.folder_store.current_frame}/{self.folder_store.total_frame})")
            self.add_folder(self.folder_store.folder_path)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def draw_mask(self):
        def handle_output_path_change(output_path):
            if output_path:
                self.mask_store.mask_path = output_path
                self.add_mask(self.mask_store.mask_path)
        if self.image_store.image_path:
            self.label_window = LabelWindow(self.image_store.image_path)
            self.label_window.show()
            self.label_window.image_label.output_path_changed.connect(handle_output_path_change)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

