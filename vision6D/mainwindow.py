# General import
import numpy as np
import functools
import trimesh
import json
import copy

# Qt5 import
from PyQt5 import QtWidgets, QtGui
from pyvistaqt import QtInteractor, MainWindow
from PyQt5.QtCore import Qt

# self defined package import
import vision6D as vis

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

        # Initialize file paths
        self.video_path = None
        self.image_path = None
        self.mask_path = None
        self.mesh_path = None
        self.pose_path = None

        # Dialogs to record users input info
        self.input_dialog = QtWidgets.QInputDialog()
        self.file_dialog = QtWidgets.QFileDialog()
        self.get_text_dialog = vis.GetTextDialog()

        # Set panel bar
        self.set_panel_bar()
        
        # Set menu bar
        self.set_menu_bars()

        # Create the plotter
        self.create_plotter()

        # Set up the main layout with the left panel and the render window using QSplitter
        self.main_layout = QtWidgets.QHBoxLayout(self.main_widget)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.panel_widget)
        self.splitter.addWidget(self.plotter)
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

    def showMaximized(self):
        super(MyMainWindow, self).showMaximized()
        self.splitter.setSizes([int(self.width() * 0.05), int(self.width() * 0.95)])

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
            self.hintLabel.hide()  # Hide hint when dragging
        else:
            e.ignore()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            file_path = url.toLocalFile()
            # Load mesh file
            if file_path.endswith(('.mesh', '.ply', '.stl', '.obj', '.off', '.dae', '.fbx', '.3ds', '.x3d')):  # add mesh
                self.mesh_path = file_path
                self.add_mesh_file(prompt=False)
            # Load video file
            elif file_path.endswith(('.avi', '.mp4', '.mkv', '.mov', '.fly', '.wmv', '.mpeg', '.asf', '.webm')):
                self.video_path = file_path
                self.add_video_file(prompt=False)
            # Load image/mask file
            elif file_path.endswith(('.png', '.jpg', 'jpeg', '.tiff', '.bmp', '.webp', '.ico')):  # add image/mask
                yes_no_box = vis.YesNoBox()
                yes_no_box.setIcon(QtWidgets.QMessageBox.Question)
                yes_no_box.setWindowTitle("Vision6D")
                yes_no_box.setText("Do you want to load the image as mask?")
                yes_no_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                button_clicked = yes_no_box.exec_()
                if not yes_no_box.canceled:
                    if button_clicked == QtWidgets.QMessageBox.Yes:
                        self.mask_path = file_path
                        self.add_mask_file(prompt=False)
                    elif button_clicked == QtWidgets.QMessageBox.No:
                        self.image_path = file_path
                        self.add_image_file(prompt=False)
            elif file_path.endswith('.npy'):
                self.pose_path = file_path
                self.add_pose_file()
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
        fileMenu.addAction('Add Workspace', self.add_workspace)
        fileMenu.addAction('Add Video', self.add_video_file)
        fileMenu.addAction('Add Image', self.add_image_file)
        fileMenu.addAction('Add Mask', self.add_mask_file)
        fileMenu.addAction('Draw Mask', self.draw_mask)
        fileMenu.addAction('Add Mesh', self.add_mesh_file)
        fileMenu.addAction('Clear', self.clear_plot)

        # allow to export files
        exportMenu = mainMenu.addMenu('Export')
        exportMenu.addAction('Image', self.export_image)
        exportMenu.addAction('Mask', self.export_mask)
        exportMenu.addAction('Pose', self.export_pose)
        exportMenu.addAction('Mesh Render', self.export_mesh_render)
        exportMenu.addAction('SegMesh Render', self.export_segmesh_render)
        
        # Add video related actions
        VideoMenu = mainMenu.addMenu('Video')
        VideoMenu.addAction('Play', self.play_video)
        VideoMenu.addAction('Sample', self.sample_video)
        VideoMenu.addAction('Delete', self.delete_video)
        save_per_frame_info = functools.partial(self.load_per_frame_info, save=True)
        VideoMenu.addAction('Save Frame', save_per_frame_info)
        VideoMenu.addAction('Prev Frame', self.prev_frame)
        VideoMenu.addAction('Next Frame', self.next_frame)
                
        # Add camera related actions
        CameraMenu = mainMenu.addMenu('Camera')
        CameraMenu.addAction('Calibrate', self.camera_calibrate)
        CameraMenu.addAction('Reset Camera (d)', self.reset_camera)
        CameraMenu.addAction('Zoom In (x)', self.zoom_in)
        CameraMenu.addAction('Zoom Out (z)', self.zoom_out)

        # add mirror actors related actions
        mirrorMenu = mainMenu.addMenu('Mirror')
        mirror_x = functools.partial(self.mirror_actors, direction='x')
        mirrorMenu.addAction('Mirror X axis', mirror_x)
        mirror_y = functools.partial(self.mirror_actors, direction='y')
        mirrorMenu.addAction('Mirror Y axis', mirror_y)
        
        # Add register related actions
        RegisterMenu = mainMenu.addMenu('Register')
        RegisterMenu.addAction('Reset GT Pose (k)', self.reset_gt_pose)
        RegisterMenu.addAction('Update GT Pose (l)', self.update_gt_pose)
        RegisterMenu.addAction('Current Pose (t)', self.current_pose)
        RegisterMenu.addAction('Undo Pose (s)', self.undo_pose)

        # Add pnp algorithm related actions
        PnPMenu = mainMenu.addMenu('PnP')
        PnPMenu.addAction('EPnP with mesh', self.epnp_mesh)
        epnp_nocs_mask = functools.partial(self.epnp_mask, True)
        PnPMenu.addAction('EPnP with nocs mask', epnp_nocs_mask)
        epnp_latlon_mask = functools.partial(self.epnp_mask, False)
        PnPMenu.addAction('EPnP with latlon mask', epnp_latlon_mask)

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
        self.save_frame_button.clicked.connect(lambda _, save=True: self.load_per_frame_info(save))
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

        # Create the color dropdown menu (comboBox)
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

        # Create the spacing button (comboBox)
        self.spacing_button = QtWidgets.QPushButton("Spacing")
        self.spacing_button.clicked.connect(self.set_spacing)
        actor_grid_layout.addWidget(self.spacing_button, 0, 2)

        # Create the hide button
        hide_button = QtWidgets.QPushButton("toggle meshes")
        hide_button.clicked.connect(self.toggle_hide_meshes_button)
        actor_grid_layout.addWidget(hide_button, 0, 3)

        # Create the remove button
        remove_button = QtWidgets.QPushButton("Remove Actor")
        remove_button.clicked.connect(self.remove_actors_button)
        actor_grid_layout.addWidget(remove_button, 0, 4)
        display_layout.addLayout(actor_grid_layout)

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
        self.output_text.setReadOnly(False)
        # Access to the system clipboard
        self.clipboard = QtGui.QGuiApplication.clipboard()
        output_layout.addWidget(self.output_text)
        self.output.setLayout(output_layout)
        self.panel_layout.addWidget(self.output)

    #^ Plot
    def create_plotter(self):
        self.frame = QtWidgets.QFrame()
        self.frame.setFixedSize(*self.window_size)
        self.plotter = QtInteractor(self.frame)
        # self.plotter.setFixedSize(*self.window_size)
        self.signal_close.connect(self.plotter.close)

    def show_plot(self):
        self.plotter.enable_joystick_actor_style()
        self.plotter.enable_trackball_actor_style()
        self.plotter.iren.interactor.AddObserver("LeftButtonPressEvent", self.pick_callback)

        self.plotter.add_axes()
        self.plotter.add_camera_orientation_widget()

        self.plotter.show()
        self.show()
