# General import
# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"
from functools import partial

import numpy as np
import trimesh

# Qt5 import
from PyQt5 import QtWidgets, QtGui
from pyvistaqt import MainWindow

# self defined package import
from .stores import QtStore
from .stores import PlotStore
from .stores import ImageStore
from .stores import MaskStore
from .stores import MeshStore
from .stores import VideoStore
from .stores import FolderStore
from .stores import WorkspaceStore

# import panels
from .components.panel import DisplayPanel
from .components.panel import OutputPanel

# import menus
from .components.menu import CameraMenu
from .components.menu import ExportMenu
from .components.menu import FileMenu
from .components.menu import MirrorMenu
from .components.menu import PnPMenu
from .components.menu import RegisterMenu
from .components.menu import VideoFolderMenu

from .widgets import YesNoBox

np.set_printoptions(suppress=True)

class AppWindow(MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        
        # Set up the main window layout
        self.setWindowTitle("Vision6D")
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        self.setAcceptDrops(True)

        frame = QtWidgets.QFrame()
        # window_size = (QtWidgets.QApplication.desktop().screenGeometry().width(), 
        #             QtWidgets.QApplication.desktop().screenGeometry().height())
        self.plot_store = PlotStore(frame, window_size=(1920, 1080))
        self.qt_store = QtStore()
        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.mesh_store = MeshStore()
        self.video_store = VideoStore()
        self.folder_store = FolderStore()
        self.workspace_store = WorkspaceStore()
       
        # buttons
        self.button_group_actors_names = QtWidgets.QButtonGroup(self)

        # Delegate roles and tasks
        self.panel_widget = QtWidgets.QWidget()
        self.set_panel()
        self.setMenuBar(self.panel_bar)

        self.file_menu = FileMenu()
        self.export_menu = ExportMenu()
        self.video_folder_menu = VideoFolderMenu()
        self.camera_menu = CameraMenu()
        self.pnp_menu = PnPMenu()
        self.register_menu = RegisterMenu()
        self.mirror_menu = MirrorMenu()
        self.set_menu_bars()

        # Set up the main layout with the left panel and the render window using QSplitter
        self.main_layout = QtWidgets.QHBoxLayout(self.main_widget)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.panel_widget)
        self.splitter.addWidget(self.plot_store.plotter)
        self.splitter.setStretchFactor(0, 1) # for the self.panel_widget
        self.splitter.setStretchFactor(1, 3) # for the self.plot_store.plotter
        self.main_layout.addWidget(self.splitter)

        # Show the plotter
        self.plot_store.show_plot()
        self.signal_close.connect(self.plot_store.plotter.close)
        self.show()

        # Add shortcut to the right, left, space buttons
        QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self).activated.connect(self.panel_display.next_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self).activated.connect(self.panel_display.prev_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self).activated.connect(self.panel_display.play_video)

        QtWidgets.QShortcut(QtGui.QKeySequence("k"), self).activated.connect(self.register_menu.reset_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("l"), self).activated.connect(self.register_menu.update_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("s"), self).activated.connect(self.undo_pose)

        # change image opacity key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("b"), self).activated.connect(lambda up=True: self.panel_display.toggle_image_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("n"), self).activated.connect(lambda up=False: self.panel_display.toggle_image_opacity(up))

        # change mask opacity key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("g"), self).activated.connect(lambda up=True: self.panel_display.toggle_mask_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("h"), self).activated.connect(lambda up=False: self.panel_display.toggle_mask_opacity(up))

        # change mesh opacity key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("y"), self).activated.connect(lambda up=True: self.panel_display.toggle_surface_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("u"), self).activated.connect(lambda up=False: self.panel_display.toggle_surface_opacity(up))

        QtWidgets.QShortcut(QtGui.QKeySequence("c"), self).activated.connect(self.plot_store.reset_camera)
        QtWidgets.QShortcut(QtGui.QKeySequence("z"), self).activated.connect(self.plot_store.zoom_out)
        QtWidgets.QShortcut(QtGui.QKeySequence("x"), self).activated.connect(self.plot_store.zoom_in)

    def set_panel(self):
        # Save reference
        self.panel_widget = self.panel_widget

        # Create a left panel layout
        self.panel_layout = QtWidgets.QVBoxLayout(self.panel_widget)

        # Create a top panel bar with a toggle button
        self.panel_bar = QtWidgets.QMenuBar()
        self.toggle_action = QtWidgets.QAction("Panel", self)
        self.toggle_action.triggered.connect(self.toggle_panel)
        self.panel_bar.addAction(self.toggle_action)

        # Create the display
        self.display = QtWidgets.QGroupBox("Console")
        self.panel_display = DisplayPanel(self)

        # Create the output
        self.output = QtWidgets.QGroupBox("Output")
        self.panel_output = OutputPanel(self.output)

        # Set the layout
        self.panel_layout.addWidget(self.display)
        self.panel_layout.addWidget(self.output)
        
        # Set the stretch factor for each section to be equal
        self.panel_layout.setStretchFactor(self.display, 1)
        self.panel_layout.setStretchFactor(self.output, 1)

    def toggle_panel(self):
        if self.panel_widget.isVisible():
            # self.panel_widget width changes when the panel is visiable or hiden
            self.panel_widget_width = self.panel_widget.width()
            self.panel_widget.hide()
            x = (self.plot_store.plotter.size().width() + self.panel_widget_width - self.qt_store.hintLabel.width()) // 2
            y = (self.plot_store.plotter.size().height() - self.qt_store.hintLabel.height()) // 2
            self.qt_store.hintLabel.move(x, y)
        else:
            self.panel_widget.show()
            x = (self.plot_store.plotter.size().width() - self.panel_widget_width - self.qt_store.hintLabel.width()) // 2
            y = (self.plot_store.plotter.size().height() - self.qt_store.hintLabel.height()) // 2
            self.qt_store.hintLabel.move(x, y)

    def set_menu_bars(self):
        mainMenu = self.menuBar()
        
        # allow to add files
        fileMenu = mainMenu.addMenu('File')
        fileMenu.addAction('Add Workspace', partial(self.file_menu.add_workspace, prompt=True))
        fileMenu.addAction('Add Folder', partial(self.add_folder))
        fileMenu.addAction('Add Video', partial(self.file_menu.add_video, prompt=True))
        fileMenu.addAction('Add Image', partial(self.file_menu.add_image, prompt=True))
        fileMenu.addAction('Add Mask', partial(self.file_menu.add_mask, prompt=True))
        fileMenu.addAction('Add Mesh', partial(self.add_mesh, prompt=True))
        fileMenu.addAction('Draw Mask', self.panel_display.draw_mask)
        fileMenu.addAction('Clear', self.panel_display.clear_plot)

        # allow to export files
        exportMenu = mainMenu.addMenu('Export')
        exportMenu.addAction('Image', self.export_image)
        exportMenu.addAction('Mask', self.export_mask)
        exportMenu.addAction('Pose', self.export_pose)
        exportMenu.addAction('Mesh Render', self.export_mesh_render)
        exportMenu.addAction('SegMesh Render', self.export_segmesh_render)
        
        # Add video related actions
        VideoMenu = mainMenu.addMenu('Video/Folder')
        VideoMenu.addAction('Play', self.panel_display.play_video)
        VideoMenu.addAction('Sample', self.panel_display.sample_video)
        VideoMenu.addAction('Delete', self.video_folder_menu.delete)
        VideoMenu.addAction('Save Frame', partial(self.panel_display.save_frame))
        VideoMenu.addAction('Prev Frame', self.panel_display.prev_frame)
        VideoMenu.addAction('Next Frame', self.panel_display.next_frame)
                
        # Add camera related actions
        CameraMenu = mainMenu.addMenu('Camera')
        CameraMenu.addAction('Calibrate', self.calibrate)
        CameraMenu.addAction('Reset Camera (d)', self.plot_store.reset_camera)
        CameraMenu.addAction('Zoom In (x)', self.plot_store.zoom_in)
        CameraMenu.addAction('Zoom Out (z)', self.plot_store.zoom_out)

        # add mirror actors related actions
        mirrorMenu = mainMenu.addMenu('Mirror')
        mirror_x = partial(self.mirror_menu.mirror_actors, direction='x')
        mirrorMenu.addAction('Mirror X axis', mirror_x)
        mirror_y = partial(self.mirror_menu.mirror_actors, direction='y')
        mirrorMenu.addAction('Mirror Y axis', mirror_y)
        
        # Add register related actions
        RegisterMenu = mainMenu.addMenu('Register')
        RegisterMenu.addAction('Reset GT Pose (k)', self.register_menu.reset_gt_pose)
        RegisterMenu.addAction('Update GT Pose (l)', self.register_menu.update_gt_pose)
        RegisterMenu.addAction('Undo Pose (s)', self.undo_pose)

        # Add pnp algorithm related actions
        PnPMenu = mainMenu.addMenu('PnP')
        PnPMenu.addAction('EPnP with mesh', self.epnp_mesh)
        PnPMenu.addAction('EPnP with nocs mask', partial(self.epnp_mask, True))
        PnPMenu.addAction('EPnP with latlon mask', partial(self.epnp_mask, False))

    def showMaximized(self):
        super().showMaximized()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
            self.qt_store.hintLabel.hide()  # Hide hint when dragging
        else:
            e.ignore()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isdir(file_path):
                self.video_store.video_path = None
                self.file_menu.add_folder(folder_path=file_path)
            else:
                # Load workspace json file
                if file_path.endswith(('.json')): 
                    self.file_menu.add_workspace(workspace_path=file_path)
                # Load mesh file
                elif file_path.endswith(('.mesh', '.ply', '.stl', '.obj', '.off', '.dae', '.fbx', '.3ds', '.x3d')):
                    self.add_mesh(mesh_path=file_path)
                # Load video file
                elif file_path.endswith(('.avi', '.mp4', '.mkv', '.mov', '.fly', '.wmv', '.mpeg', '.asf', '.webm')):
                    self.folder_store.folder_path = None
                    self.file_menu.add_video(video_path=file_path)
                # Load image/mask file
                elif file_path.endswith(('.png', '.jpg', 'jpeg', '.tiff', '.bmp', '.webp', '.ico')):  # add image/mask
                    yes_no_box = YesNoBox()
                    yes_no_box.setIcon(QtWidgets.QMessageBox.Question)
                    yes_no_box.setWindowTitle("Vision6D")
                    yes_no_box.setText("Do you want to load the image as mask?")
                    yes_no_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                    button_clicked = yes_no_box.exec_()
                    if not yes_no_box.canceled:
                        if button_clicked == QtWidgets.QMessageBox.Yes: self.file_menu.add_mask(mask_path=file_path)
                        elif button_clicked == QtWidgets.QMessageBox.No: self.file_menu.add_image(image_path=file_path)
                elif file_path.endswith('.npy'): 
                    self.mesh_store.add_pose(file_path)
                else:
                    QtWidgets.QMessageBox.warning(self, 'vision6D', "File format is not supported!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    return 0
                
    def resize(self):
        x = (self.plot_store.plotter.size().width() - self.qt_store.hintLabel.width()) // 2
        y = (self.plot_store.plotter.size().height() - self.qt_store.hintLabel.height()) // 2
        self.qt_store.hintLabel.move(x, y)

    def resizeEvent(self, e):
        self.resize()
        super().resizeEvent(e)

    def add_folder(self):
        flag = self.file_menu.add_folder(prompt=True)
        if flag:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Not a valid folder, please reload a folder", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    # FileMenu
    def add_mesh(self, file_path='', prompt=False):
        flag = self.file_menu.add_mesh(file_path, prompt)
        if not flag:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a mesh first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    # Camera Menu
    def calibrate(self):
        msg = self.camera_menu.calibrate()
        if not msg:
            QtWidgets.QMessageBox.warning(self, 'vision6D', msg, QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    # Export Menu
    def export_image(self):
        if self.image_store.image_actor: self.export_menu.export_image()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def export_mask(self):
        if self.mask_store.mask_actor: self.export_menu.export_mask()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a mask first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
    def export_pose(self):
        if self.mesh_store.reference: 
            self.export_menu.export_pose()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
    def export_mesh_render(self):
        if self.mesh_store.reference: 
            self.export_menu.export_mesh_render()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
    def export_segmesh_render(self):
        if self.mesh_store.reference and self.mask_store.mask_actor:
            self.export_menu.export_segmesh_render()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a mesh or mask first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    # PnP Menu
    def epnp_mesh(self):
        msg = self.pnp_menu.epnp_mesh()
        if msg: QtWidgets.QMessageBox.warning(self, 'vision6D', msg, QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def epnp_mask(self, nocs_method):
        msg = self.pnp_menu.epnp_mask(nocs_method)
        if msg: QtWidgets.QMessageBox.warning(self, 'vision6D', msg, QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                
    # Register Menu
    def undo_pose(self):
        checked_button = self.display_panel.button_group_actors_names.checkedButton()
        if not checked_button:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Choose a mesh actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else:
            actor_name = checked_button.text()
            self.register_menu.undo_pose(actor_name)