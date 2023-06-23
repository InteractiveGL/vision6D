# General import
# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"
from functools import partial

import numpy as np
import PIL.Image

# Qt5 import
from PyQt5 import QtWidgets, QtGui
from pyvistaqt import MainWindow

# self defined package import
from .stores import PvQtStore
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
        window_size = (1920, 1080)
        self.plot_store = PlotStore(frame, window_size)
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
        self.main_layout.addWidget(self.splitter)

        # Show the plotter
        self.plot_store.show_plot()
        self.signal_close.connect(self.plot_store.plotter.close)
        self.show()

        # Add shortcut to the right, left, space buttons
        QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self).activated.connect(self.video_folder_menu.next_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self).activated.connect(self.video_folder_menu.prev_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self).activated.connect(self.video_folder_menu.play_video)

        QtWidgets.QShortcut(QtGui.QKeySequence("k"), self).activated.connect(self.register_menu.reset_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("l"), self).activated.connect(self.register_menu.update_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("t"), self).activated.connect(self.register_menu.current_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("s"), self).activated.connect(self.register_menu.undo_pose)

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
        self.panel_display = DisplayPanel(self.display)

        self.pvqt_store = PvQtStore(button_group_actors_names=self.button_group_actors_names)
        
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
        fileMenu.addAction('Add Folder', partial(self.file_menu.add_folder, prompt=True))
        fileMenu.addAction('Add Video', partial(self.file_menu.add_video, prompt=True))
        fileMenu.addAction('Add Image', partial(self.file_menu.add_image, prompt=True))
        fileMenu.addAction('Add Mask', partial(self.file_menu.add_mask, prompt=True))
        fileMenu.addAction('Draw Mask', self.file_menu.draw_mask)
        fileMenu.addAction('Add Mesh', partial(self.file_menu.add_mesh, prompt=True))
        fileMenu.addAction('Clear', self.file_menu.clear_plot)

        # allow to export files
        exportMenu = mainMenu.addMenu('Export')
        exportMenu.addAction('Image', self.export_menu.export_image)
        exportMenu.addAction('Mask', self.export_menu.export_mask)
        exportMenu.addAction('Pose', self.export_menu.export_pose)
        exportMenu.addAction('Mesh Render', self.export_menu.export_mesh_render)
        exportMenu.addAction('SegMesh Render', self.export_menu.export_segmesh_render)
        
        # Add video related actions
        VideoMenu = mainMenu.addMenu('Video/Folder')
        VideoMenu.addAction('Play', self.video_folder_menu.play_video)
        VideoMenu.addAction('Sample', self.video_folder_menu.sample_video)
        VideoMenu.addAction('Delete', self.video_folder_menu.delete)
        VideoMenu.addAction('Save Frame', partial(self.video_folder_menu.save_frame))
        VideoMenu.addAction('Prev Frame', self.video_folder_menu.prev_frame)
        VideoMenu.addAction('Next Frame', self.video_folder_menu.next_frame)
                
        # Add camera related actions
        CameraMenu = mainMenu.addMenu('Camera')
        CameraMenu.addAction('Calibrate', self.camera_menu.calibrate)
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
        RegisterMenu.addAction('Current Pose (t)', self.register_menu.current_pose)
        RegisterMenu.addAction('Undo Pose (s)', self.register_menu.undo_pose)

        # Add pnp algorithm related actions
        PnPMenu = mainMenu.addMenu('PnP')
        PnPMenu.addAction('EPnP with mesh', self.pnp_menu.epnp_mesh)
        epnp_nocs_mask = partial(self.pnp_menu.epnp_mask, True)
        PnPMenu.addAction('EPnP with nocs mask', epnp_nocs_mask)
        epnp_latlon_mask = partial(self.pnp_menu.epnp_mask, False)
        PnPMenu.addAction('EPnP with latlon mask', epnp_latlon_mask)

    def showMaximized(self):
        super().showMaximized()
        self.splitter.setSizes([int(self.width() * 0.05), int(self.width() * 0.95)])

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
                self.folder_store.add_folder(file_path)
            else:
                # Load workspace json file
                if file_path.endswith(('.json')): self.workspace_store.add_workspace(file_path)
                # Load mesh file
                elif file_path.endswith(('.mesh', '.ply', '.stl', '.obj', '.off', '.dae', '.fbx', '.3ds', '.x3d')):
                    self.mesh_store.add_mesh(file_path)
                # Load video file
                elif file_path.endswith(('.avi', '.mp4', '.mkv', '.mov', '.fly', '.wmv', '.mpeg', '.asf', '.webm')):
                    self.folder_store.folder_path = None
                    self.video_store.add_video(file_path)
                # Load image/mask file
                elif file_path.endswith(('.png', '.jpg', 'jpeg', '.tiff', '.bmp', '.webp', '.ico')):  # add image/mask
                    file_data = np.array(PIL.Image.open(file_path).convert('L'), dtype='uint8')
                    unique, counts = np.unique(file_data, return_counts=True)
                    digit_counts = dict(zip(unique, counts))
                    # binary/grey mask file
                    if digit_counts[0] == np.max(counts) or digit_counts[0] == np.partition(counts, -2)[-2]: self.mask_store.add_mask(file_path) # 0 is the most or second most among all numbers   
                    # image file
                    else: self.image_store.add_image(file_path)
                elif file_path.endswith('.npy'): self.mesh_store.add_pose(file_path)
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