# General import
import numpy as np
import PIL.Image
import os

# Qt5 import
from PyQt5 import QtWidgets
from pyvistaqt import MainWindow
from PyQt5.QtCore import Qt

# self defined package import
from .stores import MainStore
from .components import Panel, Menu


np.set_printoptions(suppress=True)

class AppWindow(MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        
        # Set up the main window layout
        self.setWindowTitle("Vision6D")
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        self.setAcceptDrops(True)

        # Create stores before anything else
        self.main_store = MainStore(self)

        # Delegate roles and tasks
        self.panel_widget = QtWidgets.QWidget()
        self.panel = Panel(self.panel_widget)
        self.setMenuBar(self.panel.panel_bar)
        self.menu = Menu(self.menuBar())

        # Set up the main layout with the left panel and the render window using QSplitter
        self.main_layout = QtWidgets.QHBoxLayout(self.main_widget)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.panel_widget)
        self.splitter.addWidget(self.main_store.pvqt_store.plot_store.plotter)
        self.main_layout.addWidget(self.splitter)

        # Show the plotter
        self.main_store.pvqt_store.plot_store.show_plot()
        self.show()

    def showMaximized(self):
        super().showMaximized()
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
            if os.path.isdir(file_path):
                self.folder_path = file_path
                self.add_folder()
            else:
                # Load workspace json file
                if file_path.endswith(('.json')):
                    self.workspace_path = file_path
                    self.menu.add_workspace()
                # Load mesh file
                elif file_path.endswith(('.mesh', '.ply', '.stl', '.obj', '.off', '.dae', '.fbx', '.3ds', '.x3d')):
                    self.mesh_path = file_path
                    self.add_mesh_file()
                # Load video file
                elif file_path.endswith(('.avi', '.mp4', '.mkv', '.mov', '.fly', '.wmv', '.mpeg', '.asf', '.webm')):
                    self.video_path = file_path
                    self.add_video_file()
                # Load image/mask file
                elif file_path.endswith(('.png', '.jpg', 'jpeg', '.tiff', '.bmp', '.webp', '.ico')):  # add image/mask
                    file_data = np.array(PIL.Image.open(file_path).convert('L'), dtype='uint8')
                    unique, counts = np.unique(file_data, return_counts=True)
                    digit_counts = dict(zip(unique, counts))
                    # can only load binary/grey mask now
                    if digit_counts[0] == np.max(counts) or digit_counts[0] == np.partition(counts, -2)[-2]: # 0 is the most or second most among all numbers
                        self.mask_path = file_path
                        self.add_mask_file()
                    # image file
                    else:
                        self.image_path = file_path
                        self.add_image_file()
                elif file_path.endswith('.npy'):
                    self.pose_path = file_path
                    self.add_pose_file()
                else:
                    QtWidgets.QMessageBox.warning(self, 'vision6D', "File format is not supported!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    return 0

    def resizeEvent(self, e):
        self.main_store.qt_store.resize()
        super().resizeEvent(e)