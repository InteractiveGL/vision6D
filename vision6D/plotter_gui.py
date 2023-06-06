import pathlib
import numpy as np
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import PIL
import vtk

# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"

from PyQt5 import QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow
import vision6D as vis
from ..vision6D.GUI import MyMainWindow

np.set_printoptions(suppress=True)

class Interface(MyMainWindow):
    def __init__(self):
        super().__init__()
