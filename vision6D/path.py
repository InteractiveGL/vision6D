import os
import pathlib

# Constants
PKG_ROOT = pathlib.Path(os.path.abspath(__file__)).parent # vision6D
LATLON_PATH = PKG_ROOT / "data" / "ossiclesCoordinateMapping2.json"
ICON_PATH = PKG_ROOT / "data" / "icons"
MODEL_PATH = PKG_ROOT / "data" / "model"

# Global variables, make sure it is (width, height), just to be consistent with the vtk plotter
PLOT_SIZE = (1920, 1080)