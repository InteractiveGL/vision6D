import os
import pathlib

# Constants
PKG_ROOT = pathlib.Path(os.path.abspath(__file__)).parent # vision6D
LATLON_PATH = PKG_ROOT / "data" / "ossiclesCoordinateMapping2.json"
ICON_PATH = PKG_ROOT / "data" / "icons"