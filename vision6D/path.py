import os
import pathlib

# Constants
SAVE_ROOT = pathlib.Path.home() / "Documents" / "Vision6D" / "Output"
ICON_PATH = pathlib.Path(os.path.abspath(__file__)).parent / "data" / "icons"
MODEL_PATH = pathlib.Path(os.path.abspath(__file__)).parent / "data" / "model"