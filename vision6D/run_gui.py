from vision6D import Interface_GUI
import sys
from qtpy import QtWidgets
import vision6D as vis
import pathlib
import os

GITROOT = pathlib.Path(os.path.abspath(__file__)).parent.parent

def exe():
    app = QtWidgets.QApplication(sys.argv)
    stylesheet_pth = GITROOT / "vision6D" / "style.qss"
    with open(stylesheet_pth, "r") as file: stylesheet = file.read()  # Replace with the path to your .qss file
    app.setStyleSheet(stylesheet)
    window = Interface_GUI()
    window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    exe()
