from vision6D import Interface_GUI
import sys
from qtpy import QtWidgets
import vision6D as vis

def exe():
    app = QtWidgets.QApplication(sys.argv)
    stylesheet_pth = vis.config.GITROOT / "vision6D" / "data" / "style.qss"
    with open(stylesheet_pth, "r") as file: stylesheet = file.read()  # Replace with the path to your .qss file
    app.setStyleSheet(stylesheet)
    window = Interface_GUI()
    window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    exe()
