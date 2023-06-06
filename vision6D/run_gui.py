import sys
import os
import pathlib
from qtpy import QtWidgets
from vision6D import Interface

CWD = pathlib.Path(os.path.abspath(__file__)).parent

def exe():
    app = QtWidgets.QApplication(sys.argv)
    with open(CWD / "data" / "style.qss", "r") as f: 
        app.setStyleSheet(f.read())

    window = Interface()
    window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    exe()
