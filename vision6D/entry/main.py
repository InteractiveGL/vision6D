'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: main.py
@time: 2023-07-03 20:29
@desc: the entry to run the program
'''

import sys
import platform
import os
import pathlib
from qtpy import QtWidgets

from .. import MyMainWindow
if platform.system() == "Windows": os.environ["QT_QPA_PLATFORM"] = "windows:fontengine=freetype"

CWD = pathlib.Path(os.path.abspath(__file__)).parent

def main():
    app = QtWidgets.QApplication(sys.argv)
    with open(CWD.parent / "data" / "style.qss", "r") as f: app.setStyleSheet(f.read())
    window = MyMainWindow()
    window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
