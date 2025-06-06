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

from qtpy import QtCore, QtWidgets
from vision6D import MyMainWindow, STYLES_FILE

if platform.system() == "Windows":
    os.environ["QT_QPA_PLATFORM"] = "windows:fontengine=freetype"

def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

    app = QtWidgets.QApplication(sys.argv)

    with open(STYLES_FILE, "r") as f:
        app.setStyleSheet(f.read())
        
    window = MyMainWindow()
    window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
