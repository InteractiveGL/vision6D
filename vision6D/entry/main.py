"""
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: main.py
@time: 2023-07-03 20:29
@desc: the entry to run the program with splash screen
"""

import sys
import platform
import os
import pathlib
from qtpy import QtWidgets, QtCore, QtGui
from time import sleep

from .. import MyMainWindow

if platform.system() == "Windows":
    os.environ["QT_QPA_PLATFORM"] = "windows:fontengine=freetype"

CWD = pathlib.Path(os.path.abspath(__file__)).parent

def main():
    app = QtWidgets.QApplication(sys.argv)

    # Set application stylesheet
    with open(CWD.parent / "data" / "style.qss", "r") as f:
        app.setStyleSheet(f.read())

    # Load splash image
    splash_pix = QtGui.QPixmap(str(CWD.parent / "data" / "icons" / "logo.png"))

    # Create splash screen
    splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)

    # Add progress bar
    progress_bar = QtWidgets.QProgressBar(splash)
    progress_bar.setMaximum(100)
    progress_bar.setGeometry(
        10,
        splash_pix.height() - 30,
        splash_pix.width() - 20,
        20
    )
    progress_bar.setAlignment(QtCore.Qt.AlignCenter)
    progress_bar.setTextVisible(True)

    splash.show()
    app.processEvents()

    # Simulate loading (Replace this with actual initialization tasks)
    for i in range(0, 101, 5):
        progress_bar.setValue(i)
        sleep(0.05)  # Adjust or remove based on actual tasks
        app.processEvents()

    # Load main window after splash
    window = MyMainWindow()
    window.showMaximized()

    # Close splash after main window is ready
    splash.finish(window)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
