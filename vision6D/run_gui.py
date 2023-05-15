from vision6D import Interface_GUI
from PyQt5.QtCore import QFile
import sys
from qtpy import QtWidgets

def exe():
    app = QtWidgets.QApplication(sys.argv)
    style_file = QFile("vision6D/style.qss")
    style_file.open(QFile.ReadOnly | QFile.Text)
    app.setStyleSheet(style_file.readAll().data().decode())
    window = Interface_GUI()
    window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    exe()
