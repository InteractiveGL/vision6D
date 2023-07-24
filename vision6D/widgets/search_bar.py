from PyQt5 import QtWidgets, QtCore

class SearchBar(QtWidgets.QLineEdit):
    returnPressedSignal = QtCore.pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.returnPressedSignal.emit()
        else:
            super().keyPressEvent(event)