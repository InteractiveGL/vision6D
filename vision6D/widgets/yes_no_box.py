from PyQt5 import QtWidgets, QtGui

class YesNoBox(QtWidgets.QMessageBox):
    def __init__(self, *args, **kwargs):
        super(YesNoBox, self).__init__(*args, **kwargs)
        self.canceled = False
    def closeEvent(self, event: QtGui.QCloseEvent):
        self.canceled = True
        super(YesNoBox, self).closeEvent(event)