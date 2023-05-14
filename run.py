from vision6D import Interface
import sys
from qtpy import QtWidgets

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Interface()
    sys.exit(app.exec_())