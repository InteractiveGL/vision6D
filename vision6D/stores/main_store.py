
from .singleton import Singleton

from .qt_store import QtStore
from .paths_store import PathsStore
from .pvqt_store import PvQtStore
from .plot_store import PlotStore

class MainStore(metaclass=Singleton):

    def __init__(self, main_window):

        self.plot_store = PlotStore(main_window.signal_close)
        self.qt_store = QtStore(main_window)
        self.pvqt_store = PvQtStore()
        self.paths_store = PathsStore()

    def clear_plot(self):
        self.qt_store.reset()
        self.paths_store.reset()
        self.pvqt_store.reset()