
from PyQt5 import QtWidgets

from ..widgets import CustomQtInteractor
from .singleton import Singleton

class PlotStore(metaclass=Singleton):

    plotter: CustomQtInteractor

    def __init__(self, signal_close):

        self.frame = QtWidgets.QFrame()
        self.window_size = (1920, 1080)
        self.frame.setFixedSize(*self.window_size)
        self.plotter = CustomQtInteractor(self.frame)
        signal_close.connect(self.plotter.close)

    def show_plot(self):
        self.plotter.enable_joystick_actor_style()
        self.plotter.enable_trackball_actor_style()
        
        self.plotter.add_axes()
        self.plotter.add_camera_orientation_widget()

        self.plotter.show()