import sys
import pyvista as pv
from PyQt5 import QtWidgets, QtGui
from pyvistaqt import QtInteractor, MainWindow
from PyQt5.QtCore import Qt

import vision6D as vis

class Interactor():pass

class GUI(MainWindow):
    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)

        # Set up the main window layout
        self.setWindowTitle("Vision6D")
        self.window_size = (1920, 1080)
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)

        # Create a left menu layout
        self.menu_widget = QtWidgets.QWidget()
        self.menu_layout = QtWidgets.QVBoxLayout(self.menu_widget)

        # Create a top menu bar with a toggle button
        self.menu_bar = QtWidgets.QMenuBar()
        self.toggle_action = QtWidgets.QAction("Panel", self)
        self.toggle_action.triggered.connect(self.toggle_menu)
        self.menu_bar.addAction(self.toggle_action)
        self.setMenuBar(self.menu_bar)

        self.menu_display()
        self.menu_output()
        
        # Set the stretch factor for each section to be equal
        self.menu_layout.setStretchFactor(self.display, 1)
        self.menu_layout.setStretchFactor(self.output, 1)

        # Create the plotter
        self.create_plotter()

        # Set up the main layout with the left menu and the render window using QSplitter
        self.main_layout = QtWidgets.QHBoxLayout(self.main_widget)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.menu_widget)
        self.splitter.addWidget(self.plotter)
        self.main_layout.addWidget(self.splitter)

        if show:
            self.plotter.enable_joystick_actor_style()
            self.plotter.enable_trackball_actor_style()

            self.plotter.add_axes()
            self.plotter.add_camera_orientation_widget()

            self.plotter.show()
            self.show()

    def toggle_menu(self):
        if self.menu_widget.isVisible():
            self.menu_widget.hide()
        else:
            self.menu_widget.show()
        
    def menu_display(self):
        self.display = QtWidgets.QGroupBox("Actors")
        display_layout = QtWidgets.QVBoxLayout()
        display_layout.setContentsMargins(10, 20, 10, 10)

        # Create a scroll area for the buttons
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        display_layout.addWidget(scroll_area)

        # Create a container widget for the buttons
        button_container = QtWidgets.QWidget()
        button_layout = QtWidgets.QVBoxLayout()
        button_layout.setSpacing(0)  # Remove spacing between buttons
        # button_layout.setContentsMargins(10, 0, 0, 0)  # Remove margins
        button_container.setLayout(button_layout)

        # List of input strings (one for each button)
        input_strings = [
            "Input 1",
            "Input 2",
            "Input 3",
            "Input 4",
            "Input 5",
            "Input 1",
            "Input 2",
            "Input 3",
            "Input 4",
            "Input 5",
            "Input 1",
            "Input 2",
            "Input 3",
            "Input 4",
            "Input 5",
            "Input 1",
            "Input 2",
            "Input 3",
            "Input 4",
            "Input 5",
            "Input 1",
            "Input 2",
            "Input 3",
            "Input 4",
            "Input 5",
            "Input 1",
            "Input 2",
            "Input 3",
            "Input 4",
            "Input 5",
            "Input 1",
            "Input 2",
            "Input 3",
            "Input 4",
            "Input 5",
            "Input 1",
            "Input 2",
            "Input 3",
            "Input 4",
            "Input 5",
            # Add more input strings to see the scrollbar in action
        ]

        # Create buttons for each input string
        for input_string in input_strings:
            button = QtWidgets.QPushButton(input_string)
            button.clicked.connect(lambda checked, text=input_string: self.button_clicked(text))
            button.setFixedSize(self.display.size().width(), 100)
            button_layout.addWidget(button)

        button_layout.addStretch()

        # Set the container widget as the scroll area's widget
        scroll_area.setWidget(button_container)

        self.display.setLayout(display_layout)
        self.menu_layout.addWidget(self.display)

    def menu_output(self):
        # Add a spacer to the top of the main layout
        
        self.output = QtWidgets.QGroupBox("Logs")
        output_layout = QtWidgets.QVBoxLayout()
        output_layout.setContentsMargins(10, 20, 10, 10)

        self.output_display = QtWidgets.QTextEdit()
        self.output_display.setReadOnly(True)
        output_layout.addWidget(self.output_display)
        self.output.setLayout(output_layout)
        self.menu_layout.addWidget(self.output)
        self.output_display.append("This is the new content of the output display.")

    def button_clicked(self, text):
        print(f"Button with text '{text}' clicked.")
        self.update_plot(text)

    def update_plot(self, input_string):
        # Clear the existing actors in the plotter
        # self.plotter.clear()

        # Example: Add a mesh depending on the input string
        if input_string == "Input 1":
            mesh = pv.Cube()
        elif input_string == "Input 2":
            mesh = pv.Sphere()
        elif input_string == "Input 3":
            mesh = pv.Cone()
        elif input_string == "Input 4":
            mesh = pv.Cylinder()
        elif input_string == "Input 5":
            mesh = pv.Arrow()
        else:
            mesh = None

        # Add the new mesh to the plotter and render
        if mesh is not None:
            self.plotter.add_mesh(mesh, color='green')
            self.plotter.reset_camera()
            self.plotter.show()

    def showMaximized(self):
        super(GUI, self).showMaximized()
        self.splitter.setSizes([int(self.width() * 0.15), int(self.width() * 0.85)])

    def create_plotter(self):
        self.frame = QtWidgets.QFrame()
        self.plotter = QtInteractor(self.frame)
        # self.plotter.setFixedSize(*self.window_size) # but camera locate in the center instead of top left
        self.render = pv.Plotter(window_size=[self.window_size[0], self.window_size[1]], lighting=None, off_screen=True) 
        self.render.set_background('black'); 
        assert self.render.background_color == "black", "render's background need to be black"
        self.signal_close.connect(self.plotter.close)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    stylesheet_pth = vis.config.GITROOT / "vision6D" / "data" / "style.qss"
    with open(stylesheet_pth, "r") as file: stylesheet = file.read()  # Replace with the path to your .qss file
    app.setStyleSheet(stylesheet)
    window = GUI()
    window.showMaximized()
    sys.exit(app.exec_())
