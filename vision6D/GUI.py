import sys
import pyvista as pv
from PyQt5 import QtWidgets, QtGui
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt
import vision6D as vis

def load_stylesheet(filename):
    with open(filename, "r") as file: return file.read()

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Set up the main window layout
        self.setWindowTitle("Vision6D")
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)

        # Create a left menu layout
        self.menu_widget = QtWidgets.QWidget()
        self.menu_layout = QtWidgets.QVBoxLayout(self.menu_widget)

        # Create a top menu bar with a toggle button
        self.menu_bar = QtWidgets.QMenuBar()
        self.toggle_action = QtWidgets.QAction("Menu", self)
        self.toggle_action.triggered.connect(self.toggle_menu)
        self.menu_bar.addAction(self.toggle_action)
        self.setMenuBar(self.menu_bar)

        # Create the sections for the left menu
        # Section 1
        section1 = QtWidgets.QGroupBox("Section 1")
        section1_layout = QtWidgets.QVBoxLayout()
        button1 = QtWidgets.QPushButton("Button 1")
        button2 = QtWidgets.QPushButton("Button 2")
        section1_layout.addWidget(button1)
        section1_layout.addWidget(button2)
        section1.setLayout(section1_layout)
        self.menu_layout.addWidget(section1)

        # Section 2
        section2 = QtWidgets.QGroupBox("Section 2")
        section2_layout = QtWidgets.QVBoxLayout()
        button3 = QtWidgets.QPushButton("Button 3")
        button4 = QtWidgets.QPushButton("Button 4")
        section2_layout.addWidget(button3)
        section2_layout.addWidget(button4)
        section2.setLayout(section2_layout)
        self.menu_layout.addWidget(section2)

        # Section 3 (Output Display)
        section3 = QtWidgets.QGroupBox("Output Display")
        section3_layout = QtWidgets.QVBoxLayout()

        # Create a scroll area for the buttons
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        section3_layout.addWidget(scroll_area)

        # Create a container widget for the buttons
        button_container = QtWidgets.QWidget()
        button_layout = QtWidgets.QVBoxLayout()
        button_layout.setSpacing(0)  # Remove spacing between buttons
        # button_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
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
            button.setFixedSize(section3.size().width(), 100)
            # button.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
            button_layout.addWidget(button)

        button_layout.addStretch()

        # Set the container widget as the scroll area's widget
        scroll_area.setWidget(button_container)

        section3.setLayout(section3_layout)
        self.menu_layout.addWidget(section3)

        # Set the stretch factor for each section to be equal
        self.menu_layout.setStretchFactor(section1, 1)
        self.menu_layout.setStretchFactor(section2, 1)
        self.menu_layout.setStretchFactor(section3, 1)

        # Create the PyVista render window
        self.plotter = QtInteractor()

        # Add a simple mesh to the plotter
        sphere = pv.Sphere()
        self.plotter.add_mesh(sphere)

        # Set up the main layout with the left menu and the render window using QSplitter
        self.main_layout = QtWidgets.QHBoxLayout(self.main_widget)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.menu_widget)
        self.splitter.addWidget(self.plotter)
        self.main_layout.addWidget(self.splitter)

    def toggle_menu(self):
        if self.menu_widget.isVisible():
            self.menu_widget.hide()
        else:
            self.menu_widget.show()

    def button_clicked(self, text):
        print(f"Button with text '{text}' clicked.")

    def showMaximized(self):
        super(MainWindow, self).showMaximized()
        self.splitter.setSizes([int(self.width() * 0.15), int(self.width() * 0.85)])

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    stylesheet_pth = vis.config.GITROOT / "vision6D" / "data" / "style.qss"
    stylesheet = load_stylesheet(stylesheet_pth)  # Replace with the path to your .qss file
    app.setStyleSheet(stylesheet)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
