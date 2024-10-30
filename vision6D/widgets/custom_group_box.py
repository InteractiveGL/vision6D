from PyQt5 import QtWidgets, QtGui
from pyvistaqt import MainWindow
from PyQt5.QtCore import Qt, pyqtSignal

class CustomGroupBox(QtWidgets.QWidget):
    addButtonClicked = pyqtSignal()
    def __init__(self, title, parent=None):
        super().__init__(parent)

        # Main layout for the custom group box
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Header layout containing the title and the "+" button
        header_widget = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 5, 10, 5)

        # Optional: Add a checkbox if you want the group to be checkable
        self.checkbox = QtWidgets.QCheckBox()
        self.checkbox.setChecked(True)
        self.checkbox.toggled.connect(self.on_checkbox_toggled)  # Connect checkbox signal
        header_layout.addWidget(self.checkbox)

        # Title label
        title_label = QtWidgets.QLabel(title)
        title_font = QtGui.QFont()
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)

        # Spacer to push the "+" button to the right
        header_layout.addStretch()

        # "+" Button
        add_button = QtWidgets.QPushButton("+")
        add_button.setFixedSize(20, 20)
        add_button.clicked.connect(self.on_add_button_clicked)
        header_layout.addWidget(add_button)

        # Add the header to the main layout
        main_layout.addWidget(header_widget)

        # Content area
        self.content_widget = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 0, 10, 10)
        main_layout.addWidget(self.content_widget)

        # Initialize the content
        self.init_content()

    def init_content(self):
        # Add your content widgets here
        self.display_layout = QtWidgets.QVBoxLayout()
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.display_layout.addWidget(self.scroll_area)
        custom_widget_container = QtWidgets.QWidget()
        self.widget_layout = QtWidgets.QVBoxLayout()
        self.widget_layout.setSpacing(0)
        custom_widget_container.setLayout(self.widget_layout)
        self.widget_layout.addStretch()
        self.scroll_area.setWidget(custom_widget_container)
        self.content_layout.addLayout(self.display_layout)

    def on_checkbox_toggled(self):
        self.content_widget.setVisible(self.checkbox.isChecked())
        
    def on_add_button_clicked(self):
        self.addButtonClicked.emit()