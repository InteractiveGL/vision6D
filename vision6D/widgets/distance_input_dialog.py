from PyQt5.QtWidgets import QDialog, QLineEdit, QLabel, QPushButton, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt

class DistanceInputDialog(QDialog):
    def __init__(self, parent=None, title='', label='', value='', default_value=''):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.result = None
        self.default_value = default_value  # Store the default value for resetting

        # Create the label for the prompt
        self.prompt_label = QLabel(label)

        # Create the line edit for user input
        self.line_edit = QLineEdit()
        self.line_edit.setText(value)
        self.line_edit.setAlignment(Qt.AlignRight)

        # Create the unit label
        self.unit_label = QLabel('mm')

        # Create the Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_value)

        # Arrange the line edit, unit label, and Reset button horizontally
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.line_edit)
        input_layout.addWidget(self.unit_label)
        input_layout.addWidget(self.reset_button)

        # Create OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Arrange all widgets vertically
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.prompt_label)
        main_layout.addLayout(input_layout)
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

    def reset_value(self):
        # Reset the line edit to the default value
        self.line_edit.setText(self.default_value)

    def get_value(self):
        return self.line_edit.text()