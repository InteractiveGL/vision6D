
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, QtGui

from .singleton import Singleton

from .pvqt_store import PvQtStore
from .plot_store import PlotStore

class QtStore(metaclass=Singleton):

    def __init__(self, main_window):

        self.main_window = main_window

        # Keeping references
        self.plot_store = PlotStore()

        # Add a QLabel as an overlay hint label
        self.hintLabel = QtWidgets.QLabel(self.plot_store.plotter)
        self.hintLabel.setText("Drag and drop a file here...")
        self.hintLabel.setStyleSheet("""
                                    color: white; 
                                    background-color: rgba(0, 0, 0, 127); 
                                    padding: 10px;
                                    border: 2px dashed gray;
                                    """)
        self.hintLabel.setAlignment(Qt.AlignCenter)

        # Located in Display Panel
        self.button_group_actors_names = QtWidgets.QButtonGroup(self.main_window)
        self.button_layout = QtWidgets.QVBoxLayout()
        self.button_layout.setSpacing(0)  # Remove spacing between buttons
        self.button_layout.addStretch()
        self.color_button = QtWidgets.QPushButton("Color")

        # Located in Output Panel
        self.output_text = QtWidgets.QTextEdit()
        self.output_text.setReadOnly(False)
        self.clipboard = QtGui.QGuiApplication.clipboard()

    def remove_actor_button(self, name, button):

        # Mention the removal and remove button from group
        self.output_text.append(f"-> Remove actor: {name}")
        self.button_group_actors_names.removeButton(button)
        self.button_layout.removeWidget(button)
        button.deleteLater()

    def check_button(self, actor_name):
        for button in self.qt_store.button_group_actors_names.buttons():
            if button.text() == actor_name: 
                button.setChecked(True)
                self.button_actor_name_clicked(actor_name)
                break

    def resize(self):
        x = (self.plot_store.plotter.size().width() - self.hintLabel.width()) // 2
        y = (self.plot_store.plotter.size().height() - self.hintLabel.height()) // 2
        self.hintLabel.move(x, y)

    def reset(self):
        
        # Clear out everything in the remove menu
        for button in self.button_group_actors_names.buttons():
            name = button.text()
            if name == 'image': actor = PvQtStore().image_actor
            elif name == 'mask': actor = PvQtStore().mask_actor
            elif name in PvQtStore().mesh_actors: actor = PvQtStore().mesh_actors[name]
            self.plot_store.remove_actor(actor)
            self.remove_actor_button(name, actor)

        self.hintLabel.show()
        self.output_text.clear()
        self.color_button.setText("Color")
    