
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, QtGui

from .singleton import Singleton

from .pvqt import PvQtStore

class QtStore(metaclass=Singleton):

    def __init__(self, main_window):

        self.main_window = main_window

        # Keeping references
        self.pvqt_store = PvQtStore()

        # Add a QLabel as an overlay hint label
        self.hintLabel = QtWidgets.QLabel(self.pvqt_store.plot_store.plotter)
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

        self.play_video_button = QtWidgets.QPushButton("Play Video")
        self.play_video_button.clicked.connect(self.pvqt_store.video_store.play_video)

        # create opacity_spinbox
        self.opacity_spinbox = QtWidgets.QDoubleSpinBox()
        self.opacity_spinbox.setMinimum(0.0)
        self.opacity_spinbox.setMaximum(1.0)
        self.opacity_spinbox.setDecimals(2)
        self.opacity_spinbox.setSingleStep(0.05)
        self.ignore_spinbox_value_change = True
        self.opacity_spinbox.setValue(0.3)
        self.ignore_spinbox_value_change = False 
        self.opacity_spinbox.valueChanged.connect(self.opacity_value_change)

        # Located in Output Panel
        self.output_text = QtWidgets.QTextEdit()
        self.output_text.setReadOnly(False)
        self.clipboard = QtGui.QGuiApplication.clipboard()

    def opacity_value_change(self, value):
        if self.ignore_spinbox_value_change: return 0
        checked_button = self.qt_store.button_group_actors_names.checkedButton()
        if checked_button:
            actor_name = checked_button.text()
            if actor_name == 'image': 
                self.pvqt_store.image_store.set_image_opacity(value)
            elif actor_name == 'mask': 
                self.pvqt_store.mask_store.set_mask_opacity(value)
            elif actor_name in self.pvqt_store.mesh_stores.mesh_actors: 
                self.pvqt_store.mesh_store.set_mesh_opacity(actor_name, value)
        else:
            self.ignore_spinbox_value_change = True
            self.opacity_spinbox.setValue(value)
            self.ignore_spinbox_value_change = False
            return 0

    def button_actor_name_clicked(self, text):
        if text in self.pvqt_store.mesh_store.mesh_actors:
            # set the current mesh color
            self.color_button.setText(self.pvqt_store.mesh_store.mesh_colors[text])
            # set mesh reference
            self.pvqt_store.mesh_store.reference = text
            self.pvqt_store.camera_store.current_pose()
            curr_opacity = self.pvqt_store.mesh_store.mesh_actors[self.pvqt_store.mesh_store.reference].GetProperty().opacity
            self.opacity_spinbox.setValue(curr_opacity)
        else:
            self.color_button.setText("Color")
            if text == 'image': curr_opacity = self.pvqt_store.image_store.image_opacity
            elif text == 'mask': curr_opacity = self.pvqt_store.mask_store.mask_opacity
            else: curr_opacity = self.opacity_spinbox.value()
            self.opacity_spinbox.setValue(curr_opacity)
            # self.reference = None
        
        output = f"-> Actor {text}, and its opacity is {curr_opacity}"
        if output not in self.output_text.toPlainText(): self.output_text.append(output)

    def add_button_actor_name(self, actor_name):
        button = QtWidgets.QPushButton(actor_name)
        button.setCheckable(True)  # Set the button to be checkable
        button.clicked.connect(lambda _, text=actor_name: self.button_actor_name_clicked(text))
        button.setChecked(True)
        # TODO
        button.setFixedSize(self.display.size().width(), 50)

        self.button_layout.insertWidget(0, button) # insert from the top # self.button_layout.addWidget(button)
        self.button_group_actors_names.addButton(button)
        self.button_actor_name_clicked(actor_name)
    
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
        x = (self.pvqt_store.plot_store.plotter.size().width() - self.hintLabel.width()) // 2
        y = (self.pvqt_store.plot_store.plotter.size().height() - self.hintLabel.height()) // 2
        self.hintLabel.move(x, y)

    def reset(self):
        
        # Clear out everything in the remove menu
        for button in self.button_group_actors_names.buttons():
            name = button.text()
            if name == 'image': actor = PvQtStore().image_actor
            elif name == 'mask': actor = PvQtStore().mask_actor
            elif name in PvQtStore().mesh_actors: actor = PvQtStore().mesh_actors[name]
            self.pvqt_store.remove_actor(actor)
            self.remove_actor_button(name, actor)

        self.hintLabel.show()
        self.output_text.clear()
        self.color_button.setText("Color")
    