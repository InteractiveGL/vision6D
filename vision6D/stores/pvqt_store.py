import os

import numpy as np

from PyQt5 import QtWidgets

from .singleton import Singleton

from ..stores import PlotStore
from ..stores import QtStore
from ..stores import ImageStore
from ..stores import MaskStore
from ..stores import MeshStore

np.set_printoptions(suppress=True)

class PvQtStore(metaclass=Singleton):

    def __init__(self, button_group_actors_names=None, 
                       display_width=None, 
                       button_layout=None, 
                       color_button=None, 
                       opacity_spinbox=None):
        super().__init__()

        self.track_actors_names = []
        self.button_group_actors_names = button_group_actors_names
        self.display_width = display_width
        self.button_layout = button_layout
        self.color_button = color_button
        self.opacity_spinbox = opacity_spinbox

        # * Pv(vtk) elements
        self.plot_store = PlotStore()
        self.qt_store = QtStore()
        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.mesh_store = MeshStore()

    # button related methods
    def button_actor_name_clicked(self, text):
        if text in self.mesh_store.mesh_actors:
            # set the current mesh color
            self.color_button.setText(self.mesh_store.mesh_colors[text])
            # set mesh reference
            self.mesh_store.reference = text
            self.mesh_store.current_pose()
            curr_opacity = self.mesh_store.mesh_actors[self.mesh_store.reference].GetProperty().opacity
            self.opacity_spinbox.setValue(curr_opacity)
        else:
            self.color_button.setText("Color")
            if text == 'image': curr_opacity = self.image_store.image_opacity
            elif text == 'mask': curr_opacity = self.mask_store.mask_opacity
            else: curr_opacity = self.opacity_spinbox.value()
            self.opacity_spinbox.setValue(curr_opacity)
            # self.mesh_store.reference = None
        
        output = f"-> Actor {text}, and its opacity is {curr_opacity}"
        if output not in self.qt_store.output_text.toPlainText(): self.qt_store.output_text.append(output)

    def add_button_actor_name(self, actor_name):
        button = QtWidgets.QPushButton(actor_name)
        button.setCheckable(True)  # Set the button to be checkable
        button.clicked.connect(lambda _, text=actor_name: self.button_actor_name_clicked(text))
        button.setChecked(True)
        button.setFixedSize(self.display_width, 50)

        self.button_layout.insertWidget(0, button) # insert from the top # self.button_layout.addWidget(button)
        self.button_group_actors_names.addButton(button)
        self.button_actor_name_clicked(actor_name)
    
    def remove_actor_button(self, name, button):

        # Mention the removal and remove button from group
        self.qt_store.output_text.append(f"-> Remove actor: {name}")
        self.button_group_actors_names.removeButton(button)
        self.button_layout.removeWidget(button)
        button.deleteLater()

    def check_button(self, actor_name):
        for button in self.button_group_actors_names.buttons():
            if button.text() == actor_name: 
                button.setChecked(True)
                self.button_actor_name_clicked(actor_name)
                break
       