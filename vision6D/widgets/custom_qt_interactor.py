'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: custom_qt_interactor.py
@time: 2023-07-03 20:30
@desc: custom overwrite default qt interactor to include the mouse press and release related events.
'''

import vtk
from pyvistaqt import QtInteractor

class CustomQtInteractor(QtInteractor):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.set_background("4c4c4c")
        self.selected_actor = None  # Initialize the attribute here

    def mousePressEvent(self, event):
        # Call superclass method for left and middle buttons
        if event.button() in (1, 4): super().mousePressEvent(event)
        # Always call press_callback
        self.press_callback(self.iren.interactor)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() in (1, 2, 4):  # Left, right, and middle mouse buttons
            self.release_callback()

    def press_callback(self, obj, *args):
        x, y = obj.GetEventPosition()
        prop_picker = vtk.vtkPropPicker()
        if prop_picker.Pick(x, y, 0, self.renderer):
            self.selected_actor = prop_picker.GetActor()  # Use a different attribute name

    def release_callback(self):
        if self.selected_actor:
            name = self.selected_actor.name
            if name in self.main_window.scene.mesh_container.meshes:
                self.main_window.check_mesh_button(name, output_text=True)
            elif name in self.main_window.scene.mask_container.masks:
                self.main_window.check_mask_button(name)
        self.selected_actor = None  # Reset the attribute
