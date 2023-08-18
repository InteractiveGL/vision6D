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
        self.cell_picker = None

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == 1 or event.button() == 4:  # Left or middle mouse button
            self.press_callback(self.iren.interactor)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == 1 or event.button() == 4:  # Left or middle mouse button
            self.release_callback()
            
    def press_callback(self, obj, *args):
        x, y = obj.GetEventPosition()
        cell_picker = vtk.vtkCellPicker()
        if cell_picker.Pick(x, y, 0, self.renderer): 
            self.cell_picker = cell_picker
        else:
            if self.main_window.image_store.image_actor: 
                self.main_window.check_button('image')

    def release_callback(self):
        if self.cell_picker: 
            picked_actor = self.cell_picker.GetActor()
            name = picked_actor.name
            if (name in self.main_window.mesh_store.meshes) or (name == 'mask'):
                self.main_window.check_button(name)
        self.cell_picker = None