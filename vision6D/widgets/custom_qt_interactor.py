from typing import Optional

import vtk
from pyvistaqt import QtInteractor

class CustomQtInteractor(QtInteractor):
    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == 1 or event.button() == 4:  # Left or middle mouse button
            self.press_callback(self.iren.interactor)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == 1 or event.button() == 4:  # Left or middle mouse button
            if self.picker: self.release_callback()
            
    def press_callback(self, obj, *args):
        x, y = obj.GetEventPosition()
        picker = vtk.vtkCellPicker()
        if picker.Pick(x, y, 0, self.renderer): self.picker = picker
        else: self.picker = None

    def release_callback(self):
        from ..stores import QtStore, PvQtStore
        
        pvqt_store = PvQtStore()
        picked_actor = self.picker.GetActor()
        actor_name = picked_actor.name
        if actor_name in pvqt_store.mesh_actors:        
            if actor_name not in pvqt_store.undo_poses: pvqt_store.undo_poses[actor_name] = []
            pvqt_store.undo_poses[actor_name].append(pvqt_store.mesh_actors[actor_name].user_matrix)
            if len(pvqt_store.undo_poses[actor_name]) > 20: pvqt_store.undo_poses[actor_name].pop(0)
            # check the picked button
            QtStore().check_button(actor_name)