import vtk
from pyvistaqt import QtInteractor

class CustomQtInteractor(QtInteractor):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        # Save main_window
        self.main_window = main_window

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
        picked_actor = self.picker.GetActor()
        actor_name = picked_actor.name
        if actor_name in self.main_window.mesh_store.mesh_actors:
            # check the picked mesh button and register the rest to the mesh reference's current pose
            self.main_window.check_button(actor_name)
