'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: custom_qt_interactor.py
@time: 2023-07-03 20:30
@desc: custom overwrite default qt interactor to include the mouse press and release related events.
'''
import os
import vtk
from ..tools import utils
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
        #* very important to sync the poses if the link_mesh_button is checked
        self.main_window.on_link_mesh_button_toggle(checked=self.main_window.link_mesh_button.isChecked(), clicked=False)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        mesh_paths = []
        image_paths = []
        mask_paths = []
        unsupported_files = []

        mask_indicators = ['_mask', '_seg', '_label']
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            # Load mesh files
            if file_path.endswith(('.mesh', '.ply', '.stl', '.obj', '.off', '.dae', '.fbx', '.3ds', '.x3d')):
                mesh_paths.append(file_path)
            # Load image/mask files
            elif file_path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp', '.ico')):
                filename = os.path.basename(file_path).lower()
                if any(indicator in filename for indicator in mask_indicators): mask_paths.append(file_path)
                else: image_paths.append(file_path)
            else:
                unsupported_files.append(file_path)

        self.main_window.add_mesh_file(mesh_paths=mesh_paths)
        self.main_window.add_image_file(image_paths=image_paths)
        self.main_window.add_mask_file(mask_paths=mask_paths)
        if len(unsupported_files) > 0: utils.display_warning(f"File {unsupported_files} format is not supported!")