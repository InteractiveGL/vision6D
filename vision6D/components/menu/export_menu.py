from ...stores import PlotStore
from ...stores import ImageStore
from ...stores import MaskStore
from ...stores import QtStore

from PyQt5 import QtWidgets
import pathlib
import PIL.Image
import numpy as np

class ExportMenu():

    def __init__(self):

        self.plot_store = PlotStore()
        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.qt_store = QtStore()

    def export_image(self):
        if self.image_store.image_actor:
            image = self.image_store.render_image(camera=self.plot_store.camera.copy())
            output_path, _ = QtWidgets.QFileDialog().getSaveFileName(self, "Save File", "", "Image Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.qt_store.output_text.append(f"-> Export image render to:\n {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def export_mask(self):
        if self.mask_store.mask_actor:
            image = self.mask_store.render_mask(camera=self.plot_store.camera.copy())
            output_path, _ = QtWidgets.QFileDialog().getSaveFileName(self, "Save File", "", "Mask Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.qt_store.output_text.append(f"-> Export mask render to:\n {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a mask first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
   
    def export_pose(self):
        if self.mesh_store.reference: 
            self.mesh_store.update_gt_pose()
            output_path, _ = QtWidgets.QFileDialog().getSaveFileName(self, "Save File", "", "Pose Files (*.npy)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.npy')
                np.save(output_path, self.mesh_store.transformation_matrix)
                self.qt_store.output_text.append(f"-> Saved:\n{self.mesh_store.transformation_matrix}\nExport to:\n {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
    
    def export_mesh_render(self):
        if self.mesh_store.reference:
            image = self.mesh_store.render_mesh(camera=self.plot_store.camera.copy())
            output_path, _ = QtWidgets.QFileDialog().getSaveFileName(self, "Save File", "", "Mesh Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.qt_store.output_text.append(f"-> Export reference mesh render to:\n {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def export_segmesh_render(self):
        if self.mesh_store.reference and self.mask_store.mask_actor:
            segmask = self.mask_store.render_mask(camera=self.plot_store.camera.copy())
            if np.max(segmask) > 1: segmask = segmask / 255
            image = self.mesh_store.render_mesh(camera=self.plot_store.camera.copy())
            image = (image * segmask).astype(np.uint8)
            output_path, _ = QtWidgets.QFileDialog().getSaveFileName(self, "Save File", "", "SegMesh Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.qt_store.output_text.append(f"-> Export segmask render:\n to {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a mesh or mask first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
