from ..components import CameraStore
from ..components import ImageStore
from ..components import MaskStore
from ..components import BboxStore
from ..components import MeshStore
from ..components import VideoStore
from ..components import FolderStore

from . import ImageContainer
from . import MaskContainer
from . import BboxContainer
from . import MeshContainer

from ..tools import utils

import numpy as np

class Scene():
    def __init__(self, plotter, output_text):
        self.plotter = plotter
        self.output_text = output_text
        self.track_image_actors = []
        self.track_mask_actors = []
        self.track_mesh_actors = []
        self.track_bbox_actors = []

        self.camera_store = CameraStore(self.plotter)
        self.image_store = ImageStore(self.plotter)
        self.mask_store = MaskStore()
        self.bbox_store = BboxStore()
        self.mesh_store = MeshStore()
        self.video_store = VideoStore()
        self.folder_store = FolderStore()

        # set up the camera props
        self.image_container = ImageContainer(plotter=self.plotter)
        self.mask_container = MaskContainer(plotter=self.plotter)
        self.mesh_container = MeshContainer(plotter=self.plotter)
        self.bbox_container = BboxContainer(plotter=self.plotter)

    def set_distance2camera(self, distance):
        distance = float(distance)
        if self.image_store.images[self.image_store.reference] is not None:
            self.image_store.images[self.image_store.reference].image_pv.translate(-np.array([0, 0, self.image_store.images[self.image_store.reference].image_pv.center[-1]]), inplace=True) # very important, re-center it to [0, 0, 0]
            self.image_store.images[self.image_store.reference].image_pv.translate(np.array([0, 0, distance]), inplace=True)
        if self.mask_store.mask_actor is not None:
            self.mask_store.mask_pv.translate(-np.array([0, 0, self.mask_store.mask_pv.center[-1]]), inplace=True) # very important, re-center it to [0, 0, 0]
            self.mask_store.mask_pv.translate(np.array([0, 0, distance]), inplace=True)
        if self.bbox_store.bbox_actor is not None:
            self.bbox_store.bbox_pv.translate(-np.array([0, 0, self.bbox_store.bbox_pv.center[-1]]), inplace=True) # very important, re-center it to [0, 0, 0]
            self.bbox_store.bbox_pv.translate(np.array([0, 0, distance]), inplace=True)
        #! do not modify the distance2camera for meshes, because it will mess up the pose
        self.image_store.images[self.image_store.reference].distance2camera = distance
        self.camera_store.reset_camera()

    def tap_toggle_opacity(self):
        if self.mesh_store.meshes[self.mesh_store.reference].opacity == 1.0: 
            self.mesh_store.meshes[self.mesh_store.reference].opacity = 0.0
            self.image_store.images[self.image_store.reference].opacity = 1.0
        elif self.mesh_store.meshes[self.mesh_store.reference].opacity == 0.9:
            self.mesh_store.meshes[self.mesh_store.reference].opacity = 1.0
            self.image_store.images[self.image_store.reference].opacity = 0.0
        else:
            self.mesh_store.meshes[self.mesh_store.reference].opacity = 0.9
            self.image_store.images[self.image_store.reference].opacity = 0.9
        self.image_store.images[self.image_store.reference].actor.GetProperty().opacity = self.image_store.images[self.image_store.reference].opacity
        self.image_store.images[self.image_store.reference].opacity_spinbox.setValue(self.image_store.images[self.image_store.reference].opacity)
        self.mesh_store.meshes[self.mesh_store.reference].opacity_spinbox.setValue(self.mesh_store.meshes[self.mesh_store.reference].opacity)

    def ctrl_tap_opacity(self):
        if self.mesh_store.reference is not None:
            for mesh_data in self.mesh_store.meshes.values():
                if mesh_data.opacity != 0: mesh_data.opacity_spinbox.setValue(0)
                else: mesh_data.opacity_spinbox.setValue(mesh_data.previous_opacity)
        else:
            if self.image_store.reference is not None:
                if self.image_store.opacity != 0: self.image_store.opacity_spinbox.setValue(0)
                else: self.image_store.opacity_spinbox.setValue(self.image_store.previous_opacity)
            if self.mask_store.mask_actor is not None:
                if self.mask_store.mask_opacity != 0: self.mask_store.opacity_spinbox.setValue(0)
                else: self.mask_store.opacity_spinbox.setValue(self.mask_store.previous_opacity)
            if self.bbox_store.bbox_actor is not None:
                if self.bbox_store.bbox_opacity != 0: self.bbox_store.opacity_spinbox.setValue(0)
                else: self.bbox_store.opacity_spinbox.setValue(self.bbox_store.previous_opacity)

    def on_camera_options_selection_change(self, option):
        if option == "Set Camera":
            self.image_container.set_camera()
        elif option == "Reset Camera (c)":
            self.camera_store.reset_camera()
        elif option == "Zoom In (x)":
            self.camera_store.zoom_in()
        elif option == "Zoom Out (z)":
            self.camera_store.zoom_out()
        elif option == "Calibrate":
            self.image_container.camera_calibrate()

    def handle_transformation_matrix(self, transformation_matrix):
        self.toggle_register(transformation_matrix)
        self.mesh_container.update_gt_pose()

    def toggle_register(self, pose):
        self.mesh_store.meshes[self.mesh_store.reference].actor.user_matrix = pose
        self.mesh_store.meshes[self.mesh_store.reference].undo_poses.append(pose)
        self.mesh_store.meshes[self.mesh_store.reference].undo_poses = self.mesh_store.meshes[self.mesh_store.reference].undo_poses[-20:]

    def handle_mesh_click(self, name, output_text):
        self.mesh_store.reference = name
        mesh_data = self.mesh_store.meshes[name]
        text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
            mesh_data.actor.user_matrix[0, 0], mesh_data.actor.user_matrix[0, 1], mesh_data.actor.user_matrix[0, 2], mesh_data.actor.user_matrix[0, 3], 
            mesh_data.actor.user_matrix[1, 0], mesh_data.actor.user_matrix[1, 1], mesh_data.actor.user_matrix[1, 2], mesh_data.actor.user_matrix[1, 3], 
            mesh_data.actor.user_matrix[2, 0], mesh_data.actor.user_matrix[2, 1], mesh_data.actor.user_matrix[2, 2], mesh_data.actor.user_matrix[2, 3],
            mesh_data.actor.user_matrix[3, 0], mesh_data.actor.user_matrix[3, 1], mesh_data.actor.user_matrix[3, 2], mesh_data.actor.user_matrix[3, 3])
        if output_text:
            self.output_text.append(f"--> Mesh {name} pose is:")
            self.output_text.append(text)
        self.mesh_store.meshes[self.mesh_store.reference].undo_poses.append(mesh_data.actor.user_matrix)
        self.mesh_store.meshes[self.mesh_store.reference].undo_poses = self.mesh_store.meshes[self.mesh_store.reference].undo_poses[-20:]

    def handle_image_click(self, name):
        self.image_store.reference = name
        for image_name, image_data in self.image_store.images.items():
            if image_name != name: image_data.opacity = 0.0; image_data.opacity_spinbox.setValue(0.0)
            else: image_data.opacity = 0.9; image_data.opacity_spinbox.setValue(0.9)

    def handle_mask_click(self, name):
        # Add your mask handling code here
        pass

    def handle_bbox_click(self, name):
        # Add your bbox handling code here
        pass #* For fixing some bugs in segmesh render function

    def color_value_change(self, color, name):
        if name == 'mask': 
            try:
                self.mask_container.set_mask_color(color)
                self.mask_store.color = color
            except ValueError: 
                utils.display_warning(f"Cannot set color ({color}) to mask")
                self.mask_store.color_button.setStyleSheet(f"background-color: {self.mask_store.color}")
        elif name == 'bbox':
            try:
                self.bbox_container.set_bbox_color(color)
                self.bbox_store.color = color
            except ValueError: 
                utils.display_warning(f"Cannot set color ({color}) to bbox")
                self.bbox_store.color_button.setStyleSheet(f"background-color: {self.bbox_store.color}")
        elif name in self.mesh_store.meshes:
            try:
                color = self.mesh_container.set_color(color, name)
                self.mesh_store.meshes[name].color = color
                if color != "nocs" and color != "texture": 
                    self.mesh_store.meshes[name].color_button.setStyleSheet(f"background-color: {self.mesh_store.meshes[name].color}")
            except ValueError:
                utils.display_warning(f"Cannot set color ({color}) to {name}")

    def add_pose(self, matrix:np.ndarray=None, rot:np.ndarray=None, trans:np.ndarray=None):
        if matrix is None and (rot is not None and trans is not None): matrix = np.vstack((np.hstack((rot, trans)), [0, 0, 0, 1]))
        self.mesh_store.meshes[self.mesh_store.reference].initial_pose = matrix
        self.reset_gt_pose()

    def reset_gt_pose(self):
        if self.mesh_store.reference:
            mesh_data = self.mesh_store.meshes[self.mesh_store.reference]
            # if mesh_data.initial_pose is not None:
            text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
            mesh_data.initial_pose[0, 0], mesh_data.initial_pose[0, 1], mesh_data.initial_pose[0, 2], mesh_data.initial_pose[0, 3], 
            mesh_data.initial_pose[1, 0], mesh_data.initial_pose[1, 1], mesh_data.initial_pose[1, 2], mesh_data.initial_pose[1, 3], 
            mesh_data.initial_pose[2, 0], mesh_data.initial_pose[2, 1], mesh_data.initial_pose[2, 2], mesh_data.initial_pose[2, 3],
            mesh_data.initial_pose[3, 0], mesh_data.initial_pose[3, 1], mesh_data.initial_pose[3, 2], mesh_data.initial_pose[3, 3])
            self.output_text.append("-> Reset the GT pose to:")
            self.output_text.append(text)
            self.toggle_register(mesh_data.initial_pose)
            self.camera_store.reset_camera()
        else: utils.display_warning("Need to set a reference mesh first")

    def update_gt_pose(self):
        if self.mesh_store.reference:
            mesh_data = self.mesh_store.meshes[self.mesh_store.reference]
            # if mesh_data.initial_pose is not None:
            mesh_data.initial_pose = mesh_data.actor.user_matrix
            self.toggle_register(mesh_data.actor.user_matrix)
            text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
            mesh_data.initial_pose[0, 0], mesh_data.initial_pose[0, 1], mesh_data.initial_pose[0, 2], mesh_data.initial_pose[0, 3], 
            mesh_data.initial_pose[1, 0], mesh_data.initial_pose[1, 1], mesh_data.initial_pose[1, 2], mesh_data.initial_pose[1, 3], 
            mesh_data.initial_pose[2, 0], mesh_data.initial_pose[2, 1], mesh_data.initial_pose[2, 2], mesh_data.initial_pose[2, 3],
            mesh_data.initial_pose[3, 0], mesh_data.initial_pose[3, 1], mesh_data.initial_pose[3, 2], mesh_data.initial_pose[3, 3])
            self.output_text.append(f"-> Update the {self.mesh_store.reference} GT pose to:")
            self.output_text.append(text)
        else: utils.display_warning("Need to set a reference mesh first")