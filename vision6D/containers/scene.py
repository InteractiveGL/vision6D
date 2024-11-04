import math
import numpy as np
import pyvista as pv
from functools import partial
from ..tools import utils
from . import ImageContainer
from . import MaskContainer
from . import BboxContainer
from . import MeshContainer

class Scene():
    def __init__(self, plotter, output_text):
        self.plotter = plotter
        self.camera = pv.Camera()
        self.output_text = output_text
        
        # set up the camera props
        self.image_container = ImageContainer(plotter=self.plotter)
        self.mask_container = MaskContainer(plotter=self.plotter)
        self.mesh_container = MeshContainer(plotter=self.plotter)
        self.bbox_container = BboxContainer(plotter=self.plotter)

        self.track_image_actors = []
        self.track_mask_actors = []
        self.track_mesh_actors = []
        self.track_bbox_actors = []

        # set camera related attributes for the linemod dataset
        # self.fx = 572.4114
        # self.fy = 573.57043
        # self.cx = 325.2611
        # self.cy = 242.04899

        self.fx = 18466.768907841793
        self.fy = 19172.02089833029
        self.cx = 954.4324739015676
        self.cy = 538.2131876789998
        self.canvas_height = 1080
        self.canvas_width = 1920
        self.cam_viewup = (0, -1, 0)

    def reset_camera(self):
        self.plotter.camera = self.camera.copy()

    def zoom_in(self):
        self.plotter.camera.zoom(2)

    def zoom_out(self):
        self.plotter.camera.zoom(0.5)

    def set_camera_extrinsics(self, cam_viewup):
        self.camera.SetPosition((0, 0, -1e-8)) # Set the camera position at the origin of the world coordinate system
        self.camera.SetFocalPoint((0, 0, 0)) # Get the camera window center
        self.camera.SetViewUp(cam_viewup)
    
    def set_camera_intrinsics(self, fx, fy, cx, cy, height):
        # Set camera intrinsic attribute
        self.camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        view_angle = (180 / math.pi) * (2.0 * math.atan2(height/2.0, fy)) # or view_angle = np.degrees(2.0 * math.atan2(height/2.0, f)) # focal_length = (height / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
        self.camera.SetViewAngle(view_angle) # view angle should be in degrees

    def set_distance2camera(self, distance):
        distance = float(distance)
        image_model = self.image_container.images[self.image_container.reference]
        reference_mask = self.mask_container.masks[self.mask_container.reference]
        reference_bbox = self.bbox_container.bboxes[self.bbox_container.reference]
        if image_model is not None:
            pv_obj = pv.ImageData(dimensions=(image_model.width, image_model.height, 1), spacing=[1, 1, 1], origin=(0.0, 0.0, 0.0))
            pv_obj.point_data["values"] = image_model.source_obj.reshape((image_model.width * image_model.height, image_model.channel)) # order = 'C
            pv_obj.translate(-1 * np.array(pv_obj.center), inplace=True) # center the image at (0, 0)
            pv_obj.translate(-np.array([0, 0, pv_obj.center[-1]]), inplace=True) # very important, re-center it to [0, 0, 0]
            pv_obj.translate(np.array([0, 0, distance]), inplace=True)
        if reference_mask is not None:
            reference_mask.pv_obj.translate(-np.array([0, 0, reference_mask.pv_obj.center[-1]]), inplace=True) # very important, re-center it to [0, 0, 0]
            reference_mask.pv_obj.translate(np.array([0, 0, distance]), inplace=True)
        if reference_bbox is not None:
            reference_bbox.pv_obj.translate(-np.array([0, 0, reference_bbox.pv_obj.center[-1]]), inplace=True) # very important, re-center it to [0, 0, 0]
            reference_bbox.pv_obj.translate(np.array([0, 0, distance]), inplace=True)
        #! do not modify the distance2camera for meshes, because it will mess up the pose
        image_model.distance2camera = distance
        self.reset_camera()

    def mirror_image(self, name, direction):
        # Mirror the image
        image_model = self.image_container.images[name]
        if direction == 'x': image_model.mirror_x = not image_model.mirror_x
        elif direction == 'y': image_model.mirror_y = not image_model.mirror_y
        if image_model.mirror_x: image_model.source_obj = image_model.source_obj[:, ::-1, :]
        if image_model.mirror_y: image_model.source_obj = image_model.source_obj[::-1, :, :]
        self.image_container.add_image_actor(image_model, self.fy, self.cx, self.cy)

        # Set up the camera
        self.set_camera_intrinsics(self.fx, self.fy, self.cx, self.cy, self.canvas_height)
        self.set_camera_extrinsics(self.cam_viewup)
        self.reset_camera()

    def tap_toggle_opacity(self):
        if self.mesh_container.meshes[self.mesh_container.reference].opacity == 1.0: 
            self.mesh_container.meshes[self.mesh_container.reference].opacity = 0.0
            self.image_container.images[self.image_container.reference].opacity = 1.0
        elif self.mesh_container.meshes[self.mesh_container.reference].opacity == 0.9:
            self.mesh_container.meshes[self.mesh_container.reference].opacity = 1.0
            self.image_container.images[self.image_container.reference].opacity = 0.0
        else:
            self.mesh_container.meshes[self.mesh_container.reference].opacity = 0.9
            self.image_container.images[self.image_container.reference].opacity = 0.9
        self.image_container.images[self.image_container.reference].actor.GetProperty().opacity = self.image_container.images[self.image_container.reference].opacity
        self.image_container.images[self.image_container.reference].opacity_spinbox.setValue(self.image_container.images[self.image_container.reference].opacity)
        self.mesh_container.meshes[self.mesh_container.reference].opacity_spinbox.setValue(self.mesh_container.meshes[self.mesh_container.reference].opacity)

    def ctrl_tap_opacity(self):
        if self.mesh_container.reference is not None:
            for mesh_model in self.mesh_container.meshes.values():
                if mesh_model.opacity != 0: mesh_model.opacity_spinbox.setValue(0)
                else: mesh_model.opacity_spinbox.setValue(mesh_model.previous_opacity)
        else:
            if self.image_container.image_model.reference is not None:
                image_model = self.image_container.images[self.image_container.image_model.reference]
                if image_model.opacity != 0: image_model.opacity_spinbox.setValue(0)
                else: image_model.opacity_spinbox.setValue(image_model.previous_opacity)
            if self.mask_container.mask_model.reference is not None:
                mask_model = self.mask_container.masks[self.mask_container.mask_model.reference]
                if mask_model.opacity != 0: mask_model.opacity_spinbox.setValue(0)
                else: mask_model.opacity_spinbox.setValue(mask_model.previous_opacity)
            if self.bbox_container.bbox_model.reference is not None:
                bbox_model = self.bbox_container.bboxes[self.bbox_container.bbox_model.reference]
                if bbox_model.opacity != 0: bbox_model.opacity_spinbox.setValue(0)
                else: bbox_model.opacity_spinbox.setValue(bbox_model.previous_opacity)

    def handle_transformation_matrix(self, transformation_matrix):
        self.toggle_register(transformation_matrix)
        self.update_gt_pose()

    def toggle_register(self, pose):
        self.mesh_container.meshes[self.mesh_container.reference].actor.user_matrix = pose
        self.mesh_container.meshes[self.mesh_container.reference].undo_poses.append(pose)
        self.mesh_container.meshes[self.mesh_container.reference].undo_poses = self.mesh_container.meshes[self.mesh_container.reference].undo_poses[-20:]

    def handle_mesh_click(self, name, output_text):
        self.mesh_container.reference = name
        mesh_model = self.mesh_container.meshes[self.mesh_container.reference]
        text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
            mesh_model.actor.user_matrix[0, 0], mesh_model.actor.user_matrix[0, 1], mesh_model.actor.user_matrix[0, 2], mesh_model.actor.user_matrix[0, 3], 
            mesh_model.actor.user_matrix[1, 0], mesh_model.actor.user_matrix[1, 1], mesh_model.actor.user_matrix[1, 2], mesh_model.actor.user_matrix[1, 3], 
            mesh_model.actor.user_matrix[2, 0], mesh_model.actor.user_matrix[2, 1], mesh_model.actor.user_matrix[2, 2], mesh_model.actor.user_matrix[2, 3],
            mesh_model.actor.user_matrix[3, 0], mesh_model.actor.user_matrix[3, 1], mesh_model.actor.user_matrix[3, 2], mesh_model.actor.user_matrix[3, 3])
        if output_text:
            self.output_text.append(f"--> Mesh {name} pose is:")
            self.output_text.append(text)
        self.mesh_container.meshes[self.mesh_container.reference].undo_poses.append(mesh_model.actor.user_matrix)
        self.mesh_container.meshes[self.mesh_container.reference].undo_poses = self.mesh_container.meshes[self.mesh_container.reference].undo_poses[-20:]

    def handle_image_click(self, name):
        # Add a new image as current reference
        self.image_container.reference = name
        image_model = self.image_container.images[self.image_container.reference]
        self.image_container.add_image_actor(image_model, self.fx, self.cx, self.cy)

        # Set up the camera
        self.set_camera_intrinsics(self.fx, self.fy, self.cx, self.cy, self.canvas_height)
        self.set_camera_extrinsics(self.cam_viewup)

        # Set up the opacity box
        # Disconnect existing connections to prevent accumulation
        try: image_model.opacity_spinbox.valueChanged.disconnect()
        except TypeError: pass  # No existing connection

        image_model.opacity_spinbox.setValue(image_model.opacity)
        self.image_container.images[name].opacity_spinbox.valueChanged.connect(partial(self.image_container.set_image_opacity, name))

        # Update opacity for all images
        for image_name, img_model in self.image_container.images.items():
            if image_name != name and img_model is not None:
                img_model.opacity = 0.0
                img_model.opacity_spinbox.setValue(0.0)
                # Remove the other actors exist to free memory (very important)
                if hasattr(img_model, 'actor') and img_model.actor in self.plotter.renderer.actors.values():
                    self.plotter.remove_actor(img_model.actor)
                    del img_model.actor  # Remove reference to the actor
            else:
                img_model.opacity = 0.9
                img_model.opacity_spinbox.setValue(0.9)
        
    def handle_mask_click(self, name):
        # Add your mask handling code here
        self.mask_container.reference = name
        for mask_name, mask_model in self.mask_container.masks.items():
            if mask_name != name: mask_model.opacity = 0.0; mask_model.opacity_spinbox.setValue(0.0)
            else: mask_model.opacity = 0.9; mask_model.opacity_spinbox.setValue(0.9)

    def handle_bbox_click(self, name):
        # Add your bbox handling code here
        pass #* For fixing some bugs in segmesh render function

    def mesh_color_value_change(self, name, color):
        if name in self.mesh_container.meshes:
            try:
                color = self.mesh_container.set_color(name, color)
                self.mesh_container.meshes[name].color = color
                if color != "nocs" and color != "texture": 
                    self.mesh_container.meshes[name].color_button.setStyleSheet(f"background-color: {self.mesh_container.meshes[name].color}")
            except ValueError:
                utils.display_warning(f"Cannot set color ({color}) to {name}")

    def mask_color_value_change(self, name, color):
        if name in self.mask_container.masks: 
            try:
                self.mask_container.set_mask_color(name, color)
                self.mask_container.masks[name].color = color
            except ValueError: 
                utils.display_warning(f"Cannot set color ({color}) to mask")
                self.mask_container.mask_model.color_button.setStyleSheet(f"background-color: {self.mask_container.masks[name].color}")

    def bbox_color_value_change(self, name, color):
        if name in self.bbox_container.bboxes:
            try:
                self.bbox_container.set_bbox_color(name, color)
                self.bbox_container.bboxes[name].color = color
            except ValueError:
                utils.display_warning(f"Cannot set color ({color}) to mask")
                self.bbox_container.bbox_model.color_button.setStyleSheet(f"background-color: {self.bbox_container.bboxes[name].color}")

    def add_pose(self, matrix:np.ndarray=None, rot:np.ndarray=None, trans:np.ndarray=None):
        if matrix is None and (rot is not None and trans is not None): matrix = np.vstack((np.hstack((rot, trans)), [0, 0, 0, 1]))
        self.mesh_container.meshes[self.mesh_container.reference].initial_pose = matrix
        self.reset_gt_pose()

    def reset_gt_pose(self):
        if self.mesh_container.reference:
            mesh_model = self.mesh_container.meshes[self.mesh_container.reference]
            # if mesh_model.initial_pose is not None:
            text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
            mesh_model.initial_pose[0, 0], mesh_model.initial_pose[0, 1], mesh_model.initial_pose[0, 2], mesh_model.initial_pose[0, 3], 
            mesh_model.initial_pose[1, 0], mesh_model.initial_pose[1, 1], mesh_model.initial_pose[1, 2], mesh_model.initial_pose[1, 3], 
            mesh_model.initial_pose[2, 0], mesh_model.initial_pose[2, 1], mesh_model.initial_pose[2, 2], mesh_model.initial_pose[2, 3],
            mesh_model.initial_pose[3, 0], mesh_model.initial_pose[3, 1], mesh_model.initial_pose[3, 2], mesh_model.initial_pose[3, 3])
            self.output_text.append("-> Reset the GT pose to:")
            self.output_text.append(text)
            self.toggle_register(mesh_model.initial_pose)
            self.reset_camera()
        else: utils.display_warning("Need to set a reference mesh first")

    def update_gt_pose(self):
        if self.mesh_container.reference:
            mesh_model = self.mesh_container.meshes[self.mesh_container.reference]
            # if mesh_model.initial_pose is not None:
            mesh_model.initial_pose = mesh_model.actor.user_matrix
            self.toggle_register(mesh_model.actor.user_matrix)
            text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
            mesh_model.initial_pose[0, 0], mesh_model.initial_pose[0, 1], mesh_model.initial_pose[0, 2], mesh_model.initial_pose[0, 3], 
            mesh_model.initial_pose[1, 0], mesh_model.initial_pose[1, 1], mesh_model.initial_pose[1, 2], mesh_model.initial_pose[1, 3], 
            mesh_model.initial_pose[2, 0], mesh_model.initial_pose[2, 1], mesh_model.initial_pose[2, 2], mesh_model.initial_pose[2, 3],
            mesh_model.initial_pose[3, 0], mesh_model.initial_pose[3, 1], mesh_model.initial_pose[3, 2], mesh_model.initial_pose[3, 3])
            self.output_text.append(f"-> Update the {self.mesh_container.reference} GT pose to:")
            self.output_text.append(text)
        else: utils.display_warning("Need to set a reference mesh first")