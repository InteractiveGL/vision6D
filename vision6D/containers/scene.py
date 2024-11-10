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
    
    def set_camera_intrinsics(self, fx, fy, cx, cy, height):
        # Set camera intrinsic attribute
        self.camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        view_angle = (180 / math.pi) * (2.0 * math.atan2(height/2.0, fy)) # or view_angle = np.degrees(2.0 * math.atan2(height/2.0, f)) # focal_length = (height / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
        self.camera.SetViewAngle(view_angle) # view angle should be in degrees

    def set_camera_extrinsics(self, cam_viewup):
        self.camera.SetPosition((0, 0, -1e-8)) # Set the camera position at the origin of the world coordinate system
        self.camera.SetFocalPoint((0, 0, 0)) # Get the camera window center
        self.camera.SetViewUp(cam_viewup)
    
    def set_distance2camera(self, distance):
        distance = float(distance)
        image_model = self.image_container.images[self.image_container.reference]
        mask_model = self.mask_container.masks[self.mask_container.reference]
        bbox_model = self.bbox_container.bboxes[self.bbox_container.reference]
        if image_model is not None:
            pv_obj = pv.ImageData(dimensions=(image_model.width, image_model.height, 1), spacing=[1, 1, 1], origin=(0.0, 0.0, 0.0))
            pv_obj.point_data["values"] = image_model.source_obj.reshape((image_model.width * image_model.height, image_model.channel)) # order = 'C
            pv_obj.translate(-1 * np.array(pv_obj.center), inplace=True) # center the image at (0, 0)
            pv_obj.translate(-np.array([0, 0, pv_obj.center[-1]]), inplace=True) # very important, re-center it to [0, 0, 0]
            pv_obj.translate(np.array([0, 0, distance]), inplace=True)
        if mask_model is not None:
            mask_model.pv_obj.translate(-np.array([0, 0, mask_model.pv_obj.center[-1]]), inplace=True) # very important, re-center it to [0, 0, 0]
            mask_model.pv_obj.translate(np.array([0, 0, distance]), inplace=True)
        if bbox_model is not None:
            bbox_model.pv_obj.translate(-np.array([0, 0, bbox_model.pv_obj.center[-1]]), inplace=True) # very important, re-center it to [0, 0, 0]
            bbox_model.pv_obj.translate(np.array([0, 0, distance]), inplace=True)
        #! do not modify the distance2camera for meshes, because it will mess up the pose
        image_model.distance2camera = distance
        self.reset_camera()

    def tap_toggle_opacity(self):
        if self.mesh_container.reference is not None and self.image_container.reference is not None:
            mesh_model = self.mesh_container.meshes[self.mesh_container.reference]
            image_model = self.image_container.images[self.image_container.reference]
            if mesh_model.opacity == 1.0: 
                mesh_model.opacity = 0.0
                image_model.opacity = 1.0
            elif mesh_model.opacity == 0.9:
                mesh_model.opacity = 1.0
                image_model.opacity = 0.0
            else:
                mesh_model.opacity = 0.9
                image_model.opacity = 0.9
            image_model.actor.GetProperty().opacity = image_model.opacity
            image_model.opacity_spinbox.setValue(image_model.opacity)
            mesh_model.opacity_spinbox.setValue(mesh_model.opacity)
        elif self.mesh_container.reference is not None:
            mesh_model = self.mesh_container.meshes[self.mesh_container.reference]
            if mesh_model.opacity == 1.0: mesh_model.opacity = 0.0
            elif mesh_model.opacity == 0.9: mesh_model.opacity = 1.0
            else: mesh_model.opacity = 0.9
            mesh_model.opacity_spinbox.setValue(mesh_model.opacity)

    def ctrl_tap_opacity(self):
        if self.mesh_container.reference is not None:
            for mesh_model in self.mesh_container.meshes.values():
                if mesh_model.opacity != 0: mesh_model.opacity_spinbox.setValue(0)
                else: mesh_model.opacity_spinbox.setValue(mesh_model.previous_opacity)
        else:
            if self.image_container.reference is not None:
                image_model = self.image_container.images[self.image_container.reference]
                if image_model.opacity != 0: image_model.opacity_spinbox.setValue(0)
                else: image_model.opacity_spinbox.setValue(image_model.previous_opacity)
            if self.mask_container.reference is not None:
                mask_model = self.mask_container.masks[self.mask_container.reference]
                if mask_model.opacity != 0: mask_model.opacity_spinbox.setValue(0)
                else: mask_model.opacity_spinbox.setValue(mask_model.previous_opacity)
            if self.bbox_container.reference is not None:
                bbox_model = self.bbox_container.bboxes[self.bbox_container.reference]
                if bbox_model.opacity != 0: bbox_model.opacity_spinbox.setValue(0)
                else: bbox_model.opacity_spinbox.setValue(bbox_model.previous_opacity)

    def handle_mesh_click(self, name, output_text):
        self.mesh_container.reference = name
        mesh_model = self.mesh_container.meshes[self.mesh_container.reference]
        matrix = utils.get_actor_user_matrix(mesh_model)
        text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
            matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3], 
            matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3], 
            matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3],
            matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3])
        if output_text:
            self.output_text.append(f"--> Mesh {name} pose is:")
            self.output_text.append(text)
        mesh_model.undo_poses.append(mesh_model.actor.user_matrix)
        mesh_model.undo_poses = mesh_model.undo_poses[-20:]

    def handle_image_click(self, name):
        # Add a new image as current reference
        self.image_container.reference = name
        image_model = self.image_container.images[self.image_container.reference]
        self.image_container.add_image_actor(image_model, self.fx, self.cx, self.cy)

        # Disconnect existing connections to prevent accumulation
        try: image_model.opacity_spinbox.valueChanged.disconnect()
        except TypeError: pass  # No existing connection

        # Set up the value in the opacity box
        image_model.opacity_spinbox.setValue(image_model.opacity)
        image_model.opacity_spinbox.valueChanged.connect(partial(self.image_container.set_image_opacity, name))

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
                mesh_model = self.mesh_container.meshes[name]
                mesh_model.color = color
                if color != "nocs" and color != "texture": 
                    mesh_model.color_button.setStyleSheet(f"background-color: {mesh_model.color}")
            except ValueError:
                utils.display_warning(f"Cannot set color ({color}) to {name}")

    def mask_color_value_change(self, name, color):
        if name in self.mask_container.masks: 
            try:
                self.mask_container.set_mask_color(name, color)
                mask_model = self.mask_container.masks[name]
                mask_model.color = color
            except ValueError: 
                utils.display_warning(f"Cannot set color ({color}) to mask")
                mask_model.color_button.setStyleSheet(f"background-color: {mask_model.color}")

    def bbox_color_value_change(self, name, color):
        if name in self.bbox_container.bboxes:
            bbox_model = self.bbox_container.bboxes[name]
            try:
                self.bbox_container.set_bbox_color(name, color)
                bbox_model.color = color
            except ValueError:
                utils.display_warning(f"Cannot set color ({color}) to mask")
                bbox_model.color_button.setStyleSheet(f"background-color: {bbox_model.color}")