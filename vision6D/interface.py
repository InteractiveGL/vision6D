import sys
from typing import List, Dict, Tuple, Optional
import pathlib
import logging
import numpy as np
import math
import trimesh
import functools
import numpy as np
import matplotlib.pyplot as plt
import json
import PIL

# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"

from PyQt5.QtWidgets import QMessageBox
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow
import vision6D as vis
from .mainwindow import MyMainWindow

np.set_printoptions(suppress=True)

def try_except(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            if isinstance(args[0], MainWindow): QMessageBox.warning(args[0], 'vision6D', "Need to load a mesh first!", QMessageBox.Ok, QMessageBox.Ok)

    return wrapper

class Interface(MyMainWindow):
    def __init__(self):
        super().__init__()

        # initialize
        self.reference = None
        self.transformation_matrix = np.eye(4)

        self.image_actor = None
        self.mask_actor = None
        self.mesh_actors = {}
        self.mesh_raw = {}
        self.mesh_polydata = {}
        
        self.track_actors_names = []
        self.undo_poses = []
        self.latlon = vis.utils.load_latitude_longitude()
        
        # default opacity for image and surface
        self.set_image_opacity(0.99)
        self.set_mask_opacity(0.5)
        self.set_mesh_opacity(0.8) # self.surface_opacity = 1
        self.spacing = [0.01, 0.01, 1]
        self.set_camera_props(focal_length=50000, cam_viewup=(0, -1, 0), cam_position=-500)
        
    def set_reference(self, name:str):     
        assert name in self.meshdict.keys(), "reference name is not in the path!"
        self.reference = name
  
    def set_transformation_matrix(self, matrix:np.ndarray=None, rot:np.ndarray=None, trans:np.ndarray=None):
        if matrix is None: matrix = np.vstack((np.hstack((rot, trans)), [0, 0, 0, 1]))
        self.transformation_matrix = matrix
        self.initial_pose = self.transformation_matrix
        self.reset_gt_pose()
        self.reset_camera()

    def set_image_opacity(self, image_opacity: float):
        assert image_opacity>=0 and image_opacity<=1, "image opacity should range from 0 to 1!"
        self.image_opacity = image_opacity
        if self.image_actor is not None:
            self.image_actor.GetProperty().opacity = image_opacity
            self.plotter.add_actor(self.image_actor, pickable=False, name='image')

    def set_mask_opacity(self, mask_opacity: float):
        assert mask_opacity>=0 and mask_opacity<=1, "image opacity should range from 0 to 1!"
        self.mask_opacity = mask_opacity
        if self.mask_actor is not None:
            self.mask_actor.GetProperty().opacity = mask_opacity
            self.plotter.add_actor(self.mask_actor, pickable=False, name='mask')

    def set_mesh_opacity(self, surface_opacity: float):
        assert surface_opacity>=0 and surface_opacity<=1, "mesh opacity should range from 0 to 1!"
        self.surface_opacity = surface_opacity
        if len(self.mesh_actors) != 0:
            for actor_name, actor in self.mesh_actors.items():
                actor.user_matrix = pv.array_from_vtkmatrix(actor.GetMatrix())
                actor.GetProperty().opacity = self.surface_opacity
                self.plotter.add_actor(actor, pickable=True, name=actor_name)
    
    def set_camera_extrinsics(self, cam_position, cam_viewup):
        self.camera.SetPosition((0,0,cam_position))
        self.camera.SetFocalPoint((0,0,0))
        self.camera.SetViewUp(cam_viewup)
    
    def set_camera_intrinsics(self, width, height, cam_focal_length):
        
        # Set camera intrinsic attribute
        self.camera_intrinsics = np.array([
            [cam_focal_length, 0, width/2],
            [0, cam_focal_length, height/2],
            [0, 0, 1]
        ])
        
        cx = self.camera_intrinsics[0,2]
        cy = self.camera_intrinsics[1,2]
        f = self.camera_intrinsics[0,0]
        
        # convert the principal point to window center (normalized coordinate system) and set it
        wcx = -2*(cx - float(width)/2) / width
        wcy =  2*(cy - float(height)/2) / height
        self.camera.SetWindowCenter(wcx, wcy) # (0,0)
        
        # Setting the view angle in degrees
        view_angle = (180 / math.pi) * (2.0 * math.atan2(height/2.0, f)) # or view_angle = np.degrees(2.0 * math.atan2(height/2.0, f)) # focal_length = (1080 / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
        self.camera.SetViewAngle(view_angle) # view angle should be in degrees
 
    def set_camera_props(self, focal_length, cam_viewup, cam_position):
        # Set up the camera
        self.camera = pv.Camera()
        self.focal_length = focal_length
        self.cam_viewup = cam_viewup
        self.cam_position = cam_position
        self.set_camera_intrinsics(self.window_size[0], self.window_size[1], self.focal_length)
        self.set_camera_extrinsics(self.cam_position, self.cam_viewup)
        self.plotter.camera = self.camera.copy()

    def add_image(self, image_source):
        dim = image_source.shape
        h, w, channel = dim[0], dim[1], dim[2]

        image = pv.UniformGrid(dimensions=(w, h, 1), spacing=self.spacing, origin=(0.0, 0.0, 0.0))
        image.point_data["values"] = image_source.reshape((w * h, channel)) # order = 'C
        image = image.translate(-1 * np.array(image.center), inplace=False)

        # Then add it to the plotter
        image = self.plotter.add_mesh(image, cmap='gray', opacity=self.mask_opacity, name='image') if channel == 1 else self.plotter.add_mesh(image, rgb=True, opacity=self.image_opacity, name='image')
        actor, _ = self.plotter.add_actor(image, pickable=False, name='image')
        # Save actor for later
        self.image_actor = actor
        
        # get the image scalar
        image_data = vis.utils.get_2D_actor_scalars(self.image_actor)
        assert (image_data == image_source).all() or (image_data*255 == image_source).all(), "image_data and image_source should be equal"

        # add remove current image to removeMenu
        if 'image' not in self.track_actors_names:
            self.track_actors_names.append('image')
            remove_actor = functools.partial(self.remove_actor, 'image')
            self.removeMenu.addAction('image', remove_actor)

        # reset the camera
        self.reset_camera()

    def add_mask(self, mask_source):
        dim = mask_source.shape
        h, w, channel = dim[0], dim[1], dim[2]
        
        mask = pv.UniformGrid(dimensions=(w, h, 1), spacing=self.spacing, origin=(0.0, 0.0, 0.0))
        mask.point_data["values"] = mask_source.reshape((w * h, channel)) # order = 'C
        mask = mask.translate(-1 * np.array(mask.center), inplace=False)

        # Then add it to the plotter
        mask = self.plotter.add_mesh(mask, cmap='gray', opacity=self.mask_opacity, name='mask') if channel == 1 else self.plotter.add_mesh(mask, rgb=True, opacity=self.mask_opacity, name='mask')
        actor, _ = self.plotter.add_actor(mask, pickable=False, name='mask')
        # Save actor for later
        self.mask_actor = actor

        # get the image scalar
        mask_data = vis.utils.get_2D_actor_scalars(self.mask_actor)
        assert (mask_data == mask_source).all() or (mask_data*255 == mask_source).all(), "mask_data and mask_source should be equal"

        # add remove current image to removeMenu
        if 'mask' not in self.track_actors_names:
            self.track_actors_names.append('mask')
            remove_actor = functools.partial(self.remove_actor, 'mask')
            self.removeMenu.addAction('mask', remove_actor)

        # reset the camera
        self.reset_camera()

    def add_mesh(self, mesh_name, mesh_source):
        """ add a mesh to the pyqt frame """

        flag = False
                              
        if isinstance(mesh_source, pathlib.WindowsPath) or isinstance(mesh_source, str):
            # Load the '.mesh' file
            if '.mesh' in str(mesh_source): mesh_source = vis.utils.load_trimesh(mesh_source)
            # Load the '.ply' file
            elif '.ply' in str(mesh_source): mesh_source = pv.read(mesh_source)

        if isinstance(mesh_source, trimesh.Trimesh):
            self.mesh_raw[mesh_name] = mesh_source
            assert (mesh_source.vertices.shape[1] == 3 and mesh_source.faces.shape[1] == 3), "it should be N by 3 matrix"
            # setattr(self, f"{mesh_name}_mesh", mesh_source)
            mesh_data = pv.wrap(mesh_source)
            flag = True

        if isinstance(mesh_source, pv.PolyData):
            self.mesh_raw[mesh_name] = mesh_source
            # setattr(self, f"{mesh_name}_mesh", mesh_source)
            mesh_data = mesh_source
            flag = True

        if not flag:
            QMessageBox.warning(self, 'vision6D', "The mesh format is not supported!", QMessageBox.Ok, QMessageBox.Ok)
            return 0

        self.mesh_polydata[mesh_name] = mesh_data

        mesh = self.plotter.add_mesh(mesh_data, opacity=self.surface_opacity, name=mesh_name)

        mesh.user_matrix = self.transformation_matrix
        self.initial_pose = mesh.user_matrix
                
        # Add and save the actor
        actor, _ = self.plotter.add_actor(mesh, pickable=True, name=mesh_name)
        
        assert actor.name == mesh_name, "actor's name should equal to mesh_name"
        
        self.mesh_actors[mesh_name] = actor

        self.reset_camera()

        # add remove current mesh to removeMenu
        if mesh_name not in self.track_actors_names:
            self.track_actors_names.append(mesh_name)
            remove_actor = functools.partial(self.remove_actor, mesh_name)
            self.removeMenu.addAction(mesh_name, remove_actor)

    def toggle_image_opacity(self, *args, up):
        if up:
            self.image_opacity += 0.2
            if self.image_opacity >= 1: self.image_opacity = 1
        else:
            self.image_opacity -= 0.2
            if self.image_opacity <= 0: self.image_opacity = 0
        
        if self.image_actor is not None:
            self.image_actor.GetProperty().opacity = self.image_opacity
            self.plotter.add_actor(self.image_actor, pickable=False, name="image")

    def toggle_mask_opacity(self, *args, up):
        if up:
            self.mask_opacity += 0.2
            if self.mask_opacity >= 1: self.mask_opacity = 1
        else:
            self.mask_opacity -= 0.2
            if self.mask_opacity <= 0: self.mask_opacity = 0
        
        if self.mask_actor is not None:
            self.mask_actor.GetProperty().opacity = self.mask_opacity
            self.plotter.add_actor(self.mask_actor, pickable=False, name="mask")

    def toggle_surface_opacity(self, *args, up):    
        if up:
            self.surface_opacity += 0.2
            if self.surface_opacity > 1: self.surface_opacity = 1
        else:
            self.surface_opacity -= 0.2
            if self.surface_opacity < 0: self.surface_opacity = 0
                
        if len(self.mesh_actors) != 0:
            transformation_matrix = self.mesh_actors[self.reference].user_matrix
            for actor_name, actor in self.mesh_actors.items():
                actor.user_matrix = transformation_matrix
                actor.GetProperty().opacity = self.surface_opacity
                self.plotter.add_actor(actor, pickable=True, name=actor_name)

    def reset_camera(self, *args):
        self.plotter.camera = self.camera.copy()

    def zoom_in(self, *args):
        self.plotter.camera.zoom(2)

    def zoom_out(self, *args):
        self.plotter.camera.zoom(0.5)

    def track_click_callback(self, *args):
        if len(self.undo_poses) > 20: self.undo_poses.pop(0)
        if self.reference is not None: self.undo_poses.append(self.mesh_actors[self.reference].user_matrix)

    @try_except
    def reset_gt_pose(self, *args):
        print(f"\nRT: \n{self.initial_pose}\n")
        for actor_name, actor in self.mesh_actors.items():
            actor.user_matrix = self.initial_pose
            self.plotter.add_actor(actor, pickable=True, name=actor_name)

    def update_gt_pose(self, *args):
        if self.reference is not None:
            self.transformation_matrix = self.mesh_actors[self.reference].user_matrix
            self.transformation_matrix = self.transformation_matrix
            self.initial_pose = self.transformation_matrix
            self.reset_gt_pose()

    def current_pose(self, *args):
        if self.reference is not None:
            transformation_matrix = self.mesh_actors[self.reference].user_matrix
            print(f"\nRT: \n{transformation_matrix}\n")
            for actor_name, actor in self.mesh_actors.items():
                actor.user_matrix = transformation_matrix
                self.plotter.add_actor(actor, pickable=True, name=actor_name)

    def undo_pose(self, *args):
        if len(self.undo_poses) != 0: 
            transformation_matrix = self.undo_poses.pop()
            if (transformation_matrix == self.mesh_actors[self.reference].user_matrix).all():
                if len(self.undo_poses) != 0: transformation_matrix = self.undo_poses.pop()
            for actor_name, actor in self.mesh_actors.items():
                actor.user_matrix = transformation_matrix
                self.plotter.add_actor(actor, pickable=True, name=actor_name)

    def set_color(self, nocs_color):
        self.nocs_color = nocs_color
        if len(self.mesh_polydata) != 0:
            for mesh_name, mesh_data in self.mesh_polydata.items():
                # get the corresponding color
                colors = vis.utils.color_mesh(mesh_data.points, nocs=self.nocs_color)
                if colors.shape != mesh_data.points.shape: colors = np.ones((len(mesh_data.points), 3)) * 0.5
                assert colors.shape == mesh_data.points.shape, "colors shape should be the same as mesh_data.points shape"
                
                # color the mesh and actor
                mesh = self.plotter.add_mesh(mesh_data, scalars=colors, rgb=True, opacity=self.surface_opacity, name=mesh_name)
                transformation_matrix = pv.array_from_vtkmatrix(self.mesh_actors[mesh_name].GetMatrix())
                mesh.user_matrix = transformation_matrix
                actor, _ = self.plotter.add_actor(mesh, pickable=True, name=mesh_name)
                assert actor.name == mesh_name, "actor's name should equal to mesh_name"
                self.mesh_actors[mesh_name] = actor
        else:
            QMessageBox.warning(self, 'vision6D', "Need to load a mesh first!", QMessageBox.Ok, QMessageBox.Ok)

    def nocs_epnp(self, color_mask, mesh):
        vertices = mesh.vertices
        pts3d, pts2d = vis.utils.create_2d_3d_pairs(color_mask, vertices)
        pts2d = pts2d.astype('float32')
        pts3d = pts3d.astype('float32')
        camera_intrinsics = self.camera_intrinsics.astype('float32')
        predicted_pose = vis.utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics, self.camera.position)
        return predicted_pose

    def latlon_epnp(self, color_mask, mesh):
        binary_mask = vis.utils.color2binary_mask(color_mask)
        idx = np.where(binary_mask == 1)
        # swap the points for opencv, maybe because they handle RGB image differently (RGB -> BGR in opencv)
        idx = idx[:2][::-1]
        pts2d = np.stack((idx[0], idx[1]), axis=1)
        pts3d = []
        
        # Obtain the rg color
        color = color_mask[pts2d[:,1], pts2d[:,0]][..., :2]
        if np.max(color) > 1: color = color / 255
        gx = color[:, 0]
        gy = color[:, 1]

        lat = np.array(self.latlon[..., 0])
        lon = np.array(self.latlon[..., 1])
        lonf = lon[mesh.faces]
        msk = (np.sum(lonf>=0, axis=1)==3) & (np.sum(lat[mesh.faces]>=0, axis=1)==3)
        for i in range(len(pts2d)):
            pt = vis.utils.latLon2xyz(mesh, lat, lonf, msk, gx[i], gy[i])
            pts3d.append(pt)
       
        pts3d = np.array(pts3d).reshape((len(pts3d), 3))

        pts2d = pts2d.astype('float32')
        pts3d = pts3d.astype('float32')
        camera_intrinsics = self.camera_intrinsics.astype('float32')
        
        predicted_pose = vis.utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics, self.camera.position)

        return predicted_pose

    def epnp_mesh(self):
        if self.reference is not None:
            colors = self.mesh_polydata[self.reference].point_data.active_scalars
            if colors is None or (np.all(colors == colors[0])):
                QMessageBox.warning(self, 'vision6D', "The mesh need to be colored with nocs or latlon with gradient color", QMessageBox.Ok, QMessageBox.Ok)
                return 0
            color_mask = self.export_mesh_plot(QMessageBox.Yes, QMessageBox.Yes, QMessageBox.Yes, msg=False)
            gt_pose = self.mesh_actors[self.reference].user_matrix

            if np.sum(color_mask) == 0:
                QMessageBox.warning(self, 'vision6D', "The color mask is blank (maybe set the reference mesh wrong)", QMessageBox.Ok, QMessageBox.Ok)
                return 0
                
            if self.nocs_color:
                predicted_pose = self.nocs_epnp(color_mask, self.mesh_raw[self.reference])
                error = np.sum(np.abs(predicted_pose - gt_pose))
                QMessageBox.about(self,"vision6D", f"PREDICTED POSE: \n{predicted_pose}\nGT POSE: \n{gt_pose}\nERROR: \n{error}")
            else:
                QMessageBox.warning(self, 'vision6D', "Only works using EPnP with latlon mask", QMessageBox.Ok, QMessageBox.Ok)

        else:
            QMessageBox.warning(self, 'vision6D', "A mesh need to be loaded/mesh reference need to be set", QMessageBox.Ok, QMessageBox.Ok)
            return 0

    def epnp_mask(self, nocs_method):
        if self.mask_actor is not None:
            mask_data = vis.utils.get_2D_actor_scalars(self.mask_actor)
            if np.max(mask_data) > 1: mask_data = mask_data / 255

            # binary mask
            if np.all(np.logical_or(mask_data == 0, mask_data == 1)):
                if self.reference is not None:
                    colors = self.mesh_polydata[self.reference].point_data.active_scalars
                    if colors is None or (np.all(colors == colors[0])):
                        QMessageBox.warning(self, 'vision6D', "The mesh need to be colored with nocs or latlon with gradient color", QMessageBox.Ok, QMessageBox.Ok)
                        return 0
                    color_mask = self.export_mesh_plot(QMessageBox.Yes, QMessageBox.Yes, QMessageBox.Yes, msg=False)
                    nocs_color = False if np.sum(color_mask[..., 2]) == 0 else True
                    gt_pose = self.mesh_actors[self.reference].user_matrix
                    mesh = self.mesh_raw[self.reference]
                else: QMessageBox.warning(self, 'vision6D', "A mesh need to be loaded/mesh reference need to be set", QMessageBox.Ok, QMessageBox.Ok); return 0
                color_mask = (color_mask * mask_data).astype(np.uint8)
            # color mask
            else:
                color_mask = mask_data
                if np.sum(color_mask) != 0:
                    unique, counts = np.unique(color_mask, return_counts=True)
                    digit_counts = dict(zip(unique, counts))
                    if digit_counts[0] == np.max(counts): 
                        nocs_color = False if np.sum(color_mask[..., 2]) == 0 else True
                        gt_pose_dir = pathlib.Path(self.mask_path).parent.parent.parent/ 'labels' / 'info.json'
                        with open(gt_pose_dir) as f: data = json.load(f)
                        gt_pose = np.array(data[pathlib.Path(self.mask_path).stem]['gt_pose'])
                        id = pathlib.Path(self.mask_path).stem.split('_')[0].split('.')[1]
                        #TODO: hard coded, and needed to be updated in the future
                        mesh_path = pathlib.Path(self.mask_path).stem.split('_')[0] + '_video_trim' 
                        mesh = vis.utils.load_trimesh(pathlib.Path(self.mesh_dir / mesh_path / "mesh" / "processed_meshes" / f"{id}_right_ossicles_processed.mesh"))
                    else:
                        QMessageBox.warning(self, 'vision6D', "A color mask need to be loaded", QMessageBox.Ok, QMessageBox.Ok)
                        return 0
        
            if np.sum(color_mask) == 0:
                QMessageBox.warning(self, 'vision6D', "The color mask is blank (maybe set the reference mesh wrong)", QMessageBox.Ok, QMessageBox.Ok)
                return 0
                
            if nocs_method == nocs_color:
                if nocs_method: predicted_pose = self.nocs_epnp(color_mask, mesh)
                else: predicted_pose = self.latlon_epnp(color_mask, mesh)
                error = np.sum(np.abs(predicted_pose - gt_pose))
                QMessageBox.about(self,"vision6D", f"PREDICTED POSE: \n{predicted_pose}\nGT POSE: \n{gt_pose}\nERROR: \n{error}")
            else:
                QMessageBox.about(self,"vision6D", "Clicked the wrong method")
        else:
            QMessageBox.about(self,"vision6D", "please load a mask first")
