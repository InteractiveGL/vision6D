from typing import List, Dict, Tuple
import pathlib
import logging
import numpy as np
import math
import functools

import pyvista as pv
import matplotlib.pyplot as plt
import vision6D as vis

logger = logging.getLogger("vision6D")

class App:

    def __init__(
            self, 
            register,
            width: int=1920,
            height: int=1080,
            scale: float=1,
            # use surgical microscope for medical device with view angle 1 degree
            cam_focal_length:int=5e+4,
            cam_viewup: Tuple=(0,-1,0),
            mirror_objects: bool=False
        ):
        
        self.window_size = (int(width*scale), int(height*scale))
        self.scale = scale
        self.mirror_objects = mirror_objects
        self.reference = None
        self.transformation_matrix = None
        
        self.image_actors = {}
        self.mesh_actors = {}
        
        self.image_polydata = {}
        self.mesh_polydata = {}
        
        self.binded_meshes = {}
        
        # default opacity for image and surface
        self.set_image_opacity(0.8) # self.image_opacity = 0.35
        self.set_surface_opacity(0.99) # self.surface_opacity = 1

        # Set up the camera
        self.camera = pv.Camera()
        self.cam_focal_length = cam_focal_length
        self.cam_viewup = cam_viewup
        self.set_camera_intrinsics(self.window_size[0], self.window_size[1], self.cam_focal_length)
        self.set_camera_extrinsics(self.cam_focal_length, self.scale, self.cam_viewup)
        
        # Set the attribute and its implications
        self.set_register(register)
        
        # render image and ossicles
        self.pv_render = pv.Plotter(window_size=[self.window_size[0], self.window_size[1]], lighting=None, off_screen=True)
        self.pv_render.store_image = True
        
    def set_mirror_objects(self, mirror_objects: bool):
        self.mirror_objects = mirror_objects

    def set_register(self, register: bool):
        # plot image and ossicles
        self.register = register
        
        if self.register:
            self.pv_plotter = pv.Plotter(window_size=[self.window_size[0], self.window_size[1]])
        else:
            self.pv_plotter = pv.Plotter(window_size=[self.window_size[0], self.window_size[1]], off_screen=True)
            self.pv_plotter.store_image = True
    
    def set_image_opacity(self, image_opacity: float):
        self.image_opacity = image_opacity
    
    def set_surface_opacity(self, surface_opacity: float):
        self.surface_opacity = surface_opacity

    def set_camera_extrinsics(self, cam_focal_length, scale, cam_viewup):
        self.camera.SetPosition((0,0,-(cam_focal_length/100)/scale))
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
        
        # Setting the focal length
        view_angle = 180 / math.pi * (2.0 * math.atan2(height/2.0, f))
        self.camera.SetViewAngle(view_angle) # ~30 degree
        
    def set_transformation_matrix(self, matrix:np.ndarray):
        self.transformation_matrix = matrix
    
    def set_reference(self, name:str):
        self.reference = name
        
    def set_vertices(self, name:str, vertices: pv.pyvista_ndarray):
        if vertices.shape[0] == 3: 
            setattr(self, f"{name}_vertices", vertices)
        elif vertices.shape[1] == 3: 
            setattr(self, f"{name}_vertices", vertices.T)
        
    # Suitable for total two and above mesh quantities
    def bind_meshes(self, main_mesh: str, key: str):
        
        other_meshes = []
    
        for mesh_name in self.mesh_polydata.keys():
                if mesh_name != main_mesh:
                    other_meshes.append(mesh_name)
            
        self.binded_meshes[main_mesh] = {'key': key, 'meshes': other_meshes}
        
    def load_image(self, image_source:np.ndarray, scale_factor:list=[0.01,0.01,1]):
        
        self.image_polydata['image'] = pv.UniformGrid(dimensions=(1920, 1080, 1), spacing=scale_factor, origin=(0.0, 0.0, 0.0))
        self.image_polydata['image'].point_data["values"] = image_source.reshape((1920*1080, 3)) # order = 'C

        self.image_polydata['image'] = self.image_polydata['image'].translate(-1 * np.array(self.image_polydata['image'].center), inplace=False)
        self.image_polydata["image-origin"] = self.image_polydata['image'].copy()
            
        # Then add it to the plotter
        image = self.pv_plotter.add_mesh(self.image_polydata['image'], rgb=True, opacity=self.image_opacity, name='image')
        actor, _ = self.pv_plotter.add_actor(image, pickable=False, name="image")

        # Save actor for later
        self.image_actors["image"] = actor
        self.image_actors["image-origin"] = actor.copy()        

    def load_meshes(self, paths: Dict[str, (pathlib.Path or pv.PolyData)]):
        
        if self.transformation_matrix is None:
           raise RuntimeError("Transformation matrix is not set")
        
        for mesh_name, mesh_source in paths.items():
            
            reference_name = mesh_name

            if isinstance(mesh_source, pathlib.WindowsPath):
                # Load the '.mesh' file
                assert '.mesh' in str(mesh_source), "the file type has to be '.mesh'"
                trimesh_data = vis.utils.load_trimesh(mesh_source, self.mirror_objects)
                mesh_data = pv.wrap(trimesh_data)
            
            self.mesh_polydata[mesh_name] = mesh_data

            self.set_vertices(mesh_name, mesh_data.points)
            
            # set the color to be the meshes' initial location, and never change the color
            colors = vis.utils.color_mesh(mesh_data.points.T)
            
            # Color the vertex
            mesh_data.point_data.set_scalars(colors)

            mesh = self.pv_plotter.add_mesh(mesh_data, rgb=True, opacity = self.surface_opacity, name=mesh_name)
            
            mesh.user_matrix = self.transformation_matrix
            
            actor, _ = self.pv_plotter.add_actor(mesh, pickable=True, name=mesh_name)
            
            # Save actor for later
            self.mesh_actors[mesh_name] = actor

        if len(self.mesh_actors) == 1: self.set_reference(reference_name)

    def center_meshes(self, paths: Dict[str, (pathlib.Path or pv.PolyData)]):
        assert self.reference is not None, "Need to set the self.reference name first!"

        other_meshes = {}
        for id, obj in self.mesh_polydata.items():
            center = np.mean(obj.points, axis=0)
            obj.points -= center
            if id == self.reference:
                reference_center = center.copy()
                # vis.utils.writemesh(paths[id], obj.points.T, center=True)
            else:
                other_meshes[id] = center

        # add the offset
        for id, center in other_meshes.items():
            offset = center - reference_center
            self.mesh_polydata[id].points += offset
            # vis.utils.writemesh(paths[id], self.mesh_polydata[id].points.T, center=True)
    
        print('hhhh')

    # configure event functions
    def event_zoom_out(self, *args):
        self.pv_plotter.camera.zoom(0.5)
        logger.debug("event_zoom_out callback complete")

    def event_zoom_in(self, *args):
        self.pv_plotter.camera.zoom(2)
        logger.debug("event_zoom_in callback complete")

    def event_reset_camera(self, *args):
        self.pv_plotter.camera = self.camera.copy()
        logger.debug("reset_camera_event callback complete")
        
    def event_toggle_image_opacity(self, *args, up):
        if up:
            self.image_opacity += 0.2
            if self.image_opacity >= 1:
                self.image_opacity = 1
        else:
            self.image_opacity -= 0.2
            if self.image_opacity <= 0:
                self.image_opacity = 0
        
        self.image_actors["image"].GetProperty().opacity = self.image_opacity
        self.pv_plotter.add_actor(self.image_actors["image"], pickable=False, name="image")

        logger.debug("event_toggle_image_opacity callback complete")
        
    def event_toggle_surface_opacity(self, *args, up):    
        if up:
            self.surface_opacity += 0.2
            if self.surface_opacity > 1:
                self.surface_opacity = 1
        else:
            self.surface_opacity -= 0.2
            if self.surface_opacity < 0:
                self.surface_opacity = 0
                
        transformation_matrix = self.mesh_actors[self.reference].user_matrix
        for actor_name, actor in self.mesh_actors.items():
            actor.user_matrix = transformation_matrix
            actor.GetProperty().opacity = self.surface_opacity
            self.pv_plotter.add_actor(actor, pickable=True, name=actor_name)

        logger.debug("event_toggle_surface_opacity callback complete")
        
    def event_track_registration(self, *args):
        
        transformation_matrix = self.mesh_actors[self.reference].user_matrix
        for actor_name, actor in self.mesh_actors.items():
            actor.user_matrix = transformation_matrix
            self.pv_plotter.add_actor(actor, pickable=True, name=actor_name)
            logger.debug(f"<Actor {actor_name}> RT: \n{actor.user_matrix}")
    
    def event_realign_meshes(self, *args, main_mesh=None, other_meshes=[]):
        
        objs = {'fix' : main_mesh,
                'move': other_meshes}
        
        transformation_matrix = self.mesh_actors[f"{objs['fix']}"].user_matrix
        
        for obj in objs['move']:
            self.mesh_actors[f"{obj}"].user_matrix = transformation_matrix
            self.pv_plotter.add_actor(self.mesh_actors[f"{obj}"], pickable=True, name=obj)
        
        logger.debug(f"realign: main => {main_mesh}, others => {other_meshes} complete")
        
    def event_gt_position(self, *args):
        
        for actor_name, actor in self.mesh_actors.items():
            actor.user_matrix = self.transformation_matrix
            self.pv_plotter.add_actor(actor, pickable=True, name=actor_name)

        logger.debug(f"\ncurrent gt rt: \n{self.transformation_matrix}")
        logger.debug("event_gt_position callback complete")
        
    def event_change_gt_position(self, *args):
        self.transformation_matrix = self.mesh_actors[self.reference].user_matrix
        for actor_name, actor in self.mesh_actors.items():
            actor.user_matrix = self.transformation_matrix
            self.pv_plotter.add_actor(actor, pickable=True, name=actor_name)
        
        logger.debug(f"\ncurrent gt rt: \n{self.transformation_matrix}")
        logger.debug("event_change_gt_position callback complete")
    
    def plot(self):
        
        if self.reference is None:
           raise RuntimeError("reference name is not set")

        self.pv_plotter.enable_joystick_actor_style()
        self.pv_plotter.enable_trackball_actor_style()

        # Register callbacks
        self.pv_plotter.add_key_event('c', self.event_reset_camera)
        self.pv_plotter.add_key_event('z', self.event_zoom_out)
        self.pv_plotter.add_key_event('x', self.event_zoom_in)
        self.pv_plotter.add_key_event('t', self.event_track_registration)

        for main_mesh, mesh_data in self.binded_meshes.items():
            event_func = functools.partial(self.event_realign_meshes, main_mesh=main_mesh, other_meshes=mesh_data['meshes'])
            self.pv_plotter.add_key_event(mesh_data['key'], event_func)
        
        self.pv_plotter.add_key_event('k', self.event_gt_position)
        self.pv_plotter.add_key_event('l', self.event_change_gt_position)
        
        event_toggle_image_opacity_up_func = functools.partial(self.event_toggle_image_opacity, up=True)
        self.pv_plotter.add_key_event('v', event_toggle_image_opacity_up_func)
        event_toggle_image_opacity_down_func = functools.partial(self.event_toggle_image_opacity, up=False)
        self.pv_plotter.add_key_event('b', event_toggle_image_opacity_down_func)
        
        event_toggle_surface_opacity_up_func = functools.partial(self.event_toggle_surface_opacity, up=True)
        self.pv_plotter.add_key_event('y', event_toggle_surface_opacity_up_func)
        event_toggle_surface_opacity_up_func = functools.partial(self.event_toggle_surface_opacity, up=False)
        self.pv_plotter.add_key_event('u', event_toggle_surface_opacity_up_func)
        
        # Set the camera initial parameters
        self.pv_plotter.camera = self.camera.copy()
        
        if self.register:
            self.pv_plotter.add_axes()
            # add the camera orientation to move the camera
            _ = self.pv_plotter.add_camera_orientation_widget()
            # Actual presenting
            self.pv_plotter.show(title="vision6D") # cpos: [(0.0, 0.0, -500.0), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)]
        else:
            self.pv_plotter.disable()
            self.pv_plotter.show(title="vision6D")
            last_image = self.pv_plotter.last_image
            return last_image
        
    def render_scene(self, render_image:bool, image_source:np.ndarray=None, scale_factor:Tuple[float] = (0.01, 0.01, 1), render_objects:List=[], surface_opacity:float=1):
        
        self.pv_render.enable_joystick_actor_style()
 
        if render_image:
            image = pv.UniformGrid(dimensions=(1920, 1080, 1), spacing=scale_factor, origin=(0.0, 0.0, 0.0))
            image.point_data["values"] = image_source.reshape((1920*1080, 3)) # order = 'C
            image = image.translate(-1 * np.array(image.center), inplace=False)
            self.pv_render.add_mesh(image, rgb=True, opacity=1, name="image")
        else:
            # background set to black
            self.pv_render.set_background('black')
            
            # Render the targeting objects
            for object in render_objects:
                mesh = self.pv_render.add_mesh(self.mesh_polydata[object], rgb=True, opacity=surface_opacity)
                mesh.user_matrix = self.transformation_matrix
        
        self.pv_render.camera = self.camera.copy()
        self.pv_render.disable()
        self.pv_render.show()
        
        return self.pv_render.last_image
