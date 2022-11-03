import pathlib
import logging

import pyvista as pv

logger = logging.getLogger("visualization")

class App:
    
    def __init__(self):
        
        # Show output
        self.pl = pv.Plotter()
        self.pl.add_axes(interactive=True)
        self.actors = {}
        self.actor_attrs = {}
        
    def load_image(self, image_path:pathlib.Path, scaling:list=[1,1,1]):
        
        # Check if the file exists
        if not image_path.exists():
            logger.error(f"{image_path} does not exists")
            return None
        
        # Create image mesh and add it to the plotter
        image = pv.read(image_path)
        
        # Scale down image
        image = image.scale(scaling, inplace=False)
        
        # Then add it to the plotter
        actor = self.pl.add_mesh(image, rgb=True, opacity=0.65)
        actor, actor_attr = self.pl.add_actor(actor, name="image")
        
        # Save actor for later
        self.actors["image"] = actor
        self.actors["image-origin"] = actor.copy()
        self.actor_attrs['image'] = actor_attr
        
    def load_mesh(self, mesh_path:pathlib.Path, name:str):
        
        # Check if the file exists
        if not mesh_path.exists():
            logger.error(f"{mesh_path} does not exists")
            return None
        
        # Create image mesh and add it to the plotter
        mesh = pv.read(mesh_path)
        actor = self.pl.add_mesh(mesh, rgb=True)
        actor, actor_attr = self.pl.add_actor(actor, name=name)
        
        # Save actor for later
        self.actors[name] = actor
        self.actor_attrs[name] = actor_attr
        
    def report_registration(self, *args):
        
        for actor_name, actor in self.actors.items():
            logger.debug(f"<Actor {actor_name}> T: {actor.position} R: {actor.orientation}")
        
    def plot(self):
        
        # "xy" view
        # pl.camera_position = [(7.75202772140503, 3.917879838943482, 53.657579687507386), (7.75202772140503, 3.917879838943482, 0.4370880126953125), (0, 1, 0)]

        self.pl.enable_joystick_actor_style()
    
        def reset_camera_event():
            self.pl.camera.position = (7.75202772140503, 3.917879838943482, 53.657579687507386)
            self.pl.camera.focal_point = (7.75202772140503, 3.917879838943482, 0.4370880126953125)
            self.pl.camera.up = (0.0, 1.0, 0.0)
            logger.debug("reset_camera_event callback complete")
            
        def reset_image_position():
            self.actors["image"] = self.actors["image-origin"].copy()
            self.pl.add_actor(self.actors["image"], name="image")
            logger.debug("reset_image_position callback complete")
        
        # Register callbacks
        self.pl.add_key_event('c', reset_camera_event)
        self.pl.add_key_event('i', reset_image_position)
        self.pl.track_click_position(self.report_registration, side='left')
        
        # add the camera orientation to move the camera
        _ = self.pl.add_camera_orientation_widget()
        
        # Actual presenting
        cpos = self.pl.show(cpos="xy", return_cpos=True)
        logger.debug(f"cpos: {cpos}")