
import pathlib
import PIL.Image
from PyQt5 import QtWidgets

from .. import utils
from ..components import CameraStore
from ..components import ImageStore

class ImageContainer:
    def __init__(self, 
                plotter,
                hintLabel, 
                track_actors_names, 
                add_button_actor_name, 
                check_button,
                output_text):
          
        self.plotter = plotter
        self.hintLabel = hintLabel
        self.track_actors_names = track_actors_names
        self.add_button_actor_name = add_button_actor_name
        self.check_button = check_button
        self.output_text = output_text

        self.camera_store = CameraStore()
        self.image_store = ImageStore()

    def add_image_file(self, image_path='', prompt=False):
        if prompt:
            image_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)")
        if image_path:
            self.hintLabel.hide()
            self.add_image(image_path)

    def mirror_image(self, direction):
        if direction == 'x': self.image_store.mirror_x = not self.image_store.mirror_x
        elif direction == 'y': self.image_store.mirror_y = not self.image_store.mirror_y
        self.add_image(self.image_store.image_path)

    def add_image(self, image_source):

        image, original_image, channel = self.image_store.add_image(image_source)

        # Then add it to the plotter
        if channel == 1: 
            image = self.plotter.add_mesh(image, cmap='gray', opacity=self.image_store.image_opacity, name='image')
        else: 
            image = self.plotter.add_mesh(image, rgb=True, opacity=self.image_store.image_opacity, name='image')
        
        actor, _ = self.plotter.add_actor(image, pickable=False, name='image')

        # Save actor for later
        self.image_store.image_actor = actor

        # get the image scalar
        image_data = utils.get_image_actor_scalars(self.image_store.image_actor)
        assert (image_data == original_image).all() or (image_data*255 == original_image).all(), "image_data and image_source should be equal"
        
        # add remove current image to removeMenu
        if 'image' not in self.track_actors_names:
            self.track_actors_names.append('image')
            self.add_button_actor_name('image')
        self.check_button('image')
                                          
    def set_image_opacity(self, image_opacity: float):
        self.image_store.image_opacity = image_opacity
        self.image_store.image_actor.GetProperty().opacity = image_opacity
        self.plotter.add_actor(self.image_store.image_actor, pickable=False, name='image')

    def toggle_image_opacity(self, up):
        change = 0.05
        if not up: change *= -1
        self.image_store.update_opacity(change)
        self.plotter.add_actor(self.image_store.image_actor, pickable=False, name="image")
        self.check_button('image')

    def export_image(self):
        if self.image_store.image_actor:
            image = self.image_store.render_image(camera=self.plotter.camera.copy())
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File", "", "Image Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.output_text.append(f"-> Export image render to:\n {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0