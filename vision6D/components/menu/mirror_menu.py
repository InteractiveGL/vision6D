from functools import partial
import numpy as np
import PIL.Image

class MirrorMenu():

    def __init__(self, menu):
        # Save parameter
        self.menu = menu
        self.menu.addAction('Mirror X axis', partial(self.mirror_actors, direction='x'))
        self.menu.addAction('Mirror Y axis', partial(self.mirror_actors, direction='y'))

    def mirror_actors(self, direction):

        if direction == 'x': self.mirror_x = not self.mirror_x
        elif direction == 'y': self.mirror_y = not self.mirror_y

        #^ mirror the image actor
        if self.pvqt_store.image_store.image_actor: 
            original_image_data = np.array(PIL.Image.open(self.paths_store.image_path), dtype='uint8')
            if len(original_image_data.shape) == 2: original_image_data = original_image_data[..., None]
            self.pvqt_store.image_store.add_image(original_image_data)

        #^ mirror the mask actor
        if self.pvqt_store.mask_store.mask_actor:
            original_mask_data = np.array(PIL.Image.open(self.paths_store.mask_path), dtype='uint8')
            if len(original_mask_data.shape) == 2: original_mask_data = original_mask_data[..., None]
            self.pvqt_store.mask_store.add_mask(original_mask_data)

        #^ mirror the mesh actors
        if len(self.pvqt_store.mesh_store.mesh_actors) != 0:
            for actor_name, _ in self.pvqt_store.mesh_store.mesh_actors.items():
                transformation_matrix = self.pvqt_store.camera_store.mirror_transformation_matrix()
                self.pvqt_store.mesh_store.add_mesh(actor_name, self.pvqt_store.mesh_store.meshdict[actor_name], transformation_matrix)
            
            # Output the mirrored transformation matrix
            self.qt_store.output_text.append(f"-> Mirrored transformation matrix is: \n{transformation_matrix}")
