import numpy as np
import PIL.Image

from ...stores import QtStore
from ...stores import PlotStore
from ...stores import ImageStore
from ...stores import MaskStore
from ...stores import MeshStore

class MirrorMenu():
    def __init__(self):

        self.plot_store = PlotStore()
        self.qt_store = QtStore()
        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.mesh_store = MeshStore()

    def mirror_actors(self, direction):
        if direction == 'x': self.plot_store.mirror_x = not self.plot_store.mirror_x
        elif direction == 'y': self.plot_store.mirror_y = not self.plot_store.mirror_y

        #^ mirror the image actor
        if self.image_store.image_actor: 
            original_image_data = np.array(PIL.Image.open(self.image_store.image_path), dtype='uint8')
            if len(original_image_data.shape) == 2: original_image_data = original_image_data[..., None]
            self.image_store.add_image(original_image_data)

        #^ mirror the mask actor
        if self.mask_store.mask_actor:
            original_mask_data = np.array(PIL.Image.open(self.mask_store.mask_path), dtype='uint8')
            if len(original_mask_data.shape) == 2: original_mask_data = original_mask_data[..., None]
            self.mask_store.add_mask(original_mask_data)

        #^ mirror the mesh actors
        if len(self.mesh_store.mesh_actors) != 0:
            for actor_name, _ in self.mesh_store.mesh_actors.items():
                transformation_matrix = self.plot_store.mirror_transformation_matrix()
                self.mesh_store.add_mesh(actor_name, self.mesh_store.meshdict[actor_name], transformation_matrix)
            
            # Output the mirrored transformation matrix
            self.qt_store.output_text.append(f"-> Mirrored transformation matrix is: \n{transformation_matrix}")
