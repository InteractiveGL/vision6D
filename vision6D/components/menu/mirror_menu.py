from functools import partial

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
        if self.image_actor:
            original_image_data = np.array(PIL.Image.open(self.image_path), dtype='uint8')
            if len(original_image_data.shape) == 2: original_image_data = original_image_data[..., None]
            self.add_image(original_image_data)

        #^ mirror the mask actor
        if self.mask_actor:
            original_mask_data = np.array(PIL.Image.open(self.mask_path), dtype='uint8')
            if len(original_mask_data.shape) == 2: original_mask_data = original_mask_data[..., None]
            self.add_mask(original_mask_data)

        #^ mirror the mesh actors
        if len(self.mesh_actors) != 0:
            for actor_name, _ in self.mesh_actors.items():
                transformation_matrix = self.transformation_matrix
                if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                self.add_mesh(actor_name, self.meshdict[actor_name], transformation_matrix)
            
            # Output the mirrored transformation matrix
            self.output_text.append(f"-> Mirrored transformation matrix is: \n{transformation_matrix}")
