'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: mask_container.py
@time: 2023-07-03 20:26
@desc: create container for mask related actions in application
'''

import matplotlib.colors
from ..tools import utils
from ..components import ImageStore
from ..components import MaskStore
from ..widgets import MaskWindow, LiveWireWindow, SamWindow

class MaskContainer:
    def __init__(self, plotter):

        self.plotter = plotter
        self.image_store = ImageStore()
        self.mask_store = MaskStore()

    def mirror_mask(self, direction):
        if direction == 'x': self.mask_store.mirror_x = not self.mask_store.mirror_x
        elif direction == 'y': self.mask_store.mirror_y = not self.mask_store.mirror_y
        self.add_mask(self.mask_store.mask_path)

    def load_mask(self, mask_data):
        # Add mask surface object to the plot
        mask_mesh = self.plotter.add_mesh(mask_data, color=mask_data.color, style='surface', opacity=mask_data.mask_opacity)
        actor, _ = self.plotter.add_actor(mask_mesh, pickable=True, name='mask')
        mask_data.actor = actor
        
    def add_mask(self, mask_source):
        mask_data = self.mask_store.add_mask(mask_source, self.image_store.image_center, (self.image_store.width, self.image_store.height))
        mask_data = self.load_mask(mask_data)
        return mask_data
    
    def reset_mask(self):
        if self.mask_store.mask_path:
            self.mask_store.mirror_x = False
            self.mask_store.mirror_y = False
            mask_surface = self.mask_store.add_mask(self.mask_store.mask_path, self.image_store.image_center, (self.image_store.width, self.image_store.height))
            self.load_mask(mask_surface)

    def set_mask_opacity(self, mask_opacity: float):
        self.mask_store.previous_opacity = self.mask_store.mask_opacity
        self.mask_store.mask_opacity = mask_opacity
        self.mask_store.mask_actor.GetProperty().opacity = mask_opacity

    def set_mask_color(self, color):
        self.mask_store.mask_actor.GetMapper().SetScalarVisibility(0)
        self.mask_store.mask_actor.GetProperty().SetColor(matplotlib.colors.to_rgb(color))
    
    def draw_mask(self, live_wire=False, sam=False):
        def handle_output_path_change(output_path):
            if output_path:
                self.mask_store.mask_path = output_path
                self.add_mask(self.mask_store.mask_path)
        if self.image_store.images[self.image_store.reference].actor:
            image = utils.get_image_actor_scalars(self.image_store.images[self.image_store.reference].actor)
            if sam: self.mask_window = SamWindow(image)
            elif live_wire: self.mask_window = LiveWireWindow(image)
            else: self.mask_window = MaskWindow(image)
            self.mask_window.mask_label.output_path_changed.connect(handle_output_path_change)
        else: utils.display_warning("Need to load an image first!")