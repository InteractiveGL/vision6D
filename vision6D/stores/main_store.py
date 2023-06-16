
from .singleton import Singleton

from .qt_store import QtStore
from .paths_store import PathsStore
from .pvqt import PvQtStore

class MainStore(metaclass=Singleton):

    def __init__(self, main_window):

        self.paths_store = PathsStore()
        self.pvqt_store = PvQtStore(main_window.signal_close)
        self.qt_store = QtStore(main_window)

    def remove_actor(self, name):

        if name == 'image': 
            actor = self.pvqt_store.image_store.remove_actor()
        elif name == 'mask':
            actor =self.pvqt_store.mask_store.remove_actor()
        else: 
            actor = self.pvqt_store.mesh_store.remove_actor(name)
            if actor: self.qt_store.color_button.setText("Color")
        
        if actor:
            self.pvqt_store.remove_actor(actor)
            self.pvqt_store.track_actors_names.remove(name)

        # clear out the plot if there is no actor
        if self.pvqt_store.image_store.image_actor is None and self.pvqt_store.mask_store.mask_actor is None and len(self.pvqt_store.mesh_store.mesh_actors) == 0: 
            self.clear_plot()

    def clear_plot(self):
        self.qt_store.reset()
        self.paths_store.reset()
        self.pvqt_store.reset()