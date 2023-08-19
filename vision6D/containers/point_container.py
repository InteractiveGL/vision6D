from PyQt5 import QtWidgets

from ..tools import utils
from ..components import PointStore

class PointContainer:

    def __init__(self,
                plotter,
                hintLabel,
                track_actors_names, 
                add_button_actor_name,
                output_text):
        
        self.plotter = plotter
        self.hintLabel = hintLabel
        self.track_actors_names = track_actors_names
        self.add_button_actor_name = add_button_actor_name
        self.output_text = output_text
        self.point_store = PointStore()

    def load_points_file(self, point_path='', prompt=False):
        if prompt: 
            point_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.npy *.mesh *.ply *.stl *.obj *.off *.dae *.fbx *.3ds *.x3d)") 
        if point_path:
            self.hintLabel.hide()
            self.load_points(point_path)

    def load_points(self, point_source):
        self.point_store.load_points(point_source)
        if self.point_store.point_data is not None:
            points = self.plotter.add_points(self.point_store.point_data, render_points_as_spheres=True, point_size=3.0, color='Blue')
            actor, _ = self.plotter.add_actor(points, pickable=False, name=self.point_store.point_name)
            self.point_store.point_actors[self.point_store.point_name] = actor
            # add remove current points to removeMenu
            if self.point_store.point_name not in self.track_actors_names:
                self.track_actors_names.append(self.point_store.point_name)
                self.add_button_actor_name(self.point_store.point_name)
        else: utils.display_warning("The point format is not supported!")

    def add_points(self):
        ...