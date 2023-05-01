import pathlib
import logging
import numpy as np
import functools
import numpy as np
import PIL

# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QInputDialog, QFileDialog 
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow
import vision6D as vis

np.set_printoptions(suppress=True)

class MyMainWindow(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)
        
        # setting title
        self.setWindowTitle("Vision6D")
        self.showMaximized()
        self.window_size = (1920, 1080)
        
        # create the frame
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout()

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        # self.plotter.setFixedSize(*self.window_size) # but camera locate in the center instead of top left
        self.render = pv.Plotter(window_size=[self.window_size[0], self.window_size[1]], lighting=None, off_screen=True) 
        self.render.set_background('black'); 
        assert self.render.background_color == "black", "render's background need to be black"

        vlayout.addWidget(self.plotter.interactor)
        self.signal_close.connect(self.plotter.close)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()
        # simple dialog to record users input info
        self.input_dialog = QInputDialog()
        self.file_dialog = QFileDialog()
        
        self.image_dir = pathlib.Path('E:\\GitHub\\ossicles_6D_pose_estimation\\data\\frames')
        self.mask_dir = pathlib.Path('E:\\GitHub\\yolov8\\runs\\segment')
        self.mesh_dir = pathlib.Path('E:\\GitHub\\ossicles_6D_pose_estimation\\data\\surgical_planning')
        self.gt_poses_dir = pathlib.Path('E:\\GitHub\\ossicles_6D_pose_estimation\\data\\gt_poses')

        self.image_path = None
        self.mask_path = None
        self.mesh_path = None
        self.meshdict = {}
        
        os.makedirs(vis.config.GITROOT / "output", exist_ok=True)
        os.makedirs(vis.config.GITROOT / "output" / "image", exist_ok=True)
        os.makedirs(vis.config.GITROOT / "output" / "mask", exist_ok=True)
        os.makedirs(vis.config.GITROOT / "output" / "mesh", exist_ok=True)
        os.makedirs(vis.config.GITROOT / "output" / "segmesh", exist_ok=True)
        os.makedirs(vis.config.GITROOT / "output" / "gt_poses", exist_ok=True)
            
        # allow to add files
        fileMenu = mainMenu.addMenu('File')
        add_image_file = functools.partial(self.add_image_file, 'image')
        fileMenu.addAction('Add Image', add_image_file)
        add_mask_file = functools.partial(self.add_image_file, 'mask')
        fileMenu.addAction('Add Mask', add_mask_file)
        fileMenu.addAction('Add Mesh', self.add_mesh_file)
        fileMenu.addAction('Add Pose', self.add_pose_file)
        self.removeMenu = fileMenu.addMenu("Remove")
        fileMenu.addAction('Clear', self.clear_plot)

        # allow to export files
        exportMenu = mainMenu.addMenu('Export')
        export_image_plot = functools.partial(self.export_image_plot, 'image')
        exportMenu.addAction('Image Render', export_image_plot)
        export_mask_plot = functools.partial(self.export_image_plot, 'mask')
        exportMenu.addAction('Mask Render', export_mask_plot)
        exportMenu.addAction('Mesh Render', self.export_mesh_plot)
        exportMenu.addAction('SegMesh Render', self.export_segmesh_plot)
        exportMenu.addAction('Pose', self.export_pose)
        
        # Add set attribute menu
        setAttrMenu = mainMenu.addMenu('Set')
        set_reference_name_menu = functools.partial(self.set_attr, info="Set Reference Mesh Name", hints='ossicles')
        setAttrMenu.addAction('Set Reference', set_reference_name_menu)
        set_image_opacity_menu = functools.partial(self.set_attr, info="Set Image Opacity (range from 0 to 1)", hints='0.99')
        setAttrMenu.addAction('Set Image Opacity', set_image_opacity_menu)
        set_mask_opacity_menu = functools.partial(self.set_attr, info="Set Mask Opacity (range from 0 to 1)", hints='0.5')
        setAttrMenu.addAction('Set Mask Opacity', set_mask_opacity_menu)
        set_mesh_opacity_menu = functools.partial(self.set_attr, info="Set Mesh Opacity (range from 0 to 1)", hints='0.8')
        setAttrMenu.addAction('Set Mesh Opacity', set_mesh_opacity_menu)

        # Add camera related actions
        CameraMenu = mainMenu.addMenu('Camera')
        CameraMenu.addAction('Reset Camera (c)', self.reset_camera)
        CameraMenu.addAction('Zoom In (x)', self.zoom_in)
        CameraMenu.addAction('Zoom Out (z)', self.zoom_out)

        # Add register related actions
        RegisterMenu = mainMenu.addMenu('Regiter')
        RegisterMenu.addAction('Reset GT Pose (k)', self.reset_gt_pose)
        RegisterMenu.addAction('Update GT Pose (l)', self.update_gt_pose)
        RegisterMenu.addAction('Current Pose (t)', self.current_pose)
        RegisterMenu.addAction('Undo Pose (s)', self.undo_pose)

        # Add coloring related actions
        RegisterMenu = mainMenu.addMenu('Color')
        set_nocs_color = functools.partial(self.set_color, True)
        RegisterMenu.addAction('NOCS', set_nocs_color)
        set_latlon_color = functools.partial(self.set_color, False)
        RegisterMenu.addAction('LatLon', set_latlon_color)

        # Add pnp algorithm related actions
        PnPMenu = mainMenu.addMenu('Run')
        PnPMenu.addAction('EPnP with mesh', self.epnp_mesh)
        epnp_nocs_mask = functools.partial(self.epnp_mask, True)
        PnPMenu.addAction('EPnP with nocs mask', epnp_nocs_mask)
        epnp_latlon_mask = functools.partial(self.epnp_mask, False)
        PnPMenu.addAction('EPnP with latlon mask', epnp_latlon_mask)

        if show:
            self.plotter.set_background('#FBF0D9'); # light green shade: https://www.schemecolor.com/eye-comfort.php
            self.plotter.enable_joystick_actor_style()
            self.plotter.enable_trackball_actor_style()
            self.plotter.track_click_position(callback=self.track_click_callback, side='l')

            # camera related key bindings
            self.plotter.add_key_event('c', self.reset_camera)
            self.plotter.add_key_event('z', self.zoom_out)
            self.plotter.add_key_event('x', self.zoom_in)

            # registration related key bindings
            self.plotter.add_key_event('k', self.reset_gt_pose)
            self.plotter.add_key_event('l', self.update_gt_pose)
            self.plotter.add_key_event('t', self.current_pose)
            self.plotter.add_key_event('s', self.undo_pose)

            self.plotter.add_axes()
            self.plotter.add_camera_orientation_widget()

            self.plotter.show()
            self.show()

    def set_attr(self, info, hints):
        output, ok = self.input_dialog.getText(self, 'Input', info, text=hints)
        info = info.upper()
        if ok: 
            if 'reference'.upper() in info:
                try: self.set_reference(output)
                except AssertionError: QMessageBox.warning(self, 'vision6D', "Reference name does not exist in the paths", QMessageBox.Ok, QMessageBox.Ok)
            elif 'image opacity'.upper() in info:
                try: self.set_image_opacity(float(output), 'image')
                except AssertionError: QMessageBox.warning(self, 'vision6D', "Image opacity should range from 0 to 1", QMessageBox.Ok, QMessageBox.Ok)
            
            elif 'mask opacity'.upper() in info:
                try: self.set_image_opacity(float(output), 'mask')
                except AssertionError: QMessageBox.warning(self, 'vision6D', "Mask opacity should range from 0 to 1", QMessageBox.Ok, QMessageBox.Ok)

            elif 'mesh opacity'.upper() in info:
                try: self.set_mesh_opacity(float(output))
                except AssertionError: QMessageBox.warning(self, 'vision6D', "Mesh opacity should range from 0 to 1", QMessageBox.Ok, QMessageBox.Ok)

    def add_image_file(self, name):

        if name == 'image':
            if self.image_path == None or self.image_path == '':
                self.image_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(self.image_dir), "Files (*.png *.jpg)")
            else:
                self.image_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(pathlib.Path(self.image_path).parent), "Files (*.png *.jpg)")
            if self.image_path != '': self.add_image(self.image_path, name)
        elif name == 'mask': 
            if self.mask_path == None or self.mask_path == '':
                self.mask_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(self.mask_dir), "Files (*.png *.jpg)")
            else:
                self.mask_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(pathlib.Path(self.mask_path).parent), "Files (*.png *.jpg)")
            if self.mask_path != '': self.add_image(self.mask_path, name)

    def add_mesh_file(self):
        if self.mesh_path == None or self.mesh_path == '':
            self.mesh_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(self.mesh_dir), "Files (*.mesh *.ply)")
        else:
            self.mesh_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(pathlib.Path(self.mesh_path).parent), "Files (*.mesh *.ply)")

        if self.mesh_path != '':
            mesh_name, ok = self.input_dialog.getText(self, 'Input', 'Specify the object Class name', text='ossicles')
            if ok: 
                self.meshdict[mesh_name] = self.mesh_path
                self.add_mesh(mesh_name, self.mesh_path)
                if self.reference is None: 
                    reply = QMessageBox.question(self,"vision6D", "Do you want to make this mesh as a reference?", QMessageBox.Yes, QMessageBox.No)
                    if reply == QMessageBox.Yes: self.reference = mesh_name
      
    def add_pose_file(self):
        pose_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(self.gt_poses_dir), "Files (*.npy)")
        if pose_path != '': self.set_transformation_matrix(matrix=np.load(pose_path))
    
    def remove_actor(self, name):
        if self.reference == name: self.reference = None
        if name == 'image' or name == 'mask': 
            actor = self.image_actors[name]
            self.image_actors[name] = None
            if name == 'image': 
                self.image_path = None
            elif name == 'mask': 
                self.mask_data = None
                self.mask_path = None
        else: 
            actor = self.mesh_actors[name]
             # remove the item from the mesh dictionary
            del self.mesh_raw[name]
            del self.mesh_polydata[name]
            del self.mesh_actors[name]
            if len(self.mesh_actors) == 0:
                assert (len(self.mesh_polydata) == 0) and (len(self.mesh_raw) == 0), "self.mesh_polydata and self.mesh_raw should be empty when self.mesh_actors are empty"
                self.mesh_path = None
                self.meshdict = {}
                self.reference = None
                self.transformation_matrix = np.eye(4)
                self.undo_poses = []
                self.reset_camera()

        self.plotter.remove_actor(actor)
        actions_to_remove = [action for action in self.removeMenu.actions() if action.text() == name]
        assert len(actions_to_remove) == 1, "the actions to remove should always be 1"
        self.removeMenu.removeAction(actions_to_remove[0])
        self.remove_actors_names.remove(name)
   
    def clear_plot(self):
        
        # Clear out everything in the menu
        for action in self.removeMenu.actions():
            name = action.text()
            if name == 'image' or name == 'mask': 
                actor = self.image_actors[name]
            else: actor = self.mesh_actors[name]
            self.plotter.remove_actor(actor)
            self.removeMenu.removeAction(action)

        # Re-initial the dictionaries
        self.image_path = None
        self.mask_path = None
        self.mesh_path = None
        self.meshdict = {}

        self.reference = None
        self.transformation_matrix = np.eye(4)
        self.image_actors = {'image': None, 'mask':None}
        self.mask_data = None
        self.mesh_raw = {}
        self.mesh_polydata = {}
        self.mesh_actors = {}
        self.undo_poses = []
        self.remove_actors_names = []
        self.reset_camera()

    def export_image_plot(self, image_name):

        if self.image_actors[image_name] is None:
            if image_name == 'image': QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QMessageBox.Ok, QMessageBox.Ok)
            elif image_name == 'mask': QMessageBox.warning(self, 'vision6D', "Need to load a mask first!", QMessageBox.Ok, QMessageBox.Ok)
            return 0
        
        reply = QMessageBox.question(self,"vision6D", "Reset Camera?", QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes: camera = self.camera.copy()
        else: camera = self.plotter.camera.copy()

        self.render.clear()
        image_actor = self.image_actors[image_name].copy(deep=True)
        image_actor.GetProperty().opacity = 1
        self.render.add_actor(image_actor, pickable=False, name="image")
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)

        # obtain the rendered image
        image = self.render.last_image
        output_name = (pathlib.Path(self.image_path).stem + '.png') if image_name == 'image' else (pathlib.Path(self.mask_path).stem + '.png')
        output_path = vis.config.GITROOT / "output" / f"{image_name}" / output_name
        rendered_image = PIL.Image.fromarray(image)
        rendered_image.save(output_path)
        QMessageBox.about(self,"vision6D", f"Export to {str(output_path)}")

    def export_mesh_plot(self, reply_reset_camera=None, reply_render_mesh=None, reply_export_surface=None, msg=True):

        if self.reference is not None: 
            transformation_matrix = self.mesh_actors[self.reference].user_matrix
        else:
            QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QMessageBox.Ok, QMessageBox.Ok)
            return 0

        if reply_reset_camera is None and reply_render_mesh is None and reply_export_surface is None:
            reply_reset_camera = QMessageBox.question(self,"vision6D", "Reset Camera?", QMessageBox.Yes, QMessageBox.No)
            reply_render_mesh = QMessageBox.question(self,"vision6D", "Only render the reference mesh?", QMessageBox.Yes, QMessageBox.No)
            reply_export_surface = QMessageBox.question(self,"vision6D", "Export the mesh as surface?", QMessageBox.Yes, QMessageBox.No)
            
        if reply_reset_camera == QMessageBox.Yes: camera = self.camera.copy()
        else: camera = self.plotter.camera.copy()
        if reply_render_mesh == QMessageBox.No: render_all_meshes = True
        else: render_all_meshes = False
        if reply_export_surface == QMessageBox.No: point_clouds = True
        else: point_clouds = False
        
        self.render.clear()
        reference_name = pathlib.Path(self.meshdict[self.reference]).stem

        if self.image_actors['image'] is not None: 
            id = pathlib.Path(self.image_path).stem.split('_')[-1]
            output_name = reference_name + f'_render_{id}.png'
        else:
            output_name = reference_name + '_render.png'
            
        if render_all_meshes:
            # Render the targeting objects
            for mesh_name, mesh_data in self.mesh_polydata.items():
                colors = mesh_data.point_data.active_scalars
                if colors is None: colors = np.ones((len(mesh_data.points), 3)) * 0.5
                assert colors.shape == mesh_data.points.shape, "colors shape should be the same as mesh_data.points shape"
                mesh = self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=1, name=mesh_name) if not point_clouds else self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='points', point_size=1, render_points_as_spheres=False, opacity=1, name=mesh_name)
                mesh.user_matrix = transformation_matrix

            self.render.camera = camera
            self.render.disable(); self.render.show(auto_close=False)

            # obtain the rendered image
            image = self.render.last_image
            output_path = vis.config.GITROOT / "output" / "mesh" / output_name
            rendered_image = PIL.Image.fromarray(image)
            rendered_image.save(output_path)
            if msg: QMessageBox.about(self,"vision6D", f"Export the image to {str(output_path)}")
        else:
            mesh_name = self.reference
            mesh_data = self.mesh_polydata[mesh_name]
            colors = mesh_data.point_data.active_scalars
            if colors is None: colors = np.ones((len(mesh_data.points), 3)) * 0.5
            assert colors.shape == mesh_data.points.shape, "colors shape should be the same as mesh_data.points shape"
            mesh = self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=1, name=mesh_name) if not point_clouds else self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='points', point_size=1, render_points_as_spheres=False, opacity=1, name=mesh_name)
            mesh.user_matrix = transformation_matrix
            self.render.camera = camera
            self.render.disable(); self.render.show(auto_close=False)

            # obtain the rendered image
            image = self.render.last_image
            output_path = vis.config.GITROOT / "output" / "mesh" / output_name
            rendered_image = PIL.Image.fromarray(image)
            rendered_image.save(output_path)
            if msg: QMessageBox.about(self,"vision6D", f"Export to {str(output_path)}")

            return image

    def export_segmesh_plot(self):

        if self.reference is not None: 
            transformation_matrix = self.mesh_actors[self.reference].user_matrix
        else:
            QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QMessageBox.Ok, QMessageBox.Ok)
            return 0
        
        if self.mask_data is None: 
            QMessageBox.warning(self, 'vision6D', "Need to load a segmentation mask first", QMessageBox.Ok, QMessageBox.Ok)
            return 0

        reply_reset_camera = QMessageBox.question(self,"vision6D", "Reset Camera?", QMessageBox.Yes, QMessageBox.No)
        reply_export_surface = QMessageBox.question(self,"vision6D", "Export the mesh as surface?", QMessageBox.Yes, QMessageBox.No)

        if reply_reset_camera == QMessageBox.Yes: camera = self.camera.copy()
        else: camera = self.plotter.camera.copy()
        if reply_export_surface == QMessageBox.No: point_clouds = True
        else: point_clouds = False

        self.render.clear()
        mask_actor = self.image_actors['mask'].copy(deep=True)
        mask_actor.GetProperty().opacity = 1
        self.render.add_actor(mask_actor, pickable=False, name="mask")
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        segmask = self.render.last_image
        if np.max(segmask) > 1: segmask = segmask / 255

        self.render.clear()
        reference_name = pathlib.Path(self.meshdict[self.reference]).stem

        if self.image_actors['image'] is not None: 
            id = pathlib.Path(self.image_path).stem.split('_')[-1]
            output_name = reference_name + f'_render_{id}.png'
        else:
            output_name = reference_name + '_render.png'
        
        # Render the targeting objects
        for mesh_name, mesh_data in self.mesh_polydata.items():
            colors = mesh_data.point_data.active_scalars
            if colors is None: colors = np.ones((len(mesh_data.points), 3)) * 0.5
            assert colors.shape == mesh_data.points.shape, "colors shape should be the same as mesh_data.points shape"
            mesh = self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=1, name=mesh_name) if not point_clouds else self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='points', point_size=1, render_points_as_spheres=False, opacity=1, name=mesh_name)
            mesh.user_matrix = transformation_matrix

        self.render.camera = camera
        self.render.disable(); self.render.show(auto_close=False)

        # obtain the rendered image
        image = self.render.last_image
        image = (image * segmask).astype(np.uint8)
        output_path = vis.config.GITROOT / "output" / "segmesh" / output_name
        rendered_image = PIL.Image.fromarray(image)
        rendered_image.save(output_path)
        QMessageBox.about(self,"vision6D", f"Export the image to {str(output_path)}")
        
        return image

    def export_pose(self):
        if self.reference is None: 
            QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QMessageBox.Ok, QMessageBox.Ok)
            return 0
        
        self.update_gt_pose()

        mesh_path_name = pathlib.Path(self.mesh_path).stem.split('_')
        reference_name = mesh_path_name[0] + "_" + mesh_path_name[1] + "_" + self.reference + '_processed'

        if self.image_actors['image'] is not None: 
            id = pathlib.Path(self.image_path).stem.split('_')[-1]
            output_name = reference_name + f'_gt_pose_{id}.npy'
        else: output_name = reference_name + '_gt_pose.npy'

        output_path = vis.config.GITROOT / "output" / "gt_poses" / output_name
        np.save(output_path, self.transformation_matrix)
        QMessageBox.about(self,"vision6D", f"\nSaved:\n{self.transformation_matrix}\nExport to:\n {str(output_path)}")
