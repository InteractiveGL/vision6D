# General import
import os
import sys
import logging
import numpy as np
import pyvista as pv
import functools
import trimesh
import pathlib
import copy
import PIL
import ast

# Qt5 import
from PyQt5 import QtWidgets, QtGui
from pyvistaqt import QtInteractor, MainWindow
from PyQt5.QtWidgets import QMessageBox, QInputDialog, QFileDialog, QLineEdit, QDialogButtonBox, QFormLayout, QDialog
from PyQt5.QtCore import Qt, QPoint

# self defined package import
import vision6D as vis

np.set_printoptions(suppress=True)

class CustomDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, on_button_click=None):
        super().__init__(parent)

        self.setWindowTitle("Vision6D - Colors")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint) # Disable the question mark

        button_grid = QtWidgets.QGridLayout()
        colors = ["nocs", "cyan", "magenta", 
                "yellow", "lime", "deepskyblue", "latlon", "salmon", 
                "silver", "aquamarine", "plum", "blueviolet"]

        button_count = 0
        for i in range(2):
            for j in range(6):
                name = f"{colors[button_count]}"
                button = QtWidgets.QPushButton(name)
                button.clicked.connect(lambda _, idx=name: on_button_click(str(idx)))
                button_grid.addWidget(button, j, i)
                button_count += 1

        self.setLayout(button_grid)

class MultiInputDialog(QDialog):
    def __init__(self, parent=None, placeholder=True, line1=(None, None), line2=(None, None), line3=(None, None)):
        super().__init__(parent)

        if placeholder:
            self.args1 = QLineEdit(self, placeholderText=str(line1[1]))
            self.args2 = QLineEdit(self, placeholderText=str(line2[1]))
            self.args3 = QLineEdit(self, placeholderText=str(line3[1]))
        else:
            self.args1 = QLineEdit(self, text=str(line1[1]))
            self.args2 = QLineEdit(self, text=str(line2[1]))
            self.args3 = QLineEdit(self, text=str(line3[1]))

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow(f"{line1[0]}", self.args1)
        layout.addRow(f"{line2[0]}", self.args2)
        layout.addRow(f"{line3[0]}", self.args3)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.args1.text(), self.args2.text(), self.args3.text())

class MyMainWindow(MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        # Set up the main window layout
        self.setWindowTitle("Vision6D")
        self.window_size = (1920, 1080)
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)

        self.track_actors_names = []
        self.button_group_track_actors_names = QtWidgets.QButtonGroup(self)

        # Set panel bar
        self.set_panel_bar()
        
        # Set menu bar
        self.set_menu_bars()

        # Create the plotter
        self.create_plotter()

        # Set up the main layout with the left panel and the render window using QSplitter
        self.main_layout = QtWidgets.QHBoxLayout(self.main_widget)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.panel_widget)
        self.splitter.addWidget(self.plotter)
        self.main_layout.addWidget(self.splitter)

        # Show the plotter
        self.show_plot()
            
    # ^Main Menu
    def set_menu_bars(self):
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
        self.pose_path = None
        self.meshdict = {}
        
        os.makedirs(vis.config.GITROOT / "output", exist_ok=True)
        os.makedirs(vis.config.GITROOT / "output" / "image", exist_ok=True)
        os.makedirs(vis.config.GITROOT / "output" / "mask", exist_ok=True)
        os.makedirs(vis.config.GITROOT / "output" / "mesh", exist_ok=True)
        os.makedirs(vis.config.GITROOT / "output" / "segmesh", exist_ok=True)
        os.makedirs(vis.config.GITROOT / "output" / "gt_poses", exist_ok=True)
            
        # allow to add files
        fileMenu = mainMenu.addMenu('File')
        fileMenu.addAction('Add Image', self.add_image_file)
        fileMenu.addAction('Add Mask', self.add_mask_file)
        fileMenu.addAction('Add Mesh', self.add_mesh_file)
        fileMenu.addAction('Add Pose', self.add_pose_file)
        fileMenu.addAction('Set Image/Mask Spacing', self.set_image_spacing_attr)
        fileMenu.addAction('Clear', self.clear_plot)

        # allow to export files
        exportMenu = mainMenu.addMenu('Export')
        exportMenu.addAction('Image Render', self.export_image_plot)
        exportMenu.addAction('Mask Render', self.export_mask_plot)
        exportMenu.addAction('Mesh Render', self.export_mesh_plot)
        exportMenu.addAction('SegMesh Render', self.export_segmesh_plot)
        exportMenu.addAction('Pose', self.export_pose)
        
        # Add set attribute menu
        setAttrMenu = mainMenu.addMenu('Set')
        setAttrMenu.addAction('Set Opacity (bn, hj, yu)', self.set_opacity_attr)
                
        # Add camera related actions
        CameraMenu = mainMenu.addMenu('Camera')
        CameraMenu.addAction('Config Camera', self.set_camera_attr)
        CameraMenu.addAction('Reset Camera (c)', self.reset_camera)
        CameraMenu.addAction('Zoom In (x)', self.zoom_in)
        CameraMenu.addAction('Zoom Out (z)', self.zoom_out)

        # add mirror actors related actions
        mirrorMenu = mainMenu.addMenu('Mirror')
        mirror_x = functools.partial(self.mirror_actors, direction='x')
        mirrorMenu.addAction('Mirror X axis', mirror_x)
        mirror_y = functools.partial(self.mirror_actors, direction='y')
        mirrorMenu.addAction('Mirror Y axis', mirror_y)
        
        # Add register related actions
        RegisterMenu = mainMenu.addMenu('Register')
        RegisterMenu.addAction('Reset GT Pose (k)', self.reset_gt_pose)
        RegisterMenu.addAction('Update GT Pose (l)', self.update_gt_pose)
        RegisterMenu.addAction('Current Pose (t)', self.current_pose)
        RegisterMenu.addAction('Undo Pose (s)', self.undo_pose)

        # Add pnp algorithm related actions
        PnPMenu = mainMenu.addMenu('Run')
        PnPMenu.addAction('EPnP with mesh', self.epnp_mesh)
        epnp_nocs_mask = functools.partial(self.epnp_mask, True)
        PnPMenu.addAction('EPnP with nocs mask', epnp_nocs_mask)
        epnp_latlon_mask = functools.partial(self.epnp_mask, False)
        PnPMenu.addAction('EPnP with latlon mask', epnp_latlon_mask)

    def set_camera_attr(self):
        dialog = MultiInputDialog(line1=("Focal Length", self.focal_length), line2=("View Up", self.cam_viewup), line3=("Cam Position", self.cam_position))
        if dialog.exec():
            focal_length, cam_viewup, cam_position = dialog.getInputs()
            pre_focal_length, pre_cam_viewup, pre_cam_position = self.focal_length, self.cam_viewup, self.cam_position
            if not (focal_length == '' or cam_viewup == '' or cam_position == ''):
                try:
                    self.focal_length, self.cam_viewup, self.cam_position = ast.literal_eval(focal_length), ast.literal_eval(cam_viewup), ast.literal_eval(cam_position)
                    self.set_camera_props(self.focal_length, self.cam_viewup, self.cam_position)
                except:
                    self.focal_length, self.cam_viewup, self.cam_position = pre_focal_length, pre_cam_viewup, pre_cam_position
                    QMessageBox.warning(self, 'vision6D', "Error occured, check the format of the input values", QMessageBox.Ok, QMessageBox.Ok)

    def set_opacity_attr(self):
        dialog = MultiInputDialog(placeholder=False, line1=("Image Opacity", self.image_opacity), line2=("Mask Opacity", self.mask_opacity), line3=("Mesh Opacity", self.surface_opacity))
        if dialog.exec():
            image_opacity, mask_opacity, surface_opacity = dialog.getInputs()
            pre_image_opacity, pre_mask_opacity, pre_surface_opacity = self.image_opacity, self.mask_opacity, self.surface_opacity
            if not (image_opacity == '' or mask_opacity == '' or surface_opacity == ''):
                try:
                    self.image_opacity, self.mask_opacity, self.surface_opacity = ast.literal_eval(image_opacity), ast.literal_eval(mask_opacity), ast.literal_eval(surface_opacity)
                    try:
                        self.set_image_opacity(self.image_opacity)
                    except AssertionError:
                        self.image_opacity = pre_image_opacity
                        QMessageBox.warning(self, 'vision6D', "Image opacity should range from 0 to 1", QMessageBox.Ok, QMessageBox.Ok)
                    try: 
                        self.set_mask_opacity(self.mask_opacity)
                    except AssertionError: 
                        self.mask_opacity = pre_mask_opacity
                        QMessageBox.warning(self, 'vision6D', "Mask opacity should range from 0 to 1", QMessageBox.Ok, QMessageBox.Ok)
                    try: 
                        self.set_mesh_opacity(self.surface_opacity)
                    except AssertionError: 
                        self.surface_opacity = pre_surface_opacity
                        QMessageBox.warning(self, 'vision6D', "Mesh opacity should range from 0 to 1", QMessageBox.Ok, QMessageBox.Ok)
                except:
                    self.image_opacity, self.mask_opacity, self.surface_opacity = pre_image_opacity, pre_mask_opacity, pre_surface_opacity
                    QMessageBox.warning(self, 'vision6D', "Error occured, check the format of the input values", QMessageBox.Ok, QMessageBox.Ok)
           
    def set_image_spacing_attr(self):
        spacing, ok = self.input_dialog.getText(self, 'Input', "Set Image/Mask Spacing", text=str(self.spacing))
        if ok: 
            try: 
                self.spacing = ast.literal_eval(spacing)
                if self.image_path: self.add_image(self.image_path)
                if self.mask_path: self.add_mask(self.mask_path)
            except: 
                QMessageBox.warning(self, 'vision6D', "Spacing format is not correct", QMessageBox.Ok, QMessageBox.Ok)

    def add_image_file(self):
        if self.image_path == None or self.image_path == '':
            self.image_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(self.image_dir), "Files (*.png *.jpg)")
        else:
            self.image_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(pathlib.Path(self.image_path).parent), "Files (*.png *.jpg)")
        if self.image_path != '': 
            image_source = np.array(PIL.Image.open(self.image_path), dtype='uint8')
            if len(image_source.shape) == 2: image_source = image_source[..., None]
            if self.mirror_x: image_source = image_source[:, ::-1, :]
            if self.mirror_y: image_source = image_source[::-1, :, :]
            self.add_image(image_source)
            
    def add_mask_file(self):
        if self.mask_path == None or self.mask_path == '':
            self.mask_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(self.mask_dir), "Files (*.png *.jpg)")
        else:
            self.mask_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(pathlib.Path(self.mask_path).parent), "Files (*.png *.jpg)")
        if self.mask_path != '':
            mask_source = np.array(PIL.Image.open(self.mask_path), dtype='uint8')
            if len(mask_source.shape) == 2: mask_source = mask_source[..., None]
            if self.mirror_x: mask_source = mask_source[:, ::-1, :]
            if self.mirror_y: mask_source = mask_source[::-1, :, :]
            self.add_mask(mask_source)

    def add_mesh_file(self):
        if self.mesh_path == None or self.mesh_path == '':
            self.mesh_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(self.mesh_dir), "Files (*.mesh *.ply)")
        else:
            self.mesh_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(pathlib.Path(self.mesh_path).parent), "Files (*.mesh *.ply)")

        if self.mesh_path != '':
            mesh_name, ok = self.input_dialog.getText(self, 'Input', 'Specify the object Class name', text='ossicles')
            if ok: 
                self.meshdict[mesh_name] = self.mesh_path
                mesh_source = vis.utils.load_trimesh(self.mesh_path)

                transformation_matrix = self.transformation_matrix
                if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix                   
                self.add_mesh(mesh_name, mesh_source, transformation_matrix)
                      
    def add_pose_file(self):
        self.pose_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(self.gt_poses_dir), "Files (*.npy)")
        if self.pose_path != '': 
            transformation_matrix = np.load(self.pose_path)
            self.transformation_matrix = transformation_matrix
            if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            self.add_pose(matrix=transformation_matrix)
    
    def mirror_actors(self, direction):

        if direction == 'x': mirror_x = True; mirror_y = False
        elif direction == 'y': mirror_x = False; mirror_y = True

        #^ mirror the image actor
        if self.image_actor is not None:
            original_image_data = np.array(PIL.Image.open(self.image_path), dtype='uint8')
            if len(original_image_data.shape) == 2: original_image_data = original_image_data[..., None]
            curr_image_data = vis.utils.get_image_mask_actor_scalars(self.image_actor)
            if mirror_x: curr_image_data = curr_image_data[:, ::-1, :]
            if mirror_y: curr_image_data = curr_image_data[::-1, :, :]
            if (curr_image_data == original_image_data).all(): 
                self.mirror_x = False
                self.mirror_y = False
            elif (curr_image_data == original_image_data[:, ::-1, :]).all(): 
                self.mirror_x = True
                self.mirror_y = False
            elif (curr_image_data == original_image_data[::-1, :, :]).all():
                self.mirror_x = False
                self.mirror_y = True
            elif (curr_image_data == original_image_data[:, ::-1, :][::-1, :, :]).all():
                self.mirror_x = True
                self.mirror_y = True
            self.add_image(curr_image_data)
        else:
            QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QMessageBox.Ok, QMessageBox.Ok)
            return 0

        if self.mask_actor is not None:
            #^ mirror the mask actor
            curr_mask_data = vis.utils.get_image_mask_actor_scalars(self.mask_actor)
            if mirror_x: curr_mask_data = curr_mask_data[:, ::-1, :]
            if mirror_y: curr_mask_data = curr_mask_data[::-1, :, :]
            self.add_mask(curr_mask_data)

        #^ mirror the mesh actors
        if len(self.mesh_actors) != 0:
            for actor_name, actor in self.mesh_actors.items():
                transformation_matrix = self.mesh_actors[actor_name].user_matrix
                if mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                if mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
                actor.user_matrix = transformation_matrix
                self.plotter.add_actor(actor, pickable=True, name=actor_name)

    def remove_actor(self, button):
        name = button.text()
        self.reference = None
        if name == 'image': 
            actor = self.image_actor
            self.image_actor = None
            self.image_path = None
        elif name == 'mask':
            actor = self.mask_actor
            self.mask_actor = None
            self.mask_path = None
        else: 
            actor = self.mesh_actors[name]
             # remove the item from the mesh dictionary
            del self.mesh_actors[name]
            if len(self.mesh_actors) == 0:
                self.mesh_path = None
                self.pose_path = None
                self.meshdict = {}
                self.reference = None
                self.transformation_matrix = np.eye(4)
                self.undo_poses = []
                self.colors = ["cyan", "magenta", "yellow", "lime", "deepskyblue", "salmon", "silver", "aquamarine", "plum", "blueviolet"]
                self.used_colors = []

        self.plotter.remove_actor(actor)
        self.track_actors_names.remove(name)
        # remove the button from the button group
        self.button_group_track_actors_names.removeButton(button)
        # remove the button from the self.button_layout widget
        self.button_layout.removeWidget(button)
        # offically delete the button
        button.deleteLater()
   
    def clear_plot(self):
        
        # Clear out everything in the remove menu
        for button in self.button_group_track_actors_names.buttons():
            name = button.text()
            if name == 'image': actor = self.image_actor
            elif name == 'mask': actor = self.mask_actor
            else: actor = self.mesh_actors[name]
            self.plotter.remove_actor(actor)
            # remove the button from the button group
            self.button_group_track_actors_names.removeButton(button)
            # remove the button from the self.button_layout widget
            self.button_layout.removeWidget(button)
            # offically delete the button
            button.deleteLater()

        # Re-initial the dictionaries
        self.image_path = None
        self.mask_path = None
        self.mesh_path = None
        self.pose_path = None
        self.meshdict = {}

        self.mirror_x = False
        self.mirror_y = False

        self.reference = None
        self.transformation_matrix = np.eye(4)
        self.image_actor = None
        self.mask_actor = None
        self.mesh_actors = {}
        self.undo_poses = []
        self.track_actors_names = []

        self.colors = ["cyan", "magenta", "yellow", "lime", "deepskyblue", "salmon", "silver", "aquamarine", "plum", "blueviolet"]
        self.used_colors = []

    def export_image_plot(self):

        if self.image_actor is None:
            QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QMessageBox.Ok, QMessageBox.Ok)
            return 0
        
        reply = QMessageBox.question(self,"vision6D", "Reset Camera?", QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes: camera = self.camera.copy()
        else: camera = self.plotter.camera.copy()

        self.render.clear()
        image_actor = self.image_actor.copy(deep=True)
        image_actor.GetProperty().opacity = 1
        self.render.add_actor(image_actor, pickable=False, name="image")
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)

        # obtain the rendered image
        image = self.render.last_image
        mirror = np.any((self.mirror_x, self.mirror_y))
        output_name = pathlib.Path(self.image_path).stem if not mirror else pathlib.Path(self.image_path).stem + "_mirrored"
        output_path = vis.config.GITROOT / "output" / "image" / (output_name + '.png')
        rendered_image = PIL.Image.fromarray(image)
        rendered_image.save(output_path)
        self.output_text.clear(); self.output_text.append(f"Export image render to:\n {str(output_path)}")

    def export_mask_plot(self):
        if self.mask_actor is None:
            QMessageBox.warning(self, 'vision6D', "Need to load a mask first!", QMessageBox.Ok, QMessageBox.Ok)
            return 0
        
        reply = QMessageBox.question(self,"vision6D", "Reset Camera?", QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes: camera = self.camera.copy()
        else: camera = self.plotter.camera.copy()

        self.render.clear()
        mask_actor = self.mask_actor.copy(deep=True)
        mask_actor.GetProperty().opacity = 1
        self.render.add_actor(mask_actor, pickable=False, name="mask")
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)

        # obtain the rendered image
        image = self.render.last_image
        mirror = np.any((self.mirror_x, self.mirror_y))
        output_name = pathlib.Path(self.mask_path).stem if not mirror else pathlib.Path(self.mask_path).stem + "_mirrored"
        output_path = vis.config.GITROOT / "output" / "mask" / (output_name + '.png')
        rendered_image = PIL.Image.fromarray(image)
        rendered_image.save(output_path)
        self.output_text.clear(); self.output_text.append(f"Export mask render to:\n {str(output_path)}")

    def export_mesh_plot(self, reply_reset_camera=None, reply_render_mesh=None, reply_export_surface=None, msg=True, save_render=True):

        if self.reference is not None: transformation_matrix = self.mesh_actors[self.reference].user_matrix
        else: QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QMessageBox.Ok, QMessageBox.Ok); return 0

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

        mirror = np.any((self.mirror_x, self.mirror_y))
        if self.image_actor is not None: 
            id = pathlib.Path(self.image_path).stem.split('_')[-1]
            output_name = reference_name + f'_render_{id}' if not mirror else reference_name + f'_mirrored_render_{id}'
        else:
            output_name = reference_name + '_render' if not mirror else reference_name + '_mirrored_render'
            
        if render_all_meshes:
            # Render the targeting objects
            for mesh_name, mesh_actor in self.mesh_actors.items():
                vertices, faces = vis.utils.get_mesh_actor_vertices_faces(mesh_actor)
                colors = vis.utils.get_mesh_actor_scalars(mesh_actor)
                if colors is None: colors = np.ones((len(vertices), 3)) * 0.5
                assert colors.shape == vertices.shape, "colors shape should be the same as vertices shape"
                
                mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
                mesh = self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=1, name=mesh_name) if not point_clouds else self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='points', point_size=1, render_points_as_spheres=False, opacity=1, name=mesh_name)
                mesh.user_matrix = transformation_matrix

            self.render.camera = camera
            self.render.disable(); self.render.show(auto_close=False)

            # obtain the rendered image
            image = self.render.last_image
            output_path = vis.config.GITROOT / "output" / "mesh" / (output_name + ".png")
            rendered_image = PIL.Image.fromarray(image)
            rendered_image.save(output_path)
            if msg: self.output_text.clear(); self.output_text.append(f"Export all meshes render to:\n {str(output_path)}")
        else:
            mesh_name = self.reference
            mesh_actor = self.mesh_actors[mesh_name]

            vertices, faces = vis.utils.get_mesh_actor_vertices_faces(mesh_actor)
            colors = vis.utils.get_mesh_actor_scalars(mesh_actor)
            if colors is None: colors = np.ones((len(vertices), 3)) * 0.5
            assert colors.shape == vertices.shape, "colors shape should be the same as vertices shape"
            
            mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
            mesh = self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=1, name=mesh_name) if not point_clouds else self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='points', point_size=1, render_points_as_spheres=False, opacity=1, name=mesh_name)
            mesh.user_matrix = transformation_matrix
            self.render.camera = camera
            self.render.disable(); self.render.show(auto_close=False)

            # obtain the rendered image
            image = self.render.last_image
            if save_render:
                output_path = vis.config.GITROOT / "output" / "mesh" / (output_name + ".png")
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
            if msg: self.output_text.clear(); self.output_text.append(f"Export reference mesh render to:\n {str(output_path)}")

            return image

    def export_segmesh_plot(self):

        if self.reference is not None: 
            transformation_matrix = self.mesh_actors[self.reference].user_matrix
        else:
            QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QMessageBox.Ok, QMessageBox.Ok)
            return 0
        
        if self.mask_actor is None: 
            QMessageBox.warning(self, 'vision6D', "Need to load a segmentation mask first", QMessageBox.Ok, QMessageBox.Ok)
            return 0

        reply_reset_camera = QMessageBox.question(self,"vision6D", "Reset Camera?", QMessageBox.Yes, QMessageBox.No)
        reply_export_surface = QMessageBox.question(self,"vision6D", "Export the mesh as surface?", QMessageBox.Yes, QMessageBox.No)

        if reply_reset_camera == QMessageBox.Yes: camera = self.camera.copy()
        else: camera = self.plotter.camera.copy()
        if reply_export_surface == QMessageBox.No: point_clouds = True
        else: point_clouds = False

        self.render.clear()
        mask_actor = self.mask_actor.copy(deep=True)
        mask_actor.GetProperty().opacity = 1
        self.render.add_actor(mask_actor, pickable=False, name="mask")
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        segmask = self.render.last_image
        if np.max(segmask) > 1: segmask = segmask / 255

        self.render.clear()
        reference_name = pathlib.Path(self.meshdict[self.reference]).stem

        mirror = np.any((self.mirror_x, self.mirror_y))
        if self.image_actor is not None: 
            id = pathlib.Path(self.image_path).stem.split('_')[-1]
            output_name = reference_name + f'_render_{id}' if not mirror else reference_name + f'_mirrored_render_{id}'
        else:
            output_name = reference_name + '_render' if not mirror else reference_name + '_mirrored_render'
        
        # Render the targeting objects
        for mesh_name, mesh_actor in self.mesh_actors.items():
            vertices, faces = vis.utils.get_mesh_actor_vertices_faces(mesh_actor)
            colors = vis.utils.get_mesh_actor_scalars(mesh_actor)
            if colors is None: colors = np.ones((len(vertices), 3)) * 0.5
            assert colors.shape == vertices.shape, "colors shape should be the same as vertices shape"
           
            mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
            mesh = self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=1, name=mesh_name) if not point_clouds else self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='points', point_size=1, render_points_as_spheres=False, opacity=1, name=mesh_name)
            mesh.user_matrix = transformation_matrix

        self.render.camera = camera
        self.render.disable(); self.render.show(auto_close=False)

        # obtain the rendered image
        image = self.render.last_image
        image = (image * segmask).astype(np.uint8)
        output_path = vis.config.GITROOT / "output" / "segmesh" / (output_name + ".png")
        rendered_image = PIL.Image.fromarray(image)
        rendered_image.save(output_path)
        self.output_text.clear(); self.output_text.append(f"Export segmask render:\n to {str(output_path)}")
        
        return image

    def export_pose(self):
        if self.reference is None: 
            QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QMessageBox.Ok, QMessageBox.Ok)
            return 0
        
        self.update_gt_pose()

        mesh_path_name = pathlib.Path(self.mesh_path).stem.split('_')
        
        mirror = np.any((self.mirror_x, self.mirror_y))
        if self.image_actor is not None: 
            id = pathlib.Path(self.image_path).stem.split('_')[-1]
            output_name = "_".join(mesh_path_name[:2]) + f'_gt_pose_{id}' if not mirror else "_".join(mesh_path_name[:2]) + f'_mirrored_gt_pose_{id}'
        else: output_name = "_".join(mesh_path_name[:2]) + '_gt_pose' if not mirror else "_".join(mesh_path_name[:2]) + f'_mirrored_gt_pose'

        output_path = vis.config.GITROOT / "output" / "gt_poses" / (output_name + ".npy")
        np.save(output_path, self.transformation_matrix)
        self.output_text.clear(); self.output_text.append(f"\nSaved:\n{self.transformation_matrix}\nExport to:\n {str(output_path)}")

    # ^Panel
    def set_panel_bar(self):
        # Create a left panel layout
        self.panel_widget = QtWidgets.QWidget()
        self.panel_layout = QtWidgets.QVBoxLayout(self.panel_widget)

        # Create a top panel bar with a toggle button
        self.panel_bar = QtWidgets.QMenuBar()
        self.toggle_action = QtWidgets.QAction("Panel", self)
        self.toggle_action.triggered.connect(self.toggle_panel)
        self.panel_bar.addAction(self.toggle_action)
        self.setMenuBar(self.panel_bar)

        self.panel_display()
        self.panel_output()
        
        # Set the stretch factor for each section to be equal
        self.panel_layout.setStretchFactor(self.display, 1)
        self.panel_layout.setStretchFactor(self.output, 1)

    def toggle_panel(self):
        if self.panel_widget.isVisible():
            self.panel_widget.hide()
        else:
            self.panel_widget.show()
        
    def add_button_actor_name(self, actor_name):
        button = QtWidgets.QPushButton(actor_name)
        button.setCheckable(True)  # Set the button to be checkable
        button.clicked.connect(lambda checked, text=actor_name: self.set_mesh_reference(text))
        button.setFixedSize(self.display.size().width(), 50)
        self.button_layout.insertWidget(0, button) # insert from the top # self.button_layout.addWidget(button)
        self.button_group_track_actors_names.addButton(button)

    def update_color_button_text(self, text, popup):
        self.color_button.setText(text)
        popup.close() # automatically close the popup window

    def show_color_popup(self):

        checked_button = self.button_group_track_actors_names.checkedButton()
        if checked_button is None:
            QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QMessageBox.Ok, QMessageBox.Ok)
            return 0

        if checked_button.text() == "image" or checked_button.text() == "mask":
            QMessageBox.warning(self, 'vision6D', "Only be able to color mesh actors", QMessageBox.Ok, QMessageBox.Ok)
            return 0

        popup = CustomDialog(self, on_button_click=lambda text: self.update_color_button_text(text, popup))
        button_position = self.color_button.mapToGlobal(QPoint(0, 0))
        popup.move(button_position + QPoint(self.color_button.width(), 0))
        popup.exec_()

        text = self.color_button.text()
        if text == 'nocs': self.set_scalar(True, checked_button.text())
        elif text == 'latlon': self.set_scalar(False, checked_button.text())
        else: self.set_color(text, checked_button.text())

    def remove_actors_button(self):
        checked_button = self.button_group_track_actors_names.checkedButton()
        if checked_button is None: 
            QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QMessageBox.Ok, QMessageBox.Ok)
        else: self.remove_actor(checked_button)

    def panel_display(self):
        self.display = QtWidgets.QGroupBox("Console")
        display_layout = QtWidgets.QVBoxLayout()
        display_layout.setContentsMargins(10, 20, 10, 10)

        #* Create the top widgets (layout)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 10, 0, 10)

        # Create the color dropdown menu (comboBox)
        self.color_button = QtWidgets.QPushButton("Select Color")
        self.color_button.clicked.connect(self.show_color_popup)
        top_layout.addWidget(self.color_button, 1) # 1 is for the stretch factor

        # Create the slider
        slider = QtWidgets.QSlider(Qt.Horizontal)
        slider.setMinimum(0.01)
        slider.setMaximum(9.99)
        slider.setValue(8)
        slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        slider.setTickInterval(1)
        top_layout.addWidget(slider, 2)

        # Create the second button
        remove_button = QtWidgets.QPushButton("Remove")
        remove_button.clicked.connect(self.remove_actors_button)
        top_layout.addWidget(remove_button, 1)

        display_layout.addLayout(top_layout)

        #* Create the bottom widgets
        actor_widget = QtWidgets.QLabel("Actors")
        display_layout.addWidget(actor_widget)

        # Create a scroll area for the buttons
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        display_layout.addWidget(scroll_area)

        # Create a container widget for the buttons
        button_container = QtWidgets.QWidget()
        self.button_layout = QtWidgets.QVBoxLayout()
        self.button_layout.setSpacing(0)  # Remove spacing between buttons
        button_container.setLayout(self.button_layout)

        self.button_layout.addStretch()

        # Set the container widget as the scroll area's widget
        scroll_area.setWidget(button_container)

        self.display.setLayout(display_layout)
        self.panel_layout.addWidget(self.display)

    def panel_output(self):
        # Add a spacer to the top of the main layout
        self.output = QtWidgets.QGroupBox("Output")
        output_layout = QtWidgets.QVBoxLayout()
        output_layout.setContentsMargins(10, 20, 10, 10)

        self.output_text = QtWidgets.QTextEdit()
        self.output_text.setReadOnly(True)
        output_layout.addWidget(self.output_text)
        self.output.setLayout(output_layout)
        self.panel_layout.addWidget(self.output)

    #^ Show plot
    def create_plotter(self):
        self.frame = QtWidgets.QFrame()
        self.plotter = QtInteractor(self.frame)
        # self.plotter.setFixedSize(*self.window_size) # but camera locate in the center instead of top left
        self.render = pv.Plotter(window_size=[self.window_size[0], self.window_size[1]], lighting=None, off_screen=True) 
        self.render.set_background('black'); 
        assert self.render.background_color == "black", "render's background need to be black"
        self.signal_close.connect(self.plotter.close)

    def show_plot(self):
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

        # opacity related key bindings
        toggle_image_opacity_up = functools.partial(self.toggle_image_opacity, up=True)
        self.plotter.add_key_event('b', toggle_image_opacity_up)
        toggle_image_opacity_down = functools.partial(self.toggle_image_opacity, up=False)
        self.plotter.add_key_event('n', toggle_image_opacity_down)

        toggle_mask_opacity_up = functools.partial(self.toggle_mask_opacity, up=True)
        self.plotter.add_key_event('h', toggle_mask_opacity_up)
        toggle_mask_opacity_down = functools.partial(self.toggle_mask_opacity, up=False)
        self.plotter.add_key_event('j', toggle_mask_opacity_down)
        
        toggle_surface_opacity_up = functools.partial(self.toggle_surface_opacity, up=True)
        self.plotter.add_key_event('y', toggle_surface_opacity_up)
        toggle_surface_opacity_down = functools.partial(self.toggle_surface_opacity, up=False)
        self.plotter.add_key_event('u', toggle_surface_opacity_down)

        self.plotter.add_axes()
        self.plotter.add_camera_orientation_widget()

        self.plotter.show()
        self.show()

    def showMaximized(self):
        super(MyMainWindow, self).showMaximized()
        self.splitter.setSizes([int(self.width() * 0.2), int(self.width() * 0.8)])
