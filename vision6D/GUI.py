# General import
import os
import logging
import numpy as np
import pyvista as pv
import functools
import trimesh
import pathlib
import PIL
import ast
import json
import math

# Qt5 import
from PyQt5 import QtWidgets, QtGui
from pyvistaqt import QtInteractor, MainWindow
from PyQt5.QtCore import Qt, QPoint

# self defined package import
import vision6D as vis

np.set_printoptions(suppress=True)

class YesNoBox(QtWidgets.QMessageBox):
    def __init__(self, *args, **kwargs):
        super(YesNoBox, self).__init__(*args, **kwargs)
        self.canceled = False

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.canceled = True
        super(YesNoBox, self).closeEvent(event)

class PopUpDialog(QtWidgets.QDialog):
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

class CameraPropsInputDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, 
                    line1=(None, None), 
                    line2=(None, None), 
                    line3=(None, None), 
                    line4=(None, None), 
                    line5=(None, None),
                    line6=(None, None)):
        super().__init__(parent)

        self.args1 = QtWidgets.QLineEdit(self, text=str(line1[1]))
        self.args2 = QtWidgets.QLineEdit(self, text=str(line2[1]))
        self.args3 = QtWidgets.QLineEdit(self, text=str(line3[1]))
        self.args4 = QtWidgets.QLineEdit(self, text=str(line4[1]))
        self.args5 = QtWidgets.QLineEdit(self, text=str(line5[1]))
        self.args6 = QtWidgets.QLineEdit(self, text=str(line6[1]))

        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)

        layout = QtWidgets.QFormLayout(self)
        layout.addRow(f"{line1[0]}", self.args1)
        layout.addRow(f"{line2[0]}", self.args2)
        layout.addRow(f"{line3[0]}", self.args3)
        layout.addRow(f"{line4[0]}", self.args4)
        layout.addRow(f"{line5[0]}", self.args5)
        layout.addRow(f"{line6[0]}", self.args6)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.args1.text(), 
                self.args2.text(), 
                self.args3.text(),
                self.args4.text(),
                self.args5.text(),
                self.args6.text())
    
class GetTextDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(GetTextDialog, self).__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        
        self.setWindowTitle("Vision6D")
        self.introLabel = QtWidgets.QLabel("Input the Ground Truth Pose:")
        self.btnloadfromfile = QtWidgets.QPushButton("Load from file", self)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.introLabel)
        hbox.addWidget(self.btnloadfromfile)
        hbox.setContentsMargins(0, 0, 0, 0)
        self.hboxWidget = QtWidgets.QWidget()
        self.hboxWidget.setLayout(hbox)

        self.btnloadfromfile.clicked.connect(self.load_from_file)
        self.textEdit = QtWidgets.QTextEdit(self)
        self.textEdit.setPlainText(f"[[1, 0, 0, 0], \n[0, 1, 0, 0], \n[0, 0, 1, 0], \n[0, 0, 0, 1]]")
        self.btnSubmit = QtWidgets.QPushButton("Submit", self)
        self.btnSubmit.clicked.connect(self.submit_text)

        layout.addWidget(self.hboxWidget)
        layout.addWidget(self.textEdit)
        layout.addWidget(self.btnSubmit)

    def submit_text(self):
        self.user_text = self.textEdit.toPlainText()
        self.accept()

    def load_from_file(self):
        file_dialog = QtWidgets.QFileDialog()
        pose_path, _ = file_dialog.getOpenFileName(None, "Open file", "", "Files (*.npy)")
        if pose_path != '':
            gt_pose = np.load(pose_path)
            self.textEdit.setPlainText(f"[[{np.around(gt_pose[0, 0], 8)}, {np.around(gt_pose[0, 1], 8)}, {np.around(gt_pose[0, 2], 8)}, {np.around(gt_pose[0, 3], 8)}],\n"
                                    f"[{np.around(gt_pose[1, 0], 8)}, {np.around(gt_pose[1, 1], 8)}, {np.around(gt_pose[1, 2], 8)}, {np.around(gt_pose[1, 3], 8)}],\n"
                                    f"[{np.around(gt_pose[2, 0], 8)}, {np.around(gt_pose[2, 1], 8)}, {np.around(gt_pose[2, 2], 8)}, {np.around(gt_pose[2, 3], 8)}],\n"
                                    f"[{np.around(gt_pose[3, 0], 8)}, {np.around(gt_pose[3, 1], 8)}, {np.around(gt_pose[3, 2], 8)}, {np.around(gt_pose[3, 3], 8)}]]")

    def get_text(self):
        return self.user_text

class MyMainWindow(MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        # Set up the main window layout
        self.setWindowTitle("Vision6D")
        self.window_size = (1920, 1080)
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        self.setAcceptDrops(True)

        self.track_actors_names = []
        self.button_group_actors_names = QtWidgets.QButtonGroup(self)

        # Set panel bar
        self.set_panel_bar()
        
        # Set menu bar
        self.set_menu_bars()

        # Create the plotter
        self.create_plotter()
        
        # Set the camera
        self.fx = 50000
        self.fy = 50000
        self.cx = 960
        self.cy = 540
        self.cam_viewup = (0, -1, 0)
        self.cam_position = -500
        self.set_camera_props()

        # Set up the main layout with the left panel and the render window using QSplitter
        self.main_layout = QtWidgets.QHBoxLayout(self.main_widget)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.panel_widget)
        self.splitter.addWidget(self.plotter)
        self.main_layout.addWidget(self.splitter)

        # Add a QLabel as an overlay hint label
        self.hintLabel = QtWidgets.QLabel(self.plotter)
        self.hintLabel.setText("Drag and drop a file here...")
        self.hintLabel.setStyleSheet("""
                                    color: white; 
                                    background-color: rgba(0, 0, 0, 127); 
                                    padding: 10px;
                                    border: 2px dashed gray;
                                    """)
        self.hintLabel.setAlignment(Qt.AlignCenter)

        # Show the plotter
        self.show_plot()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
            self.hintLabel.hide()  # Hide hint when dragging
        else:
            e.ignore()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith(('.mesh', '.ply', '.stl', '.obj', '.off', '.dae', '.fbx', '.3ds', '.x3d')):  # add mesh
                self.mesh_path = file_path
                self.add_mesh_file(prompt=False)
            elif file_path.endswith(('.png', '.jpg')):  # add image/mask
                yes_no_box = YesNoBox()
                yes_no_box.setIcon(QtWidgets.QMessageBox.Question)
                yes_no_box.setWindowTitle("Vision6D")
                yes_no_box.setText("Do you want to load the image as mask?")
                yes_no_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                button_clicked = yes_no_box.exec_()
                if not yes_no_box.canceled:
                    if button_clicked == QtWidgets.QMessageBox.Yes:
                        self.mask_path = file_path
                        self.add_mask_file(prompt=False)
                    elif button_clicked == QtWidgets.QMessageBox.No:
                        self.image_path = file_path
                        self.add_image_file(prompt=False)
            elif file_path.endswith('.npy'):
                self.pose_path = file_path
                self.add_pose_file(prompt=False)
            else:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "File format is not supported!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return 0

    def resizeEvent(self, e):
        x = (self.plotter.size().width() - self.hintLabel.width()) // 2
        y = (self.plotter.size().height() - self.hintLabel.height()) // 2
        self.initialHintLabelPosition = self.hintLabel.pos()
        self.hintLabel.move(x, y)
        super().resizeEvent(e)

    # ^Main Menu
    def set_menu_bars(self):
        mainMenu = self.menuBar()
        # simple dialog to record users input info
        self.input_dialog = QtWidgets.QInputDialog()
        self.file_dialog = QtWidgets.QFileDialog()
        self.get_text_dialog = GetTextDialog()
        
        self.image_path = None
        self.mask_path = None
        self.mesh_path = None
        self.pose_path = None
        self.meshdict = {}
            
        # allow to add files
        fileMenu = mainMenu.addMenu('File')
        fileMenu.addAction('Add Workspace', self.add_workspace)
        fileMenu.addAction('Add Image', self.add_image_file)
        fileMenu.addAction('Add Mask', self.add_mask_file)
        fileMenu.addAction('Add Mesh', self.add_mesh_file)
        fileMenu.addAction('Add Pose', self.add_pose_file)
        fileMenu.addAction('Clear', self.clear_plot)

        # allow to export files
        exportMenu = mainMenu.addMenu('Export')
        exportMenu.addAction('Image Render', self.export_image_plot)
        exportMenu.addAction('Mask Render', self.export_mask_plot)
        exportMenu.addAction('Mesh Render', self.export_mesh_plot)
        exportMenu.addAction('SegMesh Render', self.export_segmesh_plot)
        exportMenu.addAction('Pose', self.export_pose)
                
        # Add camera related actions
        CameraMenu = mainMenu.addMenu('Camera')
        CameraMenu.addAction('Set Camera', self.set_camera)
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

    def set_camera_extrinsics(self):
        self.camera.SetPosition((0,0,self.cam_position))
        self.camera.SetFocalPoint((*self.camera.GetWindowCenter(),0)) # Get the camera window center
        self.camera.SetViewUp(self.cam_viewup)
    
    def set_camera_intrinsics(self):
        
        # Set camera intrinsic attribute
        self.camera_intrinsics = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
                
        # convert the principal point to window center (normalized coordinate system) and set it
        wcx = -2*(self.cx - float(self.window_size[0])/2) / self.window_size[0]
        wcy =  2*(self.cy - float(self.window_size[1])/2) / self.window_size[1]
        self.camera.SetWindowCenter(wcx, wcy) # (0,0)
        
        # Setting the view angle in degrees
        view_angle = (180 / math.pi) * (2.0 * math.atan2(self.window_size[1]/2.0, self.fx)) # or view_angle = np.degrees(2.0 * math.atan2(height/2.0, f)) # focal_length = (1080 / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
        self.camera.SetViewAngle(view_angle) # view angle should be in degrees
 
    def set_camera_props(self):
        # Set up the camera
        self.camera = pv.Camera()
        self.set_camera_intrinsics()
        self.set_camera_extrinsics()
        self.plotter.camera = self.camera.copy()

    def set_camera(self):
        dialog = CameraPropsInputDialog(
            line1=("Fx", self.fx), 
            line2=("Fy", self.fy), 
            line3=("Cx", self.cx), 
            line4=("Cy", self.cy), 
            line5=("View Up", self.cam_viewup), 
            line6=("Cam Position", self.cam_position))
        if dialog.exec():
            fx, fy, cx, cy, cam_viewup, cam_position = dialog.getInputs()
            pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup, pre_cam_position = self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position
            if not (fx == '' or fy == '' or cx == '' or cy == '' or cam_viewup == '' or cam_position == ''):
                try:
                    self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position = ast.literal_eval(fx), ast.literal_eval(fy), ast.literal_eval(cx), ast.literal_eval(cy), ast.literal_eval(cam_viewup), ast.literal_eval(cam_position)
                    self.set_camera_props()
                except:
                    self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position = pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup, pre_cam_position
                    QtWidgets.QMessageBox.warning(self, 'vision6D', "Error occured, check the format of the input values", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def set_spacing(self):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button is not None:
            if checked_button.text() == 'image':
                spacing, ok = self.input_dialog.getText(self, 'Input', "Set Spacing", text=str(self.image_spacing))
                if ok:
                    try: self.image_spacing = ast.literal_eval(spacing)
                    except: QtWidgets.QMessageBox.warning(self, 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    self.add_image(self.image_path)
            elif checked_button.text() == 'mask':
                spacing, ok = self.input_dialog.getText(self, 'Input', "Set Spacing", text=str(self.mask_spacing))
                if ok:
                    try: self.mask_spacing = ast.literal_eval(spacing)
                    except: QtWidgets.QMessageBox.warning(self, 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    self.add_mask(self.mask_path)
            else:
                spacing, ok = self.input_dialog.getText(self, 'Input', "Set Spacing", text=str(self.mesh_spacing))
                if ok:
                    actor_name = checked_button.text()
                    try: self.mesh_spacing = ast.literal_eval(spacing)
                    except: QtWidgets.QMessageBox.warning(self, 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    self.add_mesh(actor_name, self.meshdict[actor_name])
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def add_workspace(self):
        workspace_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.json)")
        if workspace_path != '':
            self.hintLabel.hide()
            with open(str(workspace_path), 'r') as f: 
                workspace = json.load(f)

            self.image_path = workspace['image_path']
            self.mask_path = workspace['mask_path']
            self.pose_path = workspace['pose_path']
            mesh_paths = workspace['mesh_path']

            self.add_image_file(prompt=False)
            self.add_mask_file(prompt=False)
            self.add_pose_file(prompt=False)

            for item in mesh_paths.items():
                mesh_name, self.mesh_path = item
                self.add_mesh_file(mesh_name=mesh_name, prompt=False)

    def add_image_file(self, prompt=True):
        if prompt:
            if self.image_path == None or self.image_path == '':
                self.image_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.png *.jpg)")
            else:
                self.image_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(pathlib.Path(self.image_path).parent), "Files (*.png *.jpg)")

        if self.image_path != '':
            self.hintLabel.hide()
            image_source = np.array(PIL.Image.open(self.image_path), dtype='uint8')
            if len(image_source.shape) == 2: image_source = image_source[..., None]
            self.add_image(image_source)
            
    def add_mask_file(self, prompt=True):
        if prompt:
            if self.mask_path == None or self.mask_path == '':
                self.mask_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.png *.jpg)")
            else:
                self.mask_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(pathlib.Path(self.mask_path).parent), "Files (*.png *.jpg)")
        
        if self.mask_path != '':
            self.hintLabel.hide()
            mask_source = np.array(PIL.Image.open(self.mask_path), dtype='uint8')
            if len(mask_source.shape) == 2: mask_source = mask_source[..., None]
            self.add_mask(mask_source)

    def add_mesh_file(self, mesh_name=None, prompt=True):
        if prompt:
            if self.mesh_path == None or self.mesh_path == '':
                self.mesh_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.mesh *.ply *.stl *.obj *.off *.dae *.fbx *.3ds *.x3d)")
            else:
                self.mesh_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(pathlib.Path(self.mesh_path).parent), "Files (*.mesh *.ply)")
        
        if self.mesh_path != '':
            self.hintLabel.hide()
            if mesh_name is None:
                mesh_name, ok = self.input_dialog.getText(self, 'Input', 'Specify the object Class name')#, text='ossicles')
                if not ok: return 0

            self.meshdict[mesh_name] = self.mesh_path
            self.mesh_opacity[mesh_name] = self.surface_opacity
            transformation_matrix = self.transformation_matrix
            if self.mirror_x: transformation_matrix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
            if self.mirror_y: transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix                   
            self.add_mesh(mesh_name, self.mesh_path, transformation_matrix)
                      
    def add_pose_file(self, prompt=True):
        if prompt:
            if self.pose_path == None or self.pose_path == '':
                self.pose_path, _ = self.file_dialog.getOpenFileName(None, "Open file", "", "Files (*.npy)")
            else:
                self.pose_path, _ = self.file_dialog.getOpenFileName(None, "Open file", str(pathlib.Path(self.pose_path).parent), "Files (*.npy)")
        
        if self.pose_path != '':
            self.hintLabel.hide()
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
            self.add_image(original_image_data)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

        if self.mask_actor is not None:
            #^ mirror the mask actor
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
                
    def remove_actor(self, button):
        name = button.text()
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
            del self.mesh_actors[name] # remove the item from the mesh dictionary
            del self.mesh_colors[name]
            del self.mesh_opacity[name]
            del self.meshdict[name]
            self.reference = None
            self.color_button.setText("Color")
            self.mesh_spacing = [1, 1, 1]

        self.plotter.remove_actor(actor)
        self.track_actors_names.remove(name)
        self.output_text.clear(); self.output_text.append(f"Remove actor: <span style='background-color:yellow; color:black;'>{name}</span>")
        # remove the button from the button group
        self.button_group_actors_names.removeButton(button)
        # remove the button from the self.button_layout widget
        self.button_layout.removeWidget(button)
        # offically delete the button
        button.deleteLater()

        # clear out the plot if there is no actor
        if self.image_actor is None and self.mask_actor is None and len(self.mesh_actors) == 0: self.clear_plot()
   
    def clear_plot(self):
        
        # Clear out everything in the remove menu
        for button in self.button_group_actors_names.buttons():
            name = button.text()
            if name == 'image': actor = self.image_actor
            elif name == 'mask': actor = self.mask_actor
            else: actor = self.mesh_actors[name]
            self.plotter.remove_actor(actor)
            # remove the button from the button group
            self.button_group_actors_names.removeButton(button)
            # remove the button from the self.button_layout widget
            self.button_layout.removeWidget(button)
            # offically delete the button
            button.deleteLater()

        self.hintLabel.show()

        # Re-initial the dictionaries
        self.image_path = None
        self.mask_path = None
        self.mesh_path = None
        self.pose_path = None
        self.meshdict = {}
        self.mesh_colors = {}
        self.mesh_opacity = {}

        self.image_spacing = [0.01, 0.01, 1]
        self.mask_spacing = [0.01, 0.01, 1]
        self.mesh_spacing = [1, 1, 1]

        self.mirror_x = False
        self.mirror_y = False

        self.reference = None
        self.transformation_matrix = np.eye(4)
        self.image_actor = None
        self.mask_actor = None
        self.mesh_actors = {}
        self.undo_poses = {}
        self.track_actors_names = []

        self.colors = ["cyan", "magenta", "yellow", "lime", "deepskyblue", "salmon", "silver", "aquamarine", "plum", "blueviolet"]
        self.used_colors = []
        self.color_button.setText("Color")

        self.output_text.clear()
        self.ignore_slider_value_change = True
        self.opacity_slider.setValue(100)
        self.ignore_slider_value_change = False

    def export_image_plot(self):

        if self.image_actor is None:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
        reply = QtWidgets.QMessageBox.question(self,"vision6D", "Reset Camera?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes: camera = self.camera.copy()
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
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Image Files (*.png)")
        if output_path: 
            rendered_image = PIL.Image.fromarray(image)
            rendered_image.save(output_path)
            self.output_text.clear()
            self.output_text.append(f"Export image render to:\n {str(output_path)}")

    def export_mask_plot(self):
        if self.mask_actor is None:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a mask first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
        reply = QtWidgets.QMessageBox.question(self,"vision6D", "Reset Camera?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes: camera = self.camera.copy()
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
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Mask Files (*.png)")
        if output_path:
            rendered_image = PIL.Image.fromarray(image)
            rendered_image.save(output_path)
            self.output_text.clear()
            self.output_text.append(f"Export mask render to:\n {str(output_path)}")

    def export_mesh_plot(self, reply_reset_camera=None, reply_render_mesh=None, reply_export_surface=None, save_render=True):

        if self.reference is None:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

        if reply_reset_camera is None and reply_render_mesh is None and reply_export_surface is None:
            reply_reset_camera = QtWidgets.QMessageBox.question(self,"vision6D", "Reset Camera?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            reply_render_mesh = QtWidgets.QMessageBox.question(self,"vision6D", "Only render the reference mesh?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            reply_export_surface = QtWidgets.QMessageBox.question(self,"vision6D", "Export the mesh as surface?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            
        if reply_reset_camera == QtWidgets.QMessageBox.Yes: camera = self.camera.copy()
        else: camera = self.plotter.camera.copy()
        if reply_render_mesh == QtWidgets.QMessageBox.No: render_all_meshes = True
        else: render_all_meshes = False
        if reply_export_surface == QtWidgets.QMessageBox.No: point_clouds = True
        else: point_clouds = False
        
        # Clear out the render
        self.render.clear()

        for mesh_name, mesh_actor in self.mesh_actors.items():
            if not render_all_meshes:
                if mesh_name != self.reference: continue
            vertices, faces = vis.utils.get_mesh_actor_vertices_faces(mesh_actor)
            mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
            colors = vis.utils.get_mesh_actor_scalars(mesh_actor)
            if colors is not None: 
                assert colors.shape == vertices.shape, "colors shape should be the same as vertices shape"
                mesh = self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=1, name=mesh_name) if not point_clouds else self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='points', point_size=1, render_points_as_spheres=False, opacity=1, name=mesh_name)
            else:
                mesh = self.render.add_mesh(mesh_data, color=self.mesh_colors[mesh_name], style='surface', opacity=1, name=mesh_name) if not point_clouds else self.render.add_mesh(mesh_data, color=self.mesh_colors[mesh_name], style='points', point_size=1, render_points_as_spheres=False, opacity=1, name=mesh_name)
            mesh.user_matrix = self.mesh_actors[self.reference].user_matrix
      
        self.render.camera = camera
        self.render.disable(); self.render.show(auto_close=False)

        # obtain the rendered image
        image = self.render.last_image

        if save_render:
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Mesh Files (*.png)")
            if output_path:
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.output_text.clear()
                self.output_text.append(f"Export reference mesh render to:\n {str(output_path)}")

        return image

    def export_segmesh_plot(self):

        if self.reference is None:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
        if self.mask_actor is None: 
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a segmentation mask first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

        reply_reset_camera = QtWidgets.QMessageBox.question(self,"vision6D", "Reset Camera?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        reply_export_surface = QtWidgets.QMessageBox.question(self,"vision6D", "Export the mesh as surface?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply_reset_camera == QtWidgets.QMessageBox.Yes: camera = self.camera.copy()
        else: camera = self.plotter.camera.copy()
        if reply_export_surface == QtWidgets.QMessageBox.No: point_clouds = True
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
                
        # Render the targeting objects
        vertices, faces = vis.utils.get_mesh_actor_vertices_faces(self.mesh_actors[self.reference])
        mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        colors = vis.utils.get_mesh_actor_scalars(self.mesh_actors[self.reference])
        if colors is not None: 
            assert colors.shape == vertices.shape, "colors shape should be the same as vertices shape"
            mesh = self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=1, name=self.reference) if not point_clouds else self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='points', point_size=1, render_points_as_spheres=False, opacity=1, name=self.reference)
        else:
            mesh = self.render.add_mesh(mesh_data, color=self.mesh_colors[self.reference], style='surface', opacity=1, name=self.reference) if not point_clouds else self.render.add_mesh(mesh_data, color=self.mesh_colors[self.reference], style='points', point_size=1, render_points_as_spheres=False, opacity=1, name=self.reference)

        mesh.user_matrix = self.mesh_actors[self.reference].user_matrix

        self.render.camera = camera
        self.render.disable(); self.render.show(auto_close=False)

        # obtain the rendered image
        image = self.render.last_image
        image = (image * segmask).astype(np.uint8)
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "SegMesh Files (*.png)")
        if output_path:
            rendered_image = PIL.Image.fromarray(image)
            rendered_image.save(output_path)
            self.output_text.clear()
            self.output_text.append(f"Export segmask render:\n to {str(output_path)}")
            
        # return image

    def export_pose(self):
        if self.reference is None: 
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        self.update_gt_pose()
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Pose Files (*.npy)")
        if output_path:
            np.save(output_path, self.transformation_matrix)
            self.output_text.clear()
            self.output_text.append(f"\nSaved:\n{self.transformation_matrix}\nExport to:\n {str(output_path)}")

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
            # self.panel_widget width changes when the panel is visiable or hiden
            self.panel_widget_width = self.panel_widget.width()
            self.panel_widget.hide()
            x = (self.plotter.size().width() + self.panel_widget_width - self.hintLabel.width()) // 2
            y = (self.plotter.size().height() - self.hintLabel.height()) // 2
            self.hintLabel.move(x, y)
        else:
            self.panel_widget.show()
            x = (self.plotter.size().width() - self.panel_widget_width - self.hintLabel.width()) // 2
            y = (self.plotter.size().height() - self.hintLabel.height()) // 2
            self.hintLabel.move(x, y)

    def add_button_actor_name(self, actor_name):
        button = QtWidgets.QPushButton(actor_name)
        button.setCheckable(True)  # Set the button to be checkable
        button.clicked.connect(lambda _, text=actor_name: self.button_actor_name_clicked(text))
        button.setChecked(True)
        button.setFixedSize(self.display.size().width(), 50)
        self.button_layout.insertWidget(0, button) # insert from the top # self.button_layout.addWidget(button)
        self.button_group_actors_names.addButton(button)
        self.button_actor_name_clicked(actor_name)

    def update_color_button_text(self, text, popup):
        self.color_button.setText(text)
        popup.close() # automatically close the popup window

    def show_color_popup(self):

        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button is None:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

        actor_name = checked_button.text()

        if actor_name not in self.mesh_actors:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Only be able to color mesh actors", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

        popup = PopUpDialog(self, on_button_click=lambda text: self.update_color_button_text(text, popup))
        button_position = self.color_button.mapToGlobal(QPoint(0, 0))
        popup.move(button_position + QPoint(self.color_button.width(), 0))
        popup.exec_()

        text = self.color_button.text()
        self.mesh_colors[actor_name] = text
        if text == 'nocs': self.set_scalar(True, actor_name)
        elif text == 'latlon': self.set_scalar(False, actor_name)
        else: self.set_color(text, actor_name)

    def remove_actors_button(self):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button is None: 
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else: self.remove_actor(checked_button)

    def opacity_value_change(self, value):
        if self.ignore_slider_value_change: return 0
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button is not None:
            actor_name = checked_button.text()
            if actor_name == 'image': self.set_image_opacity(value / 100)
            elif actor_name == 'mask': self.set_mask_opacity(value / 100)
            else: 
                self.mesh_opacity[actor_name] = value / 100
                self.set_mesh_opacity(actor_name, self.mesh_opacity[actor_name])
            self.output_text.clear()
            self.output_text.append(f"Current actor <span style='background-color:yellow; color:black;'>{actor_name}</span>'s opacity is {value / 100}")
        else:
            self.ignore_slider_value_change = True
            self.opacity_slider.setValue(100)
            self.ignore_slider_value_change = False
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        
    def panel_display(self):
        self.display = QtWidgets.QGroupBox("Console")
        display_layout = QtWidgets.QVBoxLayout()
        display_layout.setContentsMargins(10, 20, 10, 10)

        #* Create the top widgets (layout)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 10, 0, 10)

        # Create the color dropdown menu (comboBox)
        self.color_button = QtWidgets.QPushButton("Color")
        self.color_button.clicked.connect(self.show_color_popup)
        top_layout.addWidget(self.color_button, 0.5) # 1 is for the stretch factor

        # Create the color dropdown menu (comboBox)
        self.spacing_button = QtWidgets.QPushButton("Spacing")
        self.spacing_button.clicked.connect(self.set_spacing)
        top_layout.addWidget(self.spacing_button, 0.5) # 1 is for the stretch factor

        # Create the opacity slider
        self.opacity_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.opacity_slider.setSingleStep(1)
        self.ignore_slider_value_change = False 
        self.opacity_slider.valueChanged.connect(self.opacity_value_change)
        top_layout.addWidget(self.opacity_slider, 1)

        # Create the second button
        remove_button = QtWidgets.QPushButton("Remove")
        remove_button.clicked.connect(self.remove_actors_button)
        top_layout.addWidget(remove_button, 0.5)

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
        self.frame.setFixedSize(*self.window_size)
        self.plotter = QtInteractor(self.frame)
        # self.plotter.setFixedSize(*self.window_size) # but camera locate in the center instead of top left
        self.render = pv.Plotter(window_size=[self.window_size[0], self.window_size[1]], lighting=None, off_screen=True) 
        self.render.set_background('black'); 
        assert self.render.background_color == "black", "render's background need to be black"
        self.signal_close.connect(self.plotter.close)

    def show_plot(self):
        self.plotter.enable_joystick_actor_style()
        self.plotter.enable_trackball_actor_style()
        self.plotter.iren.interactor.AddObserver("LeftButtonPressEvent", self.pick_callback)

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

    def showMaximized(self):
        super(MyMainWindow, self).showMaximized()
        self.splitter.setSizes([int(self.width() * 0.2), int(self.width() * 0.8)])
