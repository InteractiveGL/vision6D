import ast
import copy

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QPoint
import numpy as np
import trimesh
import pyvista as pv

from ... import utils
from ...stores import PvQtStore, PathsStore, QtStore, MainStore
from ...widgets import GetTextDialog, CameraPropsInputDialog, PopUpDialog, LabelWindow

class DisplayPanel:
    
    def __init__(self, display):

        # Save references
        self.display = display

        # Create a reference to the store
        self.main_store = MainStore()
        self.pvqt_store = PvQtStore()
        self.paths_store = PathsStore()
        self.qt_store = QtStore()

        display_layout = QtWidgets.QVBoxLayout()
        display_layout.setContentsMargins(10, 15, 10, 0)

        #* Create the top widgets (layout)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Create Grid layout for function buttons
        top_grid_layout = QtWidgets.QGridLayout()

        # Create the set camera button
        set_camera_button = QtWidgets.QPushButton("Set Camera")
        set_camera_button.clicked.connect(self.set_camera)
        top_grid_layout.addWidget(set_camera_button, 0, 0)

        # Create the actor pose button
        actor_pose_button = QtWidgets.QPushButton("Set Pose")
        actor_pose_button.clicked.connect(self.set_pose)
        top_grid_layout.addWidget(actor_pose_button, 0, 1)

        # Create the draw mask button
        draw_mask_button = QtWidgets.QPushButton("Draw Mask")
        draw_mask_button.clicked.connect(self.pvqt_store.mask_store.draw_mask)
        top_grid_layout.addWidget(draw_mask_button, 0, 2)

        # Create the video related button
        
        top_grid_layout.addWidget(self.qt_store.play_video_button, 0, 3)

        self.sample_video_button = QtWidgets.QPushButton("Sample Video")
        self.sample_video_button.clicked.connect(self.pvqt_store.video_store.sample_video)
        top_grid_layout.addWidget(self.sample_video_button, 1, 0)

        self.save_frame_button = QtWidgets.QPushButton("Save Frame")
        self.save_frame_button.clicked.connect(lambda _, save=True: self.pvqt_store.video_store.load_per_frame_info(save))
        top_grid_layout.addWidget(self.save_frame_button, 1, 1)

        self.prev_frame_button = QtWidgets.QPushButton("Prev Frame")
        self.prev_frame_button.clicked.connect(self.pvqt_store.video_store.prev_frame)
        top_grid_layout.addWidget(self.prev_frame_button, 1, 2)

        self.next_frame_button = QtWidgets.QPushButton("Next Frame")
        self.next_frame_button.clicked.connect(self.pvqt_store.video_store.next_frame)
        top_grid_layout.addWidget(self.next_frame_button, 1, 3)

        top_grid_widget = QtWidgets.QWidget()
        top_grid_widget.setLayout(top_grid_layout)
        top_layout.addWidget(top_grid_widget)
        display_layout.addLayout(top_layout)

        #* Create the bottom widgets
        actor_widget = QtWidgets.QLabel("Actors")
        display_layout.addWidget(actor_widget)

        actor_grid_layout = QtWidgets.QGridLayout()
        actor_grid_layout.setContentsMargins(10, 15, 10, 0)

        # Create the color dropdown menu
        self.qt_store.color_button.clicked.connect(self.show_color_popup)
        actor_grid_layout.addWidget(self.qt_store.color_button, 0, 0)
        
        # Create the opacity spinbox
        actor_grid_layout.addWidget(self.qt_store.opacity_spinbox, 0, 1)

        # Create the hide button
        self.toggle_hide_meshes_flag = False
        hide_button = QtWidgets.QPushButton("toggle meshes")
        hide_button.clicked.connect(self.toggle_hide_meshes_button)
        actor_grid_layout.addWidget(hide_button, 0, 2)

        # Create the remove button
        remove_button = QtWidgets.QPushButton("Remove Actor")
        remove_button.clicked.connect(self.remove_actors_button)
        actor_grid_layout.addWidget(remove_button, 0, 3)
        display_layout.addLayout(actor_grid_layout)

        # Create the spacing button
        self.spacing_button = QtWidgets.QPushButton("Spacing")
        self.spacing_button.clicked.connect(self.set_spacing)
        actor_grid_layout.addWidget(self.spacing_button, 1, 0)

        # Create a scroll area for the buttons
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        display_layout.addWidget(scroll_area)
        self.display.setLayout(display_layout)

        # Create a container widget for the buttons
        button_container = QtWidgets.QWidget()
        button_container.setLayout(self.qt_store.button_layout)

        # Set the container widget as the scroll area's widget
        scroll_area.setWidget(button_container)

        # change image opacity key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("b"), self.qt_store.main_window).activated.connect(lambda up=True: self.toggle_image_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("n"), self.qt_store.main_window).activated.connect(lambda up=False: self.toggle_image_opacity(up))

        # change mask opacity key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("g"), self.qt_store.main_window).activated.connect(lambda up=True: self.toggle_mask_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("h"), self.qt_store.main_window).activated.connect(lambda up=False: self.toggle_mask_opacity(up))

        # change mesh opacity key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("y"), self.qt_store.main_window).activated.connect(lambda up=True: self.toggle_surface_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("u"), self.qt_store.main_window).activated.connect(lambda up=False: self.toggle_surface_opacity(up))

    def set_camera(self):
        dialog = CameraPropsInputDialog(
            line1=("Fx", self.pvqt_store.camera_store.fx), 
            line2=("Fy", self.pvqt_store.camera_store.fy), 
            line3=("Cx", self.pvqt_store.camera_store.cx), 
            line4=("Cy", self.pvqt_store.camera_store.cy), 
            line5=("View Up", self.pvqt_store.camera_store.cam_viewup), 
            line6=("Cam Position", self.pvqt_store.camera_store.cam_position))
        if dialog.exec():
            fx, fy, cx, cy, cam_viewup, cam_position = dialog.getInputs()
            if not (fx == '' or fy == '' or cx == '' or cy == '' or cam_viewup == '' or cam_position == ''):
                success = self.pvqt_store.camera_store.set_camera(fx, fy, cx, cy, cam_viewup, cam_position)
                if not success:
                    QtWidgets.QMessageBox.warning(self, 'vision6D', "Error occured, check the format of the input values", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
    
    def add_pose_file(self):
        if self.paths_store.pose_path:
            self.qt_store.hintLabel.hide()
            transformation_matrix = np.load(self.paths_store.pose_path)
            self.pvqt_store.camera_store.set_transformation_matrix(transformation_matrix)

    def set_pose(self):
        # get the gt pose
        get_text_dialog = GetTextDialog()
        res = get_text_dialog.exec_()

        if res == QtWidgets.QDialog.Accepted:
            try:
                gt_pose = ast.literal_eval(get_text_dialog.user_text)
                gt_pose = np.array(gt_pose)
                if gt_pose.shape != (4, 4): 
                    QtWidgets.QMessageBox.warning(self, 'vision6D', "It needs to be a 4 by 4 matrix", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok) 
                    return None
                else:
                    self.qt_store.hintLabel.hide()
                    self.pvqt_store.camera_store.set_transformation_matrix(gt_pose)
                    return 0
            except: 
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return None
        else: 
            return None

    def set_scalar(self, nocs, actor_name):

        # Get the current vertices colors
        vertices_color, vertices = self.pvqt_store.mesh_store.get_mesh_colors(actor_name)
        
        # get the corresponding color
        colors = utils.color_mesh(vertices_color, nocs=nocs)
        if colors.shape != vertices.shape: 
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Cannot set the selected color", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        assert colors.shape == vertices.shape, "colors shape should be the same as vertices shape"
        
        # color the mesh and actor
        self.pvqt_store.mesh_store.set_mesh_colors(actor_name, colors)

    def set_color(self, color, actor_name):
        self.pvqt_store.mesh_store.set_mesh_colors(actor_name, color)

    def show_color_popup(self):

        checked_button = self.qt_store.button_group_actors_names.checkedButton()
        if checked_button:
            actor_name = checked_button.text()
            if actor_name in self.pvqt_store.mesh_actors:
                popup = PopUpDialog(self, on_button_click=lambda text: self.update_color_button_text(text, popup))
                button_position = self.qt_store.color_button.mapToGlobal(QPoint(0, 0))
                popup.move(button_position + QPoint(self.qt_store.color_button.width(), 0))
                popup.exec_()

                text = self.qt_store.color_button.text()
                self.pvqt_store.mesh_store.mesh_colors[actor_name] = text
                if text == 'nocs': self.set_scalar(True, actor_name)
                elif text == 'latlon': self.set_scalar(False, actor_name)
                else: self.set_color(text, actor_name)
            else:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Only be able to color mesh actors", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
    
    def update_color_button_text(self, text, popup):
        self.qt_store.color_button.setText(text)
        popup.close() # automatically close the popup window
    
    def toggle_image_opacity(self, up):
        change = 0.05
        if not up: change *= -1
        self.pvqt_store.image_store.update_image_opacity(change)
        self.ignore_spinbox_value_change = True
        self.opacity_spinbox.setValue(self.pvqt_store.image_opacity)
        self.ignore_spinbox_value_change = False

    def toggle_mask_opacity(self, up):
        change = 0.05
        if not up: change *= -1
        self.pvqt_store.mask_store.update_mask_opacity(change)
        self.ignore_spinbox_value_change = True
        self.opacity_spinbox.setValue(self.pvqt_store.mask_opacity)
        self.ignore_spinbox_value_change = False

    def toggle_surface_opacity(self, up):
        current_opacity = self.opacity_spinbox.value()
        change = 0.05
        if not up: change *= -1
        current_opacity += change
        current_opacity = np.clip(current_opacity, 0, 1)
        self.opacity_spinbox.setValue(current_opacity)
  
    def toggle_hide_meshes_button(self):
        self.toggle_hide_meshes_flag = not self.toggle_hide_meshes_flag
        
        if self.toggle_hide_meshes_flag:
            for button in self.qt_store.button_group_actors_names.buttons():
                if button.text() in self.pvqt_store.mesh_actors:
                    button.setChecked(True); self.qt_store.opacity_value_change(0)
    
            checked_button = self.qt_store.button_group_actors_names.checkedButton()
            if checked_button: 
                self.ignore_spinbox_value_change = True
                self.opacity_spinbox.setValue(0.0)
                self.ignore_spinbox_value_change = False
            else: QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        
        else:
            for button in self.qt_store.button_group_actors_names.buttons():
                if button.text() in self.pvqt_store.mesh_store.mesh_actors:
                    button.setChecked(True)
                    self.qt_store.opacity_value_change(self.pvqt_store.mesh_store.store_mesh_opacity[button.text()])

            checked_button = self.qt_store.button_group_actors_names.checkedButton()
            if checked_button:
                self.ignore_spinbox_value_change = True
                if checked_button.text() in self.pvqt_store.mesh_store.mesh_actors: self.opacity_spinbox.setValue(self.pvqt_store.mesh_store.mesh_opacity[checked_button.text()])
                self.ignore_spinbox_value_change = False
            else: QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def remove_actor(self, button):
        name = button.text()
        self.main_store.remove_actor(name)
        self.qt_store.remove_actor_button(name, button)
          
    def remove_actors_button(self):
        checked_button = self.qt_store.button_group_actors_names.checkedButton()
        if checked_button: self.remove_actor(checked_button)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def set_spacing(self):
        checked_button = self.qt_store.button_group_actors_names.checkedButton()
        if checked_button:
            actor_name = checked_button.text()
            if actor_name in self.pvqt_store.mesh_store.mesh_actors:
                spacing, ok = QtWidgets.QInputDialog().getText(self, 'Input', "Set Spacing", text=str(self.pvqt_store.mesh_store.mesh_spacing))
                if ok:
                    try: self.pvqt_store.mesh_store.mesh_spacing = ast.literal_eval(spacing)
                    except: QtWidgets.QMessageBox.warning(self, 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    self.add_mesh(actor_name, self.pvqt_store.mesh_store.meshdict[actor_name])
            else:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select a mesh object instead", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select a mesh actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)