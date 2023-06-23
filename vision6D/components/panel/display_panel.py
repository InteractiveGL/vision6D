import os
import ast
import pathlib

import numpy as np
import PIL.Image

from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint

from ... import utils
from ...stores.singleton import Singleton
from ...stores import QtStore
from ...stores import PlotStore
from ...stores import ImageStore
from ...stores import MaskStore
from ...stores import MeshStore
from ...stores import VideoStore
from ...stores import FolderStore

from ...widgets import GetTextDialog, CameraPropsInputDialog, PopUpDialog

class DisplayPanel(metaclass=Singleton):
    
    def __init__(self, display, button_group_actors_names):

        # Save references
        self.display = display
        self.button_group_actors_names = button_group_actors_names
        
        # Create a reference to the store
        self.plot_store = PlotStore()
        self.qt_store = QtStore()
        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.mesh_store = MeshStore()
        self.video_store = VideoStore()
        self.folder_store = FolderStore()

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
        draw_mask_button.clicked.connect(self.draw_mask)
        top_grid_layout.addWidget(draw_mask_button, 0, 2)

        # Create the video related button
        self.play_video_button = QtWidgets.QPushButton("Play Video")
        self.play_video_button.clicked.connect(self.play_video)
        top_grid_layout.addWidget(self.play_video_button, 0, 3)

        self.sample_video_button = QtWidgets.QPushButton("Sample Video")
        self.sample_video_button.clicked.connect(self.sample_video)
        top_grid_layout.addWidget(self.sample_video_button, 1, 0)

        self.save_frame_button = QtWidgets.QPushButton("Save Frame")
        self.save_frame_button.clicked.connect(self.save_frame)
        top_grid_layout.addWidget(self.save_frame_button, 1, 1)

        self.prev_frame_button = QtWidgets.QPushButton("Prev Frame")
        self.prev_frame_button.clicked.connect(self.prev_frame)
        top_grid_layout.addWidget(self.prev_frame_button, 1, 2)

        self.next_frame_button = QtWidgets.QPushButton("Next Frame")
        self.next_frame_button.clicked.connect(self.next_frame)
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

        self.qt_store.color_button.clicked.connect(self.show_color_popup)
        actor_grid_layout.addWidget(self.qt_store.color_button, 0, 0)
        
        # Create the opacity spinbox
        self.opacity_spinbox = QtWidgets.QDoubleSpinBox()
        self.opacity_spinbox.setMinimum(0.0)
        self.opacity_spinbox.setMaximum(1.0)
        self.opacity_spinbox.setDecimals(2)
        self.opacity_spinbox.setSingleStep(0.05)
        self.ignore_spinbox_value_change = True
        self.opacity_spinbox.setValue(0.3)
        self.ignore_spinbox_value_change = False 
        self.opacity_spinbox.valueChanged.connect(self.opacity_value_change)

        actor_grid_layout.addWidget(self.opacity_spinbox, 0, 1)

        # Create the hide button
        self.toggle_hide_meshes_flag = False
        hide_button = QtWidgets.QPushButton("toggle meshes")
        hide_button.clicked.connect(self.toggle_hide_meshes_button)
        actor_grid_layout.addWidget(hide_button, 0, 2)

        # Create the remove button
        remove_button = QtWidgets.QPushButton("Remove Actor")
        remove_button.clicked.connect(self.remove_select_actor)
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
        self.button_layout = QtWidgets.QVBoxLayout()
        self.button_layout.setSpacing(0)  # Remove spacing between buttons
        self.button_layout.addStretch()
        button_container.setLayout(self.button_layout)

        # Set the container widget as the scroll area's widget
        scroll_area.setWidget(button_container)
  
    # button related methods
    def button_actor_name_clicked(self, text):
        if text in self.mesh_store.mesh_actors:
            # set the current mesh color
            self.qt_store.color_button.setText(self.mesh_store.mesh_colors[text])
            # set mesh reference
            self.mesh_store.reference = text
            self.mesh_store.current_pose()
            curr_opacity = self.mesh_store.mesh_actors[self.mesh_store.reference].GetProperty().opacity
            self.opacity_spinbox.setValue(curr_opacity)
        else:
            self.qt_store.color_button.setText("Color")
            if text == 'image': curr_opacity = self.image_store.image_opacity
            elif text == 'mask': curr_opacity = self.mask_store.mask_opacity
            else: curr_opacity = self.opacity_spinbox.value()
            self.opacity_spinbox.setValue(curr_opacity)
            # self.mesh_store.reference = None
        
        output = f"-> Actor {text}, and its opacity is {curr_opacity}"
        if output not in self.qt_store.output_text.toPlainText(): self.qt_store.output_text.append(output)

    def add_button_actor_name(self, actor_name):
        button = QtWidgets.QPushButton(actor_name)
        button.setCheckable(True)  # Set the button to be checkable
        button.clicked.connect(lambda _, text=actor_name: self.button_actor_name_clicked(text))
        button.setChecked(True)
        button.setFixedSize(self.display.size().width(), 50)

        self.button_layout.insertWidget(0, button) # insert from the top # self.button_layout.addWidget(button)
        self.button_group_actors_names.addButton(button)
        self.button_actor_name_clicked(actor_name)

    def remove_select_actor(self):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button: self.remove_actor(checked_button)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def remove_actor(self, button):
        name = button.text()
        if name == 'image': actor = self.image_store.remove_actor()
        elif name == 'mask': actor = self.mask_store.remove_actor()
        else: 
            actor = self.mesh_store.remove_actor(name)
            self.qt_store.color_button.setText("Color")
        
        if actor: 
            self.plot_store.plotter.remove_actor(actor)
            self.qt_store.track_actors_names.remove(name)

        # clear out the plot if there is no actor
        if len(self.button_group_actors_names.buttons()) == 0: self.clear_plot()

        self.remove_actor_button(name, button)
    
    def remove_actor_button(self, name, button):

        # Mention the removal and remove button from group
        self.qt_store.output_text.append(f"-> Remove actor: {name}")
        self.button_group_actors_names.removeButton(button)
        self.button_layout.removeWidget(button)
        button.deleteLater()
    
    def clear_plot(self):
        # Clear out everything in the remove menu
        for button in self.button_group_actors_names.buttons():
            self.remove_actor(button)

        self.qt_store.hintLabel.show()
        self.qt_store.output_text.clear()
        self.qt_store.color_button.setText("Color")
        self.qt_store.track_actors_names = []
        
        # Reset stores
        self.plot_store.reset()
        self.image_store.reset()
        self.mask_store.reset()
        self.mesh_store.reset()
        self.folder_store.reset()
        self.video_store.reset()
        self.workspace_store.reset()

    def check_button(self, actor_name):
        for button in self.button_group_actors_names.buttons():
            if button.text() == actor_name: 
                button.setChecked(True)
                self.button_actor_name_clicked(actor_name)
                break
    
    def opacity_value_change(self, value):
        if self.ignore_spinbox_value_change: return 0
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button:
            actor_name = checked_button.text()
            if actor_name == 'image': 
                self.image_store.set_image_opacity(value)
            elif actor_name == 'mask': 
                self.mask_store.set_mask_opacity(value)
            elif actor_name in self.mesh_stores.mesh_actors: 
                self.mesh_store.set_mesh_opacity(actor_name, value)
        else:
            self.ignore_spinbox_value_change = True
            self.opacity_spinbox.setValue(value)
            self.ignore_spinbox_value_change = False
            return 0

    def set_camera(self):
        dialog = CameraPropsInputDialog(
            line1=("Fx", self.plot_store.fx), 
            line2=("Fy", self.plot_store.fy), 
            line3=("Cx", self.plot_store.cx), 
            line4=("Cy", self.plot_store.cy), 
            line5=("View Up", self.plot_store.cam_viewup), 
            line6=("Cam Position", self.plot_store.cam_position))
        if dialog.exec():
            fx, fy, cx, cy, cam_viewup, cam_position = dialog.getInputs()
            if not (fx == '' or fy == '' or cx == '' or cy == '' or cam_viewup == '' or cam_position == ''):
                pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup, pre_cam_position = self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position
                self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position = ast.literal_eval(fx), ast.literal_eval(fy), ast.literal_eval(cx), ast.literal_eval(cy), ast.literal_eval(cam_viewup), ast.literal_eval(cam_position)
                try:
                    self.plot_store.set_camera_props()
                except:
                    self.fx, self.fy, self.cx, self.cy, self.cam_viewup, self.cam_position = pre_fx, pre_fy, pre_cx, pre_cy, pre_cam_viewup, pre_cam_position
                    QtWidgets.QMessageBox.warning(self, 'vision6D', "Error occured, check the format of the input values", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
    
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
                    self.mesh_store.set_transformation_matrix(gt_pose)
                    return 0
            except: 
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                return None
        else: 
            return None

    def draw_mask(self):
        if self.image_store.image_path:
            self.mask_store.draw_mask(self.image_store.image_path)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def set_scalar(self, nocs, actor_name):

        # Get the current vertices colors
        vertices_color, vertices = self.mesh_store.get_mesh_colors(actor_name)
        
        # get the corresponding color
        colors = utils.color_mesh(vertices_color, nocs=nocs)
        if colors.shape != vertices.shape: 
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Cannot set the selected color", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        assert colors.shape == vertices.shape, "colors shape should be the same as vertices shape"
        
        # color the mesh and actor
        self.mesh_store.set_mesh_colors(actor_name, colors)

    def set_color(self, color, actor_name):
        self.mesh_store.set_mesh_colors(actor_name, color)

    def show_color_popup(self):

        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button:
            actor_name = checked_button.text()
            if actor_name in self.mesh_store.mesh_actors:
                popup = PopUpDialog(self, on_button_click=lambda text: self.update_color_button_text(text, popup))
                button_position = self.qt_store.color_button.mapToGlobal(QPoint(0, 0))
                popup.move(button_position + QPoint(self.qt_store.color_button.width(), 0))
                popup.exec_()

                text = self.qt_store.color_button.text()
                self.mesh_store.mesh_colors[actor_name] = text
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
        self.image_store.update_image_opacity(change)
        self.ignore_spinbox_value_change = True
        self.opacity_spinbox.setValue(self.image_store.image_opacity)
        self.ignore_spinbox_value_change = False

    def toggle_mask_opacity(self, up):
        change = 0.05
        if not up: change *= -1
        self.mask_store.update_mask_opacity(change)
        self.ignore_spinbox_value_change = True
        self.opacity_spinbox.setValue(self.mask_store.mask_opacity)
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
            for button in self.button_group_actors_names.buttons():
                if button.text() in self.mesh_store.mesh_actors:
                    button.setChecked(True); self.opacity_value_change(0)
    
            checked_button = self.button_group_actors_names.checkedButton()
            if checked_button: 
                self.ignore_spinbox_value_change = True
                self.opacity_spinbox.setValue(0.0)
                self.ignore_spinbox_value_change = False
            else: QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        
        else:
            for button in self.button_group_actors_names.buttons():
                if button.text() in self.mesh_store.mesh_actors:
                    button.setChecked(True)
                    self.opacity_value_change(self.mesh_store.store_mesh_opacity[button.text()])

            checked_button = self.button_group_actors_names.checkedButton()
            if checked_button:
                self.ignore_spinbox_value_change = True
                if checked_button.text() in self.mesh_store.mesh_actors: self.opacity_spinbox.setValue(self.mesh_store.mesh_opacity[checked_button.text()])
                self.ignore_spinbox_value_change = False
            else: QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select an actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
          
    def set_spacing(self):
        checked_button = self.button_group_actors_names.checkedButton()
        if checked_button:
            actor_name = checked_button.text()
            if actor_name in self.mesh_store.mesh_actors:
                spacing, ok = QtWidgets.QInputDialog().getText(self, 'Input', "Set Spacing", text=str(self.mesh_store.mesh_spacing))
                if ok:
                    try: self.mesh_store.mesh_spacing = ast.literal_eval(spacing)
                    except: QtWidgets.QMessageBox.warning(self, 'vision6D', "Format is not correct", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
                    self.add_mesh(actor_name, self.mesh_store.meshdict[actor_name])
            else:
                QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select a mesh object instead", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to select a mesh actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def play_video(self):
        if self.video_store.video_path:
            res = self.video_store.video_player.exec_()
            if res == QtWidgets.QDialog.Accepted:
                self.video_store.current_frame = self.video_player.current_frame
                self.qt_store.output_text.append(f"-> Current frame is ({self.video_store.current_frame}/{self.video_store.total_frame})")
                video_frame = self.video_store.load_per_frame_info()
                self.image_store.add_image(video_frame)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
    
    def sample_video(self):
        if self.video_store.video_path: 
            video_sampler = self.video_store.sample_video()
            res = video_sampler.exec_()
            if res == QtWidgets.QDialog.Accepted: 
                self.video_store.fps = round(video_sampler.fps)
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def save_frame(self):
        if self.video_store.video_path:
            video_frame = self.video_store.load_per_frame_info()
            os.makedirs(pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D", exist_ok=True)
            os.makedirs(pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "frames", exist_ok=True)
            output_frame_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "frames" / f"frame_{self.video_store.current_frame}.png"
            save_frame = PIL.Image.fromarray(video_frame)
            
            # save each frame
            save_frame.save(output_frame_path)
            self.qt_store.output_text.append(f"-> Save frame {self.video_store.current_frame}: ({self.video_store.current_frame}/{self.video_store.total_frame})")
            self.image_store.image_path = str(output_frame_path)

            # save gt_pose for each frame
            os.makedirs(pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses", exist_ok=True)
            output_pose_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses" / f"pose_{self.video_store.current_frame}.npy"
            self.mesh_store.current_pose()
            np.save(output_pose_path, self.mesh_store.transformation_matrix)
            self.qt_store.output_text.append(f"-> Save frame {self.video_store.current_frame} pose: \n{self.mesh_store.transformation_matrix}")
        elif self.folder_store.folder_path:
            # save gt_pose for specific frame
            os.makedirs(pathlib.Path(self.folder_path) / "vision6D", exist_ok=True)
            os.makedirs(pathlib.Path(self.folder_path) / "vision6D" / "poses", exist_ok=True)
            output_pose_path = pathlib.Path(self.folder_path) / "vision6D" / "poses" / f"{pathlib.Path(self.mesh_store.pose_path).stem}.npy"
            self.mesh_store.current_pose()
            np.save(output_pose_path, self.mesh_store.transformation_matrix)
            self.qt_store.output_text.append(f"-> Save frame {pathlib.Path(self.mesh_store.pose_path).stem} pose: \n{self.mesh_store.transformation_matrix}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or a folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        return 0
        
    def prev_frame(self):
        if self.video_store.video_path:
            current_frame = self.video_store.prev_frame()
            self.qt_store.output_text.append(f"-> Current frame is ({current_frame}/{self.video_store.total_frame})")
            pose_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses" / f"pose_{current_frame}.npy"
            if os.path.isfile(pose_path): 
                self.mesh_store.transformation_matrix = np.load(pose_path)
                self.mesh_store.register_pose(self.mesh_store.transformation_matrix)
                self.qt_store.output_text.append(f"-> Load saved frame {current_frame} pose: \n{self.mesh_store.transformation_matrix}")
            else: self.qt_store.output_text.append(f"-> No saved pose for frame {current_frame}")
            self.video_store.video_player.slider.setValue(current_frame)
            video_frame = self.video_store.load_per_frame_info()
            self.image_store.add_image(video_frame)
        elif self.folder_store.folder_path:
            self.folder_store.prev_frame()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or a folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def next_frame(self):
        # save pose from the previous frame
        self.save_frame()
        if self.video_store.video_path:
            current_frame = self.video_store.next_frame()
            self.qt_store.output_text.append(f"-> Current frame is ({current_frame}/{self.video_store.total_frame})")
            # load pose for the current frame if the pose exist
            pose_path = pathlib.Path(self.video_store.video_path).parent / f"{pathlib.Path(self.video_store.video_path).stem}_vision6D" / "poses" / f"pose_{current_frame}.npy"
            if os.path.isfile(pose_path): 
                self.mesh_store.transformation_matrix = np.load(pose_path)
                self.mesh_store.register_pose(self.mesh_store.transformation_matrix)
                self.qt_store.output_text.append(f"-> Load saved frame {current_frame} pose: \n{self.mesh_store.transformation_matrix}")
                self.video_store.video_player.slider.setValue(current_frame)
                video_frame = self.video_store.load_per_frame_info()
                self.image_store.add_image(video_frame)
        elif self.folder_store.folder_path:
            self.folder_store.next_frame()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a video or folder!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0