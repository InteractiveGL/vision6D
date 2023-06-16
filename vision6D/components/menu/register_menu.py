from functools import partial

class RegisterMenu():

    def __init__(self, menu):

        # Save parameter
        self.menu = menu
        self.menu.addAction('Reset GT Pose (k)', self.reset_gt_pose)
        self.menu.addAction('Update GT Pose (l)', self.update_gt_pose)
        self.menu.addAction('Current Pose (t)', self.current_pose)
        self.menu.addAction('Undo Pose (s)', self.undo_pose)

        QtWidgets.QShortcut(QtGui.QKeySequence("k"), self).activated.connect(self.reset_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("l"), self).activated.connect(self.update_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("t"), self).activated.connect(self.current_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("s"), self).activated.connect(self.undo_pose)

    def reset_gt_pose(self):
        self.output_text.append(f"-> Reset the GT pose to: \n{self.initial_pose}")
        self.pvqt_store.register_pose(self.initial_pose)

    def update_gt_pose(self):
        self.initial_pose = self.transformation_matrix
        self.current_pose()
        self.output_text.append(f"Update the GT pose to: \n{self.initial_pose}")
            
    def current_pose(self):
        if self.pvqt_store.current_pose():
            self.output_text.append(f"-> Current reference mesh is: <span style='background-color:yellow; color:black;'>{self.reference}</span>")
            self.output_text.append(f"Current pose is: \n{self.transformation_matrix}")

    def undo_pose(self):
        if self.button_group_actors_names.checkedButton():
            actor_name = self.button_group_actors_names.checkedButton().text()
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Choose a mesh actor first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        if len(self.undo_poses[actor_name]) != 0: 
            self.transformation_matrix = self.undo_poses[actor_name].pop()
            if (self.transformation_matrix == self.mesh_actors[actor_name].user_matrix).all():
                if len(self.undo_poses[actor_name]) != 0: 
                    self.transformation_matrix = self.undo_poses[actor_name].pop()

            self.output_text.append(f"-> Current reference mesh is: <span style='background-color:yellow; color:black;'>{actor_name}</span>")
            self.output_text.append(f"Undo pose to: \n{self.transformation_matrix}")
                
            self.mesh_actors[actor_name].user_matrix = self.transformation_matrix
            self.plotter.add_actor(self.mesh_actors[actor_name], pickable=True, name=actor_name)
