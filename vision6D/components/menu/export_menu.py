from functools import partial

class ExportMenu():

    def __init__(self, menu):

        # Save parameter
        self.menu = menu

        self.menu.addAction('Image', self.export_image)
        self.menu.addAction('Mask', self.export_mask)
        self.menu.addAction('Pose', self.export_pose)
        self.menu.addAction('Mesh Render', self.export_mesh_render)
        self.menu.addAction('SegMesh Render', self.export_segmesh_render)

    def render_image(self, camera):
        self.render.clear()
        render_actor = self.image_actor.copy(deep=True)
        render_actor.GetProperty().opacity = 1
        self.render.add_actor(render_actor, pickable=False)
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        image = self.render.last_image
        return image
    
    def render_mask(self, camera):
        self.render.clear()
        render_actor = self.mask_actor.copy(deep=True)
        render_actor.GetProperty().opacity = 1
        self.render.add_actor(render_actor, pickable=False)
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        image = self.render.last_image
        return image

    def render_mesh(self, camera):
        self.render.clear()
        vertices, faces = vis.utils.get_mesh_actor_vertices_faces(self.mesh_actors[self.reference])
        mesh_data = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        colors = vis.utils.get_mesh_actor_scalars(self.mesh_actors[self.reference])
        if colors is not None: 
            assert colors.shape == vertices.shape, "colors shape should be the same as vertices shape"
            mesh = self.render.add_mesh(mesh_data, scalars=colors, rgb=True, style='surface', opacity=1, name=self.reference)
        else:
            mesh = self.render.add_mesh(mesh_data, color=self.mesh_colors[self.reference], style='surface', opacity=1, name=self.reference)
        mesh.user_matrix = self.mesh_actors[self.reference].user_matrix
        
        self.render.camera = camera
        self.render.disable()
        self.render.show(auto_close=False)
        image = self.render.last_image
        return image

    def export_image(self):

        if self.image_actor:
            image = self.render_image(camera=self.camera.copy())
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Image Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.output_text.append(f"-> Export image render to:\n {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load an image first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def export_mask(self):
        if self.mask_actor:
            image = self.render_mask(camera=self.camera.copy())
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Mask Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.output_text.append(f"-> Export mask render to:\n {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a mask first!", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
   
    def export_pose(self):
        if self.reference: 
            self.update_gt_pose()
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Pose Files (*.npy)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.npy')
                np.save(output_path, self.transformation_matrix)
                self.output_text.append(f"-> Saved:\n{self.transformation_matrix}\nExport to:\n {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
    
    def export_mesh_render(self, save_render=True):

        if self.reference:
            image = self.render_mesh(camera=self.camera.copy())
            if save_render:
                output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "Mesh Files (*.png)")
                if output_path:
                    if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
                    rendered_image = PIL.Image.fromarray(image)
                    rendered_image.save(output_path)
                    self.output_text.append(f"-> Export reference mesh render to:\n {str(output_path)}")
            return image
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to set a reference or load a mesh first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0

    def export_segmesh_render(self):

        if self.reference and self.mask_actor:
            segmask = self.render_mask(camera=self.camera.copy())
            if np.max(segmask) > 1: segmask = segmask / 255
            image = self.render_mesh(camera=self.camera.copy())
            image = (image * segmask).astype(np.uint8)
            output_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "SegMesh Files (*.png)")
            if output_path:
                if pathlib.Path(output_path).suffix == '': output_path = output_path.parent / (output_path.stem + '.png')
                rendered_image = PIL.Image.fromarray(image)
                rendered_image.save(output_path)
                self.output_text.append(f"-> Export segmask render:\n to {str(output_path)}")
        else:
            QtWidgets.QMessageBox.warning(self, 'vision6D', "Need to load a mesh or mask first", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
