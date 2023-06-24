import numpy as np
import trimesh

from PyQt5 import QtWidgets

from ... import utils
from ...stores import QtStore
from ...stores import MeshStore
from ...stores import ImageStore
from ...stores import MaskStore

class PnPMenu():
    def __init__(self):

        self.qt_store = QtStore()
        self.image_store = ImageStore()
        self.mask_store = MaskStore()
        self.mesh_store = MeshStore()

    def nocs_epnp(self, color_mask, mesh):
        vertices = mesh.vertices
        pts3d, pts2d = utils.create_2d_3d_pairs(color_mask, vertices)
        pts2d = pts2d.astype('float32')
        pts3d = pts3d.astype('float32')
        camera_intrinsics = self.plot_store.camera_intrinsics.astype('float32')
        predicted_pose = utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics, self.plot_store.camera.position)
        return predicted_pose

    def latlon_epnp(self, color_mask, mesh):
        binary_mask = utils.color2binary_mask(color_mask)
        idx = np.where(binary_mask == 1)
        # swap the points for opencv, maybe because they handle RGB image differently (RGB -> BGR in opencv)
        idx = idx[:2][::-1]
        pts2d = np.stack((idx[0], idx[1]), axis=1)
        pts3d = []
        
        # Obtain the rg color
        color = color_mask[pts2d[:,1], pts2d[:,0]][..., :2]
        if np.max(color) > 1: color = color / 255
        gx = color[:, 0]
        gy = color[:, 1]

        lat = np.array(self.mesh_store.latlon[..., 0])
        lon = np.array(self.mesh_store.latlon[..., 1])
        lonf = lon[mesh.faces]
        msk = (np.sum(lonf>=0, axis=1)==3) & (np.sum(lat[mesh.faces]>=0, axis=1)==3)
        for i in range(len(pts2d)):
            pt = utils.latLon2xyz(mesh, lat, lonf, msk, gx[i], gy[i])
            pts3d.append(pt)
       
        pts3d = np.array(pts3d).reshape((len(pts3d), 3))

        pts2d = pts2d.astype('float32')
        pts3d = pts3d.astype('float32')
        camera_intrinsics = self.plot_store.camera_intrinsics.astype('float32')
        
        predicted_pose = utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics, self.plot_store.camera.position)

        return predicted_pose

    def epnp_mesh(self):
        if len(self.mesh_store.mesh_actors) == 1: self.mesh_store.reference = list(self.mesh_store.mesh_actors.keys())[0]
        if self.mesh_store.reference:
            colors = utils.get_mesh_actor_scalars(self.mesh_store.mesh_actors[self.mesh_store.reference])
            if colors is not None and (not np.all(colors == colors[0])):
                color_mask = self.mesh_store.render_mesh(camera=self.plot_store.camera.copy())
                gt_pose = self.mesh_store.mesh_actors[self.mesh_store.reference].user_matrix
                if self.plot_store.mirror_x: gt_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                if self.plot_store.mirror_y: gt_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose

                if np.sum(color_mask):
                    if self.mesh_store.mesh_colors[self.mesh_store.reference] == 'nocs':
                        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_store.mesh_actors[self.mesh_store.reference])
                        mesh = trimesh.Trimesh(vertices, faces, process=False)
                        predicted_pose = self.nocs_epnp(color_mask, mesh)
                        if self.plot_store.mirror_x: predicted_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        if self.plot_store.mirror_y: predicted_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        error = np.sum(np.abs(predicted_pose - gt_pose))
                        self.qt_store.output_text.append(f"-> PREDICTED POSE WITH <span style='background-color:yellow; color:black;'>NOCS COLOR</span>: ")
                        self.qt_store.output_text.append(f"{predicted_pose}\nGT POSE: \n{gt_pose}\nERROR: \n{error}")
                    else: return "Only works using EPnP with latlon mask"
                else: return "The color mask is blank (maybe set the reference mesh wrong)"
            else: return "The mesh need to be colored, with gradient color"
        else: return "A mesh need to be loaded/mesh reference need to be set"

        return ""

    def epnp_mask(self, nocs_method):
        if self.mask_store.mask_actor:
            mask_data = self.mask_store.render_mask(camera=self.plot_store.camera.copy())
            if np.max(mask_data) > 1: mask_data = mask_data / 255

            # current shown mask is binary mask
            if np.all(np.logical_or(mask_data == 0, mask_data == 1)):
                if len(self.mesh_store.mesh_actors) == 1: 
                    self.mesh_store.reference = list(self.mesh_store.mesh_actors.keys())[0]
                if self.mesh_store.reference:
                    colors = utils.get_mesh_actor_scalars(self.mesh_store.mesh_actors[self.mesh_store.reference])
                    if colors is not None and (not np.all(colors == colors[0])):
                        color_mask = self.mesh_store.render_mesh(camera=self.plot_store.camera.copy())
                        nocs_color = (self.mesh_store.mesh_colors[self.mesh_store.reference] == 'nocs')
                        gt_pose = self.mesh_store.mesh_actors[self.mesh_store.reference].user_matrix
                        if self.plot_store.mirror_x: gt_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                        if self.plot_store.mirror_y: gt_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ gt_pose
                        vertices, faces = utils.get_mesh_actor_vertices_faces(self.mesh_store.mesh_actors[self.mesh_store.reference])
                        mesh = trimesh.Trimesh(vertices, faces, process=False)
                    else: return "The mesh need to be colored, with gradient color"
                else: return "A mesh need to be loaded/mesh reference need to be set"
                color_mask = (color_mask * mask_data).astype(np.uint8)
            
            if np.sum(color_mask):
                if nocs_method == nocs_color:
                    if nocs_method: 
                        color_theme = 'NOCS'
                        predicted_pose = self.nocs_epnp(color_mask, mesh)
                        if self.plot_store.mirror_x: predicted_pose = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        if self.plot_store.mirror_y: predicted_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ predicted_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                    else: 
                        color_theme = 'LATLON'
                        if self.plot_store.mirror_x: color_mask = color_mask[:, ::-1, :]
                        if self.plot_store.mirror_y: color_mask = color_mask[::-1, :, :]
                        predicted_pose = self.latlon_epnp(color_mask, mesh)
                    error = np.sum(np.abs(predicted_pose - gt_pose))
                    self.qt_store.output_text.append(f"-> PREDICTED POSE WITH <span style='background-color:yellow; color:black;'>{color_theme} COLOR (MASKED)</span>: ")
                    self.qt_store.output_text.append(f"{predicted_pose}\nGT POSE: \n{gt_pose}\nERROR: \n{error}")
                else: return "Clicked the wrong method"
            else: return "The color mask is blank (maybe set the reference mesh wrong)"
        else: return "please load a mask first"
        
        return ''