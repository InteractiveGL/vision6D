'''
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: utils.py
@time: 2023-07-03 20:34
@desc: the util functions for whole application
'''

import __future__
import os
import copy
import json
import logging
import pathlib
import operator
from scipy.spatial.transform import Rotation as R

import vtk
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from easydict import EasyDict
import trimesh
from PIL import Image
import cv2
# import pygeodesic.geodesic as geodesic
import vtk.util.numpy_support as vtknp # type: ignore

from PyQt5 import QtWidgets

logger = logging.getLogger("vision6D")
PKG_ROOT = pathlib.Path(os.path.abspath(__file__)).parent # vision6D

def fread(fid, _len, _type):
    if _len == 0:
        return np.empty(0)
    if _type == "int16":
        _type = np.int16
    elif _type == "int32":
        _type = np.int32
    elif _type == "float32":
        _type = np.float32
    elif _type == "double":
        _type = np.double
    elif _type == "char":
        _type = np.byte
    elif _type == "uint8":
        _type = np.uint8
    else:
        raise NotImplementedError(f"Invalid _type: {_type}")

    return np.fromfile(fid, _type, _len)

def meshread(fid, linesread=False, meshread2=False):
    """Reads mesh from fid data stream

    Parameters
    ----------
    fid (io.BufferedStream)
        Input IO stream
    _type (str, optional):
        Specifying the data _type for the last fread
    linesread (bool, optional)
        Distinguishing different use cases,
            False => meshread (default)
            True  => linesread

    """

    # Creating mesh instance
    mesh = EasyDict()

    # Reading parameters for mesh
    mesh.id = fread(fid, 1, "int32")
    mesh.numverts = fread(fid, 1, "int32")[0]
    mesh.numtris = fread(fid, 1, "int32")[0]

    # Loading mesh data
    n = fread(fid, 1, "int32")
    if n == -1:
        mesh.orient = fread(fid, 3, "int32")
        mesh.dim = fread(fid, 3, "int32")
        mesh.sz = fread(fid, 3, "float32")
        mesh.color = fread(fid, 3, "int32")
    else:
        mesh.color = np.zeros(3)
        mesh.color[0] = n
        mesh.color[1:3] = fread(fid, 2, "int32")

    # Given input parameter `linesread`
    if linesread:
        mesh.vertices = fread(fid, 3 * mesh.numverts, "float32").reshape([3, mesh.numverts], order="F")
        mesh.triangles = fread(fid, 2 * mesh.numtris, "int32").reshape([2, mesh.numtris], order="F")
    # Given input parameter `meshread2`
    elif meshread2:
        mesh.vertices = fread(fid, 3 * mesh.numverts, "double").reshape([3, mesh.numverts], order="F")
        mesh.triangles = fread(fid, 3 * mesh.numtris, "int32").reshape([3, mesh.numtris], order="F")
    # Given input parameter `meshread`
    else:
        # Loading mesh vertices and triangles
        mesh.vertices = fread(fid, 3 * mesh.numverts, "float32").reshape([3, mesh.numverts], order="F")
        mesh.triangles = fread(fid, 3 * mesh.numtris, "int32").reshape([3, mesh.numtris], order="F")

    # Return data
    return mesh

def load_meshobj(meshpath):
    with open(meshpath, "rb") as fid:
        meshobj = meshread(fid)
    return meshobj

def load_trimesh(meshpath):
    meshobj = load_meshobj(meshpath)
    # load the original ossicles
    idx = np.where(meshobj.orient != np.array((1,2,3)))
    for i in idx: meshobj.vertices[i] = (meshobj.dim[i] - 1).reshape((-1,1)) - meshobj.vertices[i]
    # check the results
    # writemesh(meshpath, meshobj, mirror)

    # Reflection Relative to YZ Plane (x):
    # if mirror: 
    #     meshobj.vertices[0] = (meshobj.dim[0] - 1) - meshobj.vertices[0].T
    #     writemesh(meshpath, meshobj, mirror=mirror)
        
    meshobj.vertices = meshobj.vertices * meshobj.sz.reshape((-1, 1))
    mesh = trimesh.Trimesh(vertices=meshobj.vertices.T, faces=meshobj.triangles.T, process=False) # mesh.vertices = meshobj.vertices.T.astype(np.float32) # mesh.faces = meshobj.triangles.T.astype(np.float32)
    assert mesh.vertices.shape == meshobj.vertices.T.shape
    assert mesh.faces.shape == meshobj.triangles.T.shape
    return mesh

def writemesh(meshpath, output_path, mesh, mirror=False, suffix=''):
    """
    write mesh object to improvise, and keep the original meshobj.sz
    """
    meshobj = load_meshobj(meshpath)
    # the shape has to be 3 x N
    meshobj.vertices = mesh.vertices.T / meshobj.sz.reshape((-1, 1)) if mesh.vertices.shape[1]==3 else mesh.vertices / meshobj.sz.reshape((-1, 1))
    meshobj.orient = np.array((1, 2, 3), dtype="int32")

    name = output_path.stem
    if "centered" in name: 
        name = '_'.join(name.split("_")[:-1])
    name += suffix
    
    if mirror:
        if 'left' in name: side = "right"
        elif "right" in name: side = "left"
        name = name.split("_")[0] + "_" + side + "_" + '_'.join(name.split("_")[2:-1])

    with open(output_path.parent / (name + ".mesh"), "wb") as f:
        f.write(meshobj.id.astype('int32'))
        f.write(meshobj.numverts.astype('int32'))
        f.write(meshobj.numtris.astype('int32'))
        f.write(np.int32(-1))
        f.write(meshobj.orient.astype('int32'))
        f.write(meshobj.dim.astype('int32'))
        f.write(meshobj.sz.astype('float32'))
        f.write(meshobj.color.astype('int32'))
        # ndarray need to be continuous!
        f.write(np.array(meshobj.vertices).astype("float32").tobytes(order='F'))
        f.write(meshobj.triangles.astype("int32").tobytes(order='F'))
        """
        # if hasattr(mesh, "opacity"):
        #     f.write(mesh.opacity.T)
        #     if hasattr(mesh, "colormap"):
        #         f.write(mesh.colormap.numcols.T)
        #         f.write(mesh.colormap.numverts.T)
        #         f.write(mesh.colormap.cols.T)
        #         f.write(mesh.colormap.vertexindexes.T.tobytes(order='C'))
        """
        
def color2binary_mask(color_mask):
    binary_mask = np.zeros(color_mask[...,:1].shape, dtype=np.uint8)
    x, y, _ = np.where(color_mask != [0., 0., 0.])
    binary_mask[x, y] = 1 
    return binary_mask

def create_2d_3d_pairs(color_mask:np.ndarray, vertices:pv.pyvista_ndarray, binary_mask:np.ndarray=None):

    if binary_mask is None: 
        binary_mask = color2binary_mask(color_mask)
        assert (binary_mask == (0.3*color_mask[..., :1] + 0.59*color_mask[..., 1:2] + 0.11*color_mask[..., 2:]).astype("bool").astype('uint8')).all()

    idx = np.where(binary_mask == 1)

    # swap the points for opencv, maybe because they handle RGB image differently (RGB -> BGR in opencv)
    idx = idx[:2][::-1]

    pts = np.stack((idx[0], idx[1]), axis=1)
    
    # Obtain the 3D verticies (normaize rgb values)
    rgb = color_mask[pts[:,1], pts[:,0]]
    if np.max(rgb) > 1: rgb = rgb / 255

    # denormalize to get the rgb value for vertices respectively
    r = de_normalize(rgb[:, 0], vertices[..., 0])
    g = de_normalize(rgb[:, 1], vertices[..., 1])
    b = de_normalize(rgb[:, 2], vertices[..., 2])
    vtx = np.stack([r, g, b], axis=1)
    
    return vtx, pts

def solve_epnp_cv2(pts2d, pts3d, camera_intrinsics):
    pts2d = pts2d.astype('float32')
    pts3d = pts3d.astype('float32')
    camera_intrinsics = camera_intrinsics.astype('float32')

    predicted_pose = np.eye(4)
    if pts2d.shape[0] > 4:
        # Use EPNP, inliers are the indices of the inliers
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, distCoeffs=np.zeros((4, 1)), confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if success:
            coordinate_change = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            predicted_pose[:3, :3] = coordinate_change @ cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = coordinate_change @ np.squeeze(translation_vector)
    return predicted_pose

def transform_vertices(vertices, transformation_matrix=np.eye(4)):

    ones = np.ones((vertices.shape[0], 1))
    homogeneous_vertices = np.append(vertices, ones, axis=1)
    transformed_vertices = (transformation_matrix @ homogeneous_vertices.T)[:3].T
    
    return transformed_vertices

def normalize(x):
    return (x - min(x)) / (max(x) - min(x))

def de_normalize(rgb, vertices):
    return rgb * (np.max(vertices) - np.min(vertices)) + np.min(vertices)

def color_mesh_nocs(vertices, color=''):
    assert vertices.shape[1] == 3, "the vertices is suppose to be transposed"
    colors = copy.deepcopy(vertices)
    # normalize vertices and center it to 0
    colors[..., 0] = normalize(vertices[..., 0])
    colors[..., 1] = normalize(vertices[..., 1])
    colors[..., 2] = normalize(vertices[..., 2])
    # elif color == 'latlon': colors = load_latitude_longitude()
    return colors
    
def save_image(array, folder, name):
    img = Image.fromarray(array)
    img.save(folder / name)

def mesh2ply(meshpath, output_path):
    mesh = load_trimesh(meshpath)
    ply_file = trimesh.exchange.ply.export_ply(mesh)
    with open(output_path, "wb") as fid:
        fid.write(ply_file)

def rigid_transform_3D(A, B):

    assert A.shape == B.shape

    # find mean
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # # ensure centroids are 3x1
    # centroid_A = centroid_A.reshape(-1, 1)
    # centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am.T @ Bm

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B
    # convert the shape from (1, 3) to (3, 1)
    t = t.reshape((-1,1))

    rt = np.vstack((np.hstack((R, t)), np.array([0,0,0,1])))

    return rt

def compute_transformation_matrix(A, B):
    # Center the points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    
    H = np.dot(AA.T, BB) # Compute the covariance matrix
    
    # The amazing SVD
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # Count for the special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    t = centroid_B - np.dot(R, centroid_A)
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t
    
    return transformation_matrix

def get_actor_user_matrix(mesh_model):
    vertices, _ = get_mesh_actor_vertices_faces(mesh_model.actor)
    matrix = compute_transformation_matrix(mesh_model.source_obj.vertices, transform_vertices(vertices, mesh_model.actor.user_matrix))
    return matrix

def load_latitude_longitude():
    # get the latitude and longitude
    # latlon_map_path = pkg_resources.resource_filename('vision6D', 'data/ossiclesCoordinateMapping.json')
    with open(PKG_ROOT / "data" / "ossiclesCoordinateMapping2.json", "r") as f: data = json.load(f)
    
    latitude = np.array(data['latitude']).reshape((len(data['latitude'])), 1)
    longitude = np.array(data['longitude']).reshape((len(data['longitude'])), 1)
    placeholder = np.zeros((len(data['longitude']), 1))
    
    # set the latlon attribute
    latlon = np.hstack((latitude, longitude, placeholder))
    return latlon

def latLon2xyzv1(m,lat,lon,gx,gy):
    vert = np.array([0, 0, 0])
    for f in m.faces:
        lonf = lon[f]
        if np.sum(lonf==0)>0 and np.sum(lonf>=0.9)>0:
            lonf[lonf==0] = 1
        V = [[lat[f[1]] - lat[f[0]], lat[f[2]] - lat[f[0]]], [lonf[1] - lonf[0], lonf[2] - lonf[0]]]
        ab = np.linalg.pinv(V) @ (np.array([gx,gy]) - np.array([lat[f[0]], lonf[0]]))
        a = ab[0]
        b = ab[1]
        if a>=0 and b>=0 and a+b<=1:
            vert = np.array(m.vertices[f[0]] + a*(m.vertices[f[1]]-m.vertices[f[0]]) + b*(m.vertices[f[2]]-m.vertices[f[0]]))
            break
    return vert

def latLon2xyz(m,lat,lonf,msk,gx,gy):
    xyz = []
    class xyznode:
        def __init__(self, pnt, d):
            self.pnt=pnt
            self.d=d
        def __le__(self, rhs):
            return self.d <= rhs.d

    indx = np.where(
        (np.sum(lat[m.faces]>=gx,axis=1)>0)&(np.sum(lat[m.faces]<=gx,axis=1)>0)&msk&
        (np.sum(lonf>=gy,axis=1)>0)&(np.sum(lonf<=gy,axis=1)>0))[0]
    if len(indx)==0:
        indx = [np.argmin(np.min((lat[m.faces]-gx)*(lat[m.faces]-gx) + (lonf-gy)*(lonf-gy), axis=1))]

    for ind in indx:
        f = m.faces[ind]
        V = np.array([[lat[f[1]] - lat[f[0]],lat[f[2]] - lat[f[0]]],
                        [lonf[ind,1] - lonf[ind,0],lonf[ind,2] - lonf[ind,0]]])##T?
        ab = np.linalg.pinv(V) @ (np.array([gx,gy]) - np.array([lat[f[0]],lonf[ind,0]]))
        a = ab[0]
        b = ab[1]
        if a>=0 and b>=0 and a + b<=1:
            xyz.append(xyznode(m.vertices[f[0]] + a * (m.vertices[f[1]] - m.vertices[f[0]]) + b * (m.vertices[f[2]] - m.vertices[f[0]]),
                        np.sum((np.array([lat[f[0]], lonf[ind,0]]) + V@ab - np.array([gx,gy]))**2)))
        else:
            c = V[:,0] @ (np.array([gx,gy]) - np.array([lat[f[0]],lonf[ind,0]]))/(np.linalg.norm(V[:,0])**2)
            d = V[:,1] @ (np.array([gx,gy]) - np.array([lat[f[0]],lonf[ind,0]])) / (np.linalg.norm(V[:,1])**2)
            v2 = np.array([lat[f[2]] - lat[f[1]], lonf[ind,2] - lonf[ind,1]])
            e = v2 @ (np.array([gx,gy]) - np.array([lat[f[1]],lonf[ind,1]])) / (np.linalg.norm(v2)**2)
            c = np.clip(c,0,1)
            d = np.clip(d,0,1)
            e = np.clip(e,0,1)
            p1 = c*V[:,0] + np.array([lat[f[0]],lonf[ind,0]])
            p2 = d*V[:,1] + np.array([lat[f[0]],lonf[ind,0]])
            p3 = e*v2     + np.array([lat[f[1]],lonf[ind,1]])
            d1 = np.sum((p1 - np.array([gx,gy]))**2)
            d2 = np.sum((p2 - np.array([gx,gy]))**2)
            d3 = np.sum((p3 - np.array([gx,gy]))**2)
            if d1 < d2 and d1<d3:
                xyz.append(xyznode(m.vertices[f[0]] + c * (m.vertices[f[1]] - m.vertices[f[0]]),d1))
            elif d2 < d3:
                xyz.append(xyznode(m.vertices[f[0]] + d * (m.vertices[f[2]] - m.vertices[f[0]]),d2))
            else:
                xyz.append(xyznode(m.vertices[f[1]] + e * (m.vertices[f[2]] - m.vertices[f[1]]),d3))
    return np.min(xyz).pnt

def get_image_actor_scalars(actor):
    input = actor.GetMapper().GetInput()
    shape = input.GetDimensions()[::-1]
    point_data = input.GetPointData().GetScalars()
    point_array = vtknp.vtk_to_numpy(point_data)
    if len(point_array.shape) == 1: point_array = point_array.reshape(*point_array.shape, 1)
    scalars = point_array.reshape(*shape[1:], point_array.shape[-1])
    return scalars

def get_mask_actor_points(actor):
    input = actor.GetMapper().GetInput()
    point_data = input.GetPoints().GetData()
    points_array = vtknp.vtk_to_numpy(point_data)
    vtk_matrix = actor.GetMatrix()
    matrix = np.array([[vtk_matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
    # Calculate points not with homogeneous form
    points = (matrix[:3, :3] @ points_array.T).T + matrix[:3, 3:].T
    # Calculate points with homogeneous coordinates
    homogeneous_points = np.hstack((points_array, np.ones((points_array.shape[0], 1))))
    transformed_points = ((matrix @ homogeneous_points.T).T)[:, :3]
    assert np.isclose(transformed_points, points).all(), "points and transformed_points should be very very close!"
    return transformed_points

def get_mesh_actor_vertices_faces(actor):
    input = actor.GetMapper().GetInput()
    points = input.GetPoints().GetData()
    cells = input.GetPolys().GetData()
    vertices = vtknp.vtk_to_numpy(points)
    """
    # popular presentation
    Triangle 1: (0, 1, 2)
    Triangle 2: (3, 4, 5)
    Triangle 3: (6, 7, 8)
    Triangle 4: (9, 10, 11)

    # When PyVista converts this mesh into a vtkPolyData object, the faces are represented as a list of vertex indices and the number of vertices in each face:
    Face 1: (3, 0, 1, 2)
    Face 2: (3, 3, 4, 5)
    Face 3: (3, 6, 7, 8)
    Face 4: (3, 9, 10, 11)
    """
    faces = vtknp.vtk_to_numpy(cells).reshape((-1, 4))
    faces = faces[:, 1:] # trim the first element in each row
    return vertices, faces

def get_mesh_actor_scalars(actor):
    input = actor.GetMapper().GetInput()
    point_data = input.GetPointData()
    scalars = point_data.GetScalars()
    if scalars: scalars = vtknp.vtk_to_numpy(scalars)
    return scalars

def create_render(w=1920, h=1080):
    render = pv.Plotter(window_size=[w, h], lighting=None, off_screen=True) 
    render.set_background('black')
    assert render.background_color == "black", "render's background need to be black"
    return render

# Calculate the rotation error
def angler_distance(R_a, R_b):
    R_diff = np.dot(R_a, R_b.T)
    angle_diff_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
    angle_diff_deg = np.degrees(angle_diff_rad)
    return angle_diff_deg

def decompose_transform(transformation_matrix, euler_sequence='xyz'):
    R_mat = transformation_matrix[:3, :3]
    rotation = R.from_matrix(R_mat)
    euler_angles = rotation.as_euler(euler_sequence, degrees=True)
    translation = transformation_matrix[:3, 3]
    return euler_angles, translation

def compose_transform(euler_angles, translation, euler_sequence='xyz'):
    rotation = R.from_euler(euler_sequence, euler_angles, degrees=True)
    R_mat = rotation.as_matrix()
    transformation_matrix = np.eye(4, dtype=float)
    transformation_matrix[:3, :3] = R_mat
    transformation_matrix[:3, 3] = translation
    return transformation_matrix

def reset_vtk_lut(colormap):
    if isinstance(colormap, str):
        color_num = 256
        viridis_cm = plt.get_cmap(colormap, color_num)
        colors = viridis_cm(np.arange(color_num))
    else:
        colors = colormap
        color_num = len(colors)
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(color_num)
    lut.Build()
    for i, color in enumerate(colors): lut.SetTableValue(i, color[0], color[1], color[2], 1.0)  # Last value is alpha (opacity)
    return lut

def display_warning(message):
    QtWidgets.QMessageBox.warning(QtWidgets.QMainWindow(), "vision6D", message, QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
    return 0

def require_attributes(attributes_with_messages):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            for attr_path, warning_message in attributes_with_messages:
                try:
                    attr_value = operator.attrgetter(attr_path)(self)
                    if not attr_value:
                        display_warning(warning_message)
                        return
                except AttributeError:
                    display_warning(warning_message)
                    return
            return func(self, *args, **kwargs)
        return wrapper
    return decorator
