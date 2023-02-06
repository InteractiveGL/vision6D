import __future__
import copy
from typing import Type

import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
import trimesh
from PIL import Image
import cv2

def fread(fid, _len, _type):
    if _len == 0:
        return np.empty(0)
    if _type == "int16":
        _type = np.int16
    elif _type == "int32":
        _type = np.int32
    elif _type == "float":
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
        mesh.sz = fread(fid, 3, "float")
        mesh.color = fread(fid, 3, "int32")
    else:
        mesh.color = np.zeros(3)
        mesh.color[0] = n
        mesh.color[1:3] = fread(fid, 2, "int32")

    # Given input parameter `linesread`
    if linesread:
        mesh.vertices = fread(fid, 3 * mesh.numverts, "float").reshape(
            [3, mesh.numverts], order="F"
        )
        mesh.triangles = fread(fid, 2 * mesh.numtris, "int32").reshape(
            [2, mesh.numtris], order="F"
        )
    # Given input parameter `meshread2`
    elif meshread2:
        mesh.vertices = fread(fid, 3 * mesh.numverts, "double").reshape(
            [3, mesh.numverts], order="F"
        )
        mesh.triangles = fread(fid, 3 * mesh.numtris, "int32").reshape(
            [3, mesh.numtris], order="F"
        )
    # Given input parameter `meshread`
    else:
        # Loading mesh vertices and triangles
        mesh.vertices = fread(fid, 3 * mesh.numverts, "float").reshape(
            [3, mesh.numverts], order="F"
        )
        mesh.triangles = fread(fid, 3 * mesh.numtris, "int32").reshape(
            [3, mesh.numtris], order="F"
        )

    # Return data
    return mesh

def load_meshobj(meshpath):
    with open(meshpath, "rb") as fid:
        meshobj = meshread(fid)
    return meshobj

def load_trimesh(meshpath, mirror=False):
    meshobj = load_meshobj(meshpath)
    # load the original ossicles
    idx = np.where(meshobj.orient != np.array((1,2,3)))
    for i in idx: meshobj.vertices[i] = (meshobj.dim[i] - 1).reshape((-1,1)) - meshobj.vertices[i]

    # Reflection Relative to YZ Plane (x):
    if mirror: meshobj.vertices[0] = (meshobj.dim[0] - 1) - meshobj.vertices[0].T
        
    meshobj.vertices = meshobj.vertices * meshobj.sz.reshape((-1, 1))
    mesh = trimesh.Trimesh(vertices=meshobj.vertices.T, faces=meshobj.triangles.T)
    return mesh

def writemesh(meshpath, mesh):
    """
    write mesh object to improvise, and keep the original meshobj.sz
    """
    meshobj = load_meshobj(meshpath)
    # the shape has to be 3 x N
    meshobj.vertices = mesh.vertices.T / meshobj.sz.reshape((-1, 1))
    meshobj.orient = np.array((1, 2, 3), dtype="int32")

    name = meshpath.stem
    if 'left' in name: side = "right"
    elif "right" in name: side = "left"
    name = name.split("_")[0] + "_" + side + "_" + '_'.join(name.split("_")[2:])
    filename = meshpath.parent / (name + ".mesh")

    with open(filename, "wb") as f:
        f.write(meshobj.id.astype('int32'))
        f.write(meshobj.numverts.astype('int32'))
        f.write(meshobj.numtris.astype('int32'))
        f.write(np.int32(-1))
        f.write(meshobj.orient.astype('int32'))
        f.write(meshobj.dim.astype('int32'))
        f.write(meshobj.sz.astype('float32'))
        f.write(meshobj.color.astype('int32'))
        # ndarray need to be continuous!
        f.write(meshobj.vertices.astype("float32").tobytes(order='F'))
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
    print("finish writing to a mesh file")
        
def color2binary_mask(color_mask):
    binary_mask = np.zeros(color_mask[...,:1].shape)
    x, y, _ = np.where(color_mask != [0., 0., 0.])
    binary_mask[x, y] = 1      
    return binary_mask

def create_2d_3d_pairs(color_mask:np.ndarray, obj:Type, object_name:str, npts:int=-1, binary_mask:np.ndarray=None):

    if binary_mask is None:
        binary_mask = color2binary_mask(color_mask)

    # make sure the binary mask only contains 0 and 1
    binary_mask = np.where(binary_mask != 0, 1, 0)
    binary_mask_bool = binary_mask.astype('bool')
    assert (binary_mask == binary_mask_bool).all(), "binary mask should be the same as binary mask bool"

    # To convert color_mask to bool type, we need to consider all three channels for color image, or conbine all channels to grey for color images!
    color_mask_bool = (0.2989 * color_mask[..., :1] + 0.5870*color_mask[..., 1:2] + 0.1140*color_mask[..., 2:]).astype("bool") 
    # # solution2
    # color_mask_bool = np.logical_or(color_mask.astype("bool")[..., :1], color_mask.astype("bool")[..., 1:2], color_mask.astype("bool")[..., 2:])
    # # solution3
    # color_mask_bool = color_mask.astype("bool")
    # color_mask_bool = (color_mask_bool[..., :1] + color_mask_bool[..., 1:2] + color_mask_bool[..., 2:]).astype("bool")
    assert (binary_mask == color_mask_bool).all(), "binary_mask is not the same as the color_mask_bool"

    # Randomly select points in the mask
    idx = np.where(binary_mask == 1)
    
    # swap the points for opencv, maybe because they handle RGB image differently (RGB -> BGR in opencv)
    # pts = np.array([(x,y) for x,y in zip(idx[1], idx[0])])
    x, y = idx[1], idx[0]
    pts = np.stack((x, y), axis=1)
    
    if npts == -1:
        rand_pts = pts
    else:
        rand_pts_idx = np.random.choice(pts.shape[0], npts)
        rand_pts = pts[rand_pts_idx,:]
        
    # # noise check
    # rand_pts = np.vstack((rand_pts, [0, 0]))
    
    # Obtain the 3D verticies (normaize rgb values)
    rgb = color_mask[rand_pts[:,1], rand_pts[:,0]]

    if np.max(rgb) > 1:
        rgb = rgb / 255

    vertices = getattr(obj, f'{object_name}_vertices')
    r = rgb[:, 0] * (np.max(vertices[0]) - np.min(vertices[0])) + np.min(vertices[0])
    g = rgb[:, 1] * (np.max(vertices[1]) - np.min(vertices[1])) + np.min(vertices[1])
    b = rgb[:, 2] * (np.max(vertices[2]) - np.min(vertices[2])) + np.min(vertices[2])
    vtx = np.stack([r, g, b], axis=1)
    
    return rand_pts, vtx

def solve_epnp_cv2(pts2d, pts3d, camera_intrinsics, camera_position):
    pts2d = pts2d.astype('float32')
    pts3d = pts3d.astype('float32')
    camera_intrinsics = camera_intrinsics.astype('float32')
    
    if pts2d.shape[0] < 4:
        predicted_pose = np.eye(4)
        # inliers = []
    else:
        dist_coeffs = np.zeros((4, 1))
        
        # Get a rotation matrix
        predicted_pose = np.eye(4)
        
        # Use EPNP
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pts3d, pts2d, camera_intrinsics, dist_coeffs, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
            
        if success:
            predicted_pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            predicted_pose[:3, 3] = np.squeeze(translation_vector) + np.array(camera_position)
            # logger.debug(len(inliers)) # 50703

    return predicted_pose

def transform_vertices(vertices, transformation_matrix=np.eye(4)):

    ones = np.ones((vertices.shape[0], 1))
    homogeneous_vertices = np.append(vertices, ones, axis=1)
    transformed_vertices = (transformation_matrix @ homogeneous_vertices.T)[:3].T
    
    return transformed_vertices

def color_mesh(vertices):
        colors = copy.deepcopy(vertices)
        # normalize vertices and center it to 0
        colors[0] = (vertices[0] - np.min(vertices[0])) / (np.max(vertices[0]) - np.min(vertices[0])) #- 0.5
        colors[1] = (vertices[1] - np.min(vertices[1])) / (np.max(vertices[1]) - np.min(vertices[1])) #- 0.5
        colors[2] = (vertices[2] - np.min(vertices[2])) / (np.max(vertices[2]) - np.min(vertices[2])) #- 0.5
        colors = colors.T #+ np.array([0.5, 0.5, 0.5])
        
        return colors

def save_image(array, folder, name):
    img = Image.fromarray(array)
    img.save(folder / name)

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