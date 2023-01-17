import __future__
import pickle
import copy
from typing import Tuple, Type

import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
import trimesh
from trimesh import Trimesh
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

def convert2ply(obj: trimesh.Trimesh, filename):
    ply_file = trimesh.exchange.ply.export_ply(obj)
    with open(filename, "wb") as f:
        f.write(ply_file)

def save_obj(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)

def meshwrite(output_filename, mesh):

    fid = open('test/data/nii_001/5997_right_output_mesh_from_df.mesh', 'rb')
    ossicle_mesh = meshread(fid, linesread=False, meshread2=False)
    # mesh.sz = np.array([1,1,1])

    # Unify the data dtypes to be the same as the the ones in the original ossicle_mesh file
    mesh.vertices = mesh.vertices.astype(str(ossicle_mesh.vertices.dtype))
    mesh.triangles = mesh.triangles.astype(str(ossicle_mesh.triangles.dtype))

    with open(output_filename, "wb") as f:
        f.write(mesh.id.T)
        f.write(mesh.numverts.T)
        f.write(mesh.numtris.T)
        f.write(np.int32(-1).T)
        f.write(mesh.orient.T)
        f.write(mesh.dim.T)
        f.write(mesh.sz.T)
        f.write(mesh.color.T)
        # ndarray need to be C-continuous!
        f.write(mesh.vertices.T.tobytes(order='C'))
        f.write(mesh.triangles.T.tobytes(order='C'))
        if hasattr(mesh, "opacity"):
            f.write(mesh.opacity.T)
            if hasattr(mesh, "colormap"):
                f.write(mesh.colormap.numcols.T)
                f.write(mesh.colormap.numverts.T)
                f.write(mesh.colormap.cols.T)
                f.write(mesh.colormap.vertexindexes.T.tobytes(order='C'))

def load_mesh_color(mesh):
    colors = copy.deepcopy(mesh.vertices)
    colors[0] = (mesh.vertices[0] - np.min(mesh.vertices[0])) / (np.max(mesh.vertices[0]) - np.min(mesh.vertices[0])) - 0.5
    colors[1] = (mesh.vertices[1] - np.min(mesh.vertices[1])) / (np.max(mesh.vertices[1]) - np.min(mesh.vertices[1])) - 0.5
    colors[2] = (mesh.vertices[2] - np.min(mesh.vertices[2])) / (np.max(mesh.vertices[2]) - np.min(mesh.vertices[2])) - 0.5
    colors = colors.T + np.array([0.5, 0.5, 0.5])

    return colors

def center_mesh(mesh):
    centered_mesh = copy.deepcopy(mesh)
    centroid = np.mean(mesh.vertices, axis=1)
    centered_mesh.vertices = mesh.vertices - np.expand_dims(centroid, axis=1)
    return centered_mesh

def create_black_bg():
    image = Image.new('RGB', (1920, 1080), color=0)
    image.save("test/data/black_background.jpg")
    
def load_trimesh(meshpath):
    with open(meshpath, "rb") as fid:
        mesh = meshread(fid)
    orient = mesh.orient / np.array([1,2,3])
    mesh.vertices = mesh.vertices * np.expand_dims(mesh.sz, axis=1) * np.expand_dims(orient, axis=1)
    mesh = trimesh.Trimesh(vertices=mesh.vertices.T, faces=mesh.triangles.T)
    return mesh

def compare_two_images():
    # They are different
    image1 = np.array(Image.open("image.png"))
    image2 = np.array(Image.open("test/data/RL_20210304_0.jpg"))
    print("hhh")
    
def color2binary_mask(color_mask):
    # binary_mask = copy.deepcopy(color_mask)
    binary_mask = np.zeros(color_mask[...,:1].shape)
    
    black_pixels_mask = np.all(color_mask == [0, 0, 0], axis=-1)
    non_black_pixels_mask = ~black_pixels_mask
    # non_black_pixels_mask = np.any(color_mask != [0, 0, 0], axis=-1)  

    binary_mask[black_pixels_mask] = [0]
    binary_mask[non_black_pixels_mask] = [1]
    
    return binary_mask

def show_plot(frame, plot, image_white_bg, image_black_bg):
    extent = 0, plot.shape[1], plot.shape[0], 0
    plt.subplot(221)
    plt.imshow(frame, alpha=1, extent=extent, origin="upper")

    plt.subplot(222)
    im1 = plt.imshow(plot, extent=extent, origin="upper")
    im2 = plt.imshow(image_white_bg, alpha=0.5, extent=extent, origin="upper")
    
    plt.subplot(223)
    plt.imshow(image_white_bg, alpha=1, extent=extent, origin="upper")
    
    plt.subplot(224)
    plt.imshow(image_black_bg, alpha=1, extent=extent, origin="upper")
    plt.show()
    
    plt.close()
    
    print('hhh')
    
def count_white_black_pixels(image_grey):
    sought = [0,0,0]
    black  = np.count_nonzero(np.all(image_grey==sought,axis=2))
    print(f"black: {black}")
    
    sought = [255,255,255]
    white  = np.count_nonzero(np.all(image_grey==sought,axis=2))
    print(f"white: {white}")

def check_pixel_in_image(image, pixel_value):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    pixels = [i for i in image.getdata()]
    assert not pixel_value in pixels, f"{pixel_value} in pixels"

def create_2d_3d_pairs(mask:np.ndarray, render:np.ndarray, obj:Type, object_name:str, npts:int=-1):
    
    # Randomly select points in the mask
    idx = np.where(mask == 1)
    
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
    rgb = render[rand_pts[:,1], rand_pts[:,0]]

    if np.max(rgb) > 1:
        rgb = rgb / 255

    vertices = getattr(obj, f'{object_name}_vertices')
    r = rgb[:, 0] * (np.max(vertices[0]) - np.min(vertices[0])) + np.min(vertices[0])
    g = rgb[:, 1] * (np.max(vertices[1]) - np.min(vertices[1])) + np.min(vertices[1])
    b = rgb[:, 2] * (np.max(vertices[2]) - np.min(vertices[2])) + np.min(vertices[2])
    vtx = np.stack([r, g, b], axis=1)
    
    return rand_pts, vtx

def transform_vertices(transformation_matrix, vertices):
    
    # fix the color
    transformation_matrix = np.eye(4)
    
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
    
def cartisian2homogeneous(vertices):
    return np.hstack((vertices, np.ones(vertices.shape[0]).reshape((-1,1))))

def homogeneous2cartisian(homo_vertices):
    return (homo_vertices[:3] / homo_vertices[3]).T

def save_image(array, folder, name):
    img = Image.fromarray(array)
    img.save(folder / name)