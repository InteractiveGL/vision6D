import __future__
import pickle
import copy

import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
import trimesh
from trimesh import Trimesh
from PIL import Image


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


def load_correct_mesh(filename):
    with open(filename, "rb") as fid:
        mesh = meshread(fid)
    orient = mesh.orient / np.array([1,2,3])
    mesh.vertices = mesh.vertices * np.expand_dims(mesh.sz, axis=1) * np.expand_dims(orient, axis=1)

    return mesh


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
