import pytest
import vision6D as vis

def test_add_mesh_from_mesh_store():
    mesh_source = vis.path.PKG_ROOT.parent / "test" / "data" / "455_right_ossicles_processed.mesh"
    mesh_store = vis.components.MeshStore(window_size=(1080, 1920))
    mesh_store.add_mesh(mesh_source=mesh_source)
