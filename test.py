# import pyvista as pv
# pl = pv.Plotter()
# actor = pl.add_mesh(pv.Sphere())
# widget = pv.AffineWidget3D(pl, actor)
# pl.show()

import pyvista as pv
sphere = pv.Sphere()
plotter = pv.Plotter(shape=(1, 2))
_ = plotter.add_mesh(sphere, show_edges=True)
plotter.subplot(0, 1)
_ = plotter.add_mesh(sphere, show_edges=True)
_ = plotter.add_camera3d_widget()
plotter.show(cpos=plotter.camera_position)