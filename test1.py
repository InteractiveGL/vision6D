import vtk

# Create a renderer, render window, and interactor
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Create an object (cube)
cubeSource = vtk.vtkCubeSource()

# Create a mapper and actor
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(cubeSource.GetOutputPort())
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Add the actor to the scene
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.2, 0.4)

# Create a transform widget
transformWidget = vtk.vtkAxesTransformWidget()
transformWidget.SetInteractor(renderWindowInteractor)

# Set up the transform representation and target actor
transformRepresentation = vtk.vtkTransform()
transformWidget.SetRepresentation(transformRepresentation)
transformWidget.SetProp3D(actor)

# Start the rendering and interaction
renderWindow.Render()
transformWidget.On()
renderWindowInteractor.Start()
