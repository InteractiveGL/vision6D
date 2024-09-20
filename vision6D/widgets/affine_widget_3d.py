"""Affine widget module."""
from __future__ import annotations
from typing import Tuple
from typing import cast
import numpy as np
import pyvista
import vtk
import sys
import traceback
import warnings

DARK_YELLOW = (0.9647058823529412, 0.7450980392156863, 0)
GLOBAL_AXES = np.eye(3)

def try_callback(func, *args):
    """Wrap a given callback in a try statement.

    Parameters
    ----------
    func : callable
        Callable object.

    *args
        Any arguments.

    """
    try:
        func(*args)
    except Exception:
        etype, exc, tb = sys.exc_info()
        stack = traceback.extract_tb(tb)[1:]
        formatted_exception = 'Encountered issue in callback (most recent call last):\n' + ''.join(
            traceback.format_list(stack) + traceback.format_exception_only(etype, exc),
        ).rstrip('\n')
        warnings.warn(formatted_exception)

def _validate_axes(axes):
    """Validate and normalize input axes.

    Axes are expected to follow the right-hand rule (e.g. third axis is the
    cross product of the first two.

    Parameters
    ----------
    axes : sequence
        The axes to be validated and normalized. Should be of shape (3, 3).

    Returns
    -------
    dict
        The validated and normalized axes.

    """
    axes = np.array(axes)
    if axes.shape != (3, 3):
        raise ValueError("`axes` must be a (3, 3) array.")

    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    if not np.allclose(np.cross(axes[0], axes[1]), axes[2]):
        raise ValueError("`axes` do not follow the right hand rule.")

    return axes


def _check_callable(func, name='callback'):
    """Check if a variable is callable."""
    if func and not callable(func):
        raise TypeError(f"`{name}` must be a callable, not {type(func)}.")
    return func


def _make_quarter_arc():
    """Make a quarter circle centered at the origin."""
    circ = pyvista.Circle(resolution=100)
    circ.faces = np.empty(0, dtype=int)
    circ.lines = np.hstack(([26], np.arange(0, 26)))
    return circ


def get_angle(v1, v2):
    """Compute the angle between two vectors in degrees.

    Parameters
    ----------
    v1 : numpy.ndarray
        First input vector.
    v2 : numpy.ndarray
        Second input vector.

    Returns
    -------
    float
        Angle between vectors in degrees.
    """
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def ray_plane_intersection(start_point, direction, plane_point, normal):
    """Compute the intersection between a ray and a plane.

    Parameters
    ----------
    start_point : ndarray
        Starting point of the ray.
    direction : ndarray
        Direction of the ray.
    plane_point : ndarray
        A point on the plane.
    normal : ndarray
        Normal to the plane.

    Returns
    -------
    ndarray
        Intersection point.
    """
    t_value = np.dot(normal, (plane_point - start_point)) / np.dot(normal, direction)
    return start_point + t_value * direction


class AffineWidget3D:
    def __init__(
        self,
        plotter,
        actor,
        origin=None,
        start=True,
        scale=0.15,
        line_radius=0.02,
        always_visible=True,
        axes_colors=None,
        axes=None,
        release_callback=None,
        interact_callback=None,
    ):
        self._axes = np.eye(4)
        self._axes_inv = np.eye(4)
        self._pl = plotter
        self._main_actor = actor
        self._selected_actor = None
        self._init_position = None
        self._mouse_move_observer = None
        self._left_press_observer = None
        self._left_release_observer = None

        if self._main_actor.user_matrix is None:
            self._main_actor.user_matrix = np.eye(4)
        self._cached_matrix = self._main_actor.user_matrix

        self._arrows = []
        self._circles = []
        self._pressing_down = False
        origin = origin if origin else actor.center
        self._origin = np.array(origin)
        if axes_colors is None:
            axes_colors = (
                pyvista.global_theme.axes.x_color,
                pyvista.global_theme.axes.y_color,
                pyvista.global_theme.axes.z_color,
            )
        self._axes_colors = axes_colors
        self._circ = _make_quarter_arc()
        self._actor_length = self._main_actor.GetLength()
        self._line_radius = line_radius
        self._user_interact_callback = _check_callable(interact_callback)
        self._user_release_callback = _check_callable(release_callback)

        self._init_actors(scale, always_visible)

        # axes must be set after initializing actors
        if axes is not None:
            try:
                _validate_axes(axes)
            except ValueError:
                for actor in self._arrows + self._circles:
                    self._pl.remove_actor(actor)
                raise
            self.axes = axes

        if start:
            self.enable()

    def _init_actors(self, scale, always_visible):
        """Initialize the widget's actors."""
        for ii, color in enumerate(self._axes_colors):
            arrow = pyvista.Arrow(
                (0, 0, 0),
                direction=GLOBAL_AXES[ii],
                scale=self._actor_length * scale * 1.15,
                tip_radius=0.05,
                shaft_radius=self._line_radius,
            )
            self._arrows.append(self._pl.add_mesh(arrow, color=color, lighting=False, render=False))
            axis_circ = self._circ.copy()
            if ii == 0:
                axis_circ = axis_circ.rotate_y(-90)
            elif ii == 1:
                axis_circ = axis_circ.rotate_x(90)
            axis_circ.points *= self._main_actor.GetLength() * (scale * 1.6)
            # axis_circ.points += self._origin
            axis_circ = axis_circ.tube(
                radius=self._line_radius * self._actor_length * scale,
                absolute=True,
                radius_factor=1.0,
            )

            self._circles.append(
                self._pl.add_mesh(
                    axis_circ,
                    color=color,
                    lighting=False,
                    render_lines_as_tubes=True,
                    render=False,
                ),
            )

        # update origin and assign a default user_matrix
        for actor in self._arrows + self._circles:
            matrix = np.eye(4)
            matrix[:3, -1] = self._origin
            actor.user_matrix = matrix

        if always_visible:
            for actor in self._arrows + self._circles:
                actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
                actor.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(0, -20000)

    def sync_widget_to_actor(self):
        """Synchronize the widget's transformation with the actor's user_matrix."""
        # Get the current transformation matrix of the actor
        user_matrix = self._main_actor.user_matrix
        
        if user_matrix is not None:
            # Apply the actor's transformation matrix to all widget components
            for component in self._arrows + self._circles:
                component.user_matrix = user_matrix

            # Redraw the plotter to reflect changes
            self._pl.render()

    def _get_world_coord_rot(self, interactor):
        """Compute rotation coordinates in the object's local space."""
        x, y = interactor.GetLastEventPosition()
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()
        coordinate.SetValue(x, y, 0)
        ren = interactor.GetRenderWindow().GetRenderers().GetFirstRenderer()
        point = np.array(coordinate.GetComputedWorldValue(ren))

        if self._selected_actor:
            index = self._circles.index(self._selected_actor)
            normal = self.axes[index]

            # Transform point to local object space for rotation
            local_point = self.world_to_local(point)
            to_widget = np.array(self._origin - local_point)

            # Perform interaction in local space
            local_point = ray_plane_intersection(local_point, to_widget, self._origin, normal)
            return self.local_to_world(local_point)
        return point
    
    def _get_world_coord_trans(self, interactor):
        """Get the world coordinates (including Z) for translation using the picker."""
        x, y = interactor.GetEventPosition()  # Get the mouse coordinates
        picker = vtk.vtkCellPicker()  # Create a picker
        renderer = interactor.GetRenderWindow().GetRenderers().GetFirstRenderer()

        # Perform picking at the mouse location
        picker.Pick(x, y, 0, renderer)

        # Get the world coordinates of the picked position
        world_position = picker.GetPickPosition()  # Returns (x, y, z) in world space

        if picker.GetCellId() == -1: return None

        return np.array(world_position)

    def _move_callback(self, interactor, _event):
        """Process actions for the move mouse event."""
        click_x, click_y = interactor.GetEventPosition()
        click_z = 0
        picker = interactor.GetPicker()
        renderer = interactor.GetInteractorStyle()._parent()._plotter.iren.get_poked_renderer()
        picker.Pick(click_x, click_y, click_z, renderer)
        actor = picker.GetActor()
        current_pos = self._get_world_coord_trans(interactor)
        matrix = self._main_actor.user_matrix

        if self._pressing_down:
            if self._selected_actor in self._arrows:
                index = self._arrows.index(self._selected_actor)
                if current_pos is not None:
                    diff = current_pos - self.init_position
                    trans_matrix = np.eye(4)
                    trans_matrix[:3, -1] = self.axes[index] * np.dot(diff, self.axes[index])
                    matrix = trans_matrix @ self._cached_matrix
            # elif self._selected_actor in self._circles:
            #     current_pos = self._get_world_coord_rot(interactor)
            #     index = self._circles.index(self._selected_actor)
            #     vec_current = current_pos - self._origin
            #     vec_init = self.init_position - self._origin
            #     normal = self.axes[index]
            #     vec_current = vec_current - np.dot(vec_current, normal) * normal
            #     vec_init = vec_init - np.dot(vec_init, normal) * normal
            #     vec_current /= np.linalg.norm(vec_current)
            #     vec_init /= np.linalg.norm(vec_init)
            #     angle = get_angle(vec_init, vec_current)
            #     cross = np.cross(vec_init, vec_current)
            #     if cross[index] < 0:
            #         angle = -angle

            #     trans = vtk.vtkTransform()
            #     trans.Translate(self._origin)
            #     trans.RotateWXYZ(
            #         angle,
            #         self._axes[index][0],
            #         self._axes[index][1],
            #         self._axes[index][2],
            #     )
            #     trans.Translate(-self._origin)
            #     trans.Update()
            #     rot_matrix = pyvista.array_from_vtkmatrix(trans.GetMatrix())
            #     matrix = rot_matrix @ self._cached_matrix

            if self._user_interact_callback:
                try_callback(self._user_interact_callback, self._main_actor.user_matrix)

            self._main_actor.user_matrix = matrix

            self.sync_widget_to_actor()

        elif self._selected_actor and self._selected_actor is not actor:
            # Return the color of the currently selected actor to normal and
            # deselect it
            if self._selected_actor in self._arrows:
                index = self._arrows.index(self._selected_actor)
            elif self._selected_actor in self._circles:
                index = self._circles.index(self._selected_actor)
            self._selected_actor.prop.color = self._axes_colors[index]
            self._selected_actor = None

        # Highlight the actor if there is no selected actor
        if actor and not self._selected_actor:
            if actor in self._arrows:
                index = self._arrows.index(actor)
                self._arrows[index].prop.color = DARK_YELLOW
                actor.prop.color = DARK_YELLOW
                self._selected_actor = actor
            elif actor in self._circles:
                index = self._circles.index(actor)
                self._circles[index].prop.color = DARK_YELLOW
                actor.prop.color = DARK_YELLOW
                self._selected_actor = actor
        self._pl.render()

    def _press_callback(self, interactor, _event):
        """Process actions for the mouse button press event."""
        if self._selected_actor:
            self._pl.enable_trackball_actor_style()
            self._pressing_down = True
            if self._selected_actor in self._circles:
                self.init_position = self._get_world_coord_rot(interactor)
            else:
                self.init_position = self._main_actor.user_matrix[:3, -1]

    def _release_callback(self, _interactor, _event):
        """Process actions for the mouse button release event."""
        self._pl.enable_trackball_actor_style()
        self._pressing_down = False
        self._cached_matrix = self._main_actor.user_matrix
        if self._user_release_callback:
            try_callback(self._user_release_callback, self._main_actor.user_matrix)

        self.sync_widget_to_actor()

    def _reset(self):
        """Reset the actor and cached transform."""
        self._main_actor.user_matrix = np.eye(4)
        self._cached_matrix = np.eye(4)

    @property
    def axes(self):
        """Return or set the axes of the widget.

        The axes will be checked for orthogonality. Non-orthogonal axes will
        raise a ``ValueError``

        Returns
        -------
        numpy.ndarray
            ``(3, 3)`` array of axes.

        """
        return self._axes[:3, :3]

    @axes.setter
    def axes(self, axes):  # numpydoc ignore=GL08
        mat = np.eye(4)
        mat[:3, :3] = _validate_axes(axes)
        mat[:3, -1] = self.origin
        self._axes = mat
        self._axes_inv = np.linalg.inv(self._axes)
        for actor in self._arrows + self._circles:
            matrix = actor.user_matrix
            # Be sure to use the inverse here
            matrix[:3, :3] = self._axes_inv[:3, :3]
            actor.user_matrix = matrix

    @property
    def origin(self) -> tuple[float, float, float]:
        """Origin of the widget.

        This is where the origin of the widget will be located and where the
        actor will be rotated about.

        Returns
        -------
        tuple
            Widget origin.

        """
        return cast(Tuple[float, float, float], tuple(self._origin))

    @origin.setter
    def origin(self, value):  # numpydoc ignore=GL08
        value = np.array(value)
        diff = value - self._origin

        for actor in self._circles + self._arrows:
            if actor.user_matrix is None:
                actor.user_matrix = np.eye(4)
            matrix = actor.user_matrix
            matrix[:3, -1] += diff
            actor.user_matrix = matrix

        self._origin = value

    def enable(self):
        """Enable the widget."""
        if not self._pl._picker_in_use:
            self._pl.enable_mesh_picking(show_message=False, show=False, picker='hardware')
        self._mouse_move_observer = self._pl.iren.add_observer(
            "MouseMoveEvent",
            self._move_callback,
        )
        self._left_press_observer = self._pl.iren.add_observer(
            "LeftButtonPressEvent",
            self._press_callback,
            interactor_style_fallback=False,
        )
        self._left_release_observer = self._pl.iren.add_observer(
            "LeftButtonReleaseEvent",
            self._release_callback,
            interactor_style_fallback=False,
        )

    def disable(self):
        """Disable the widget."""
        self._pl.disable_picking()
        if self._mouse_move_observer:
            self._pl.iren.remove_observer(self._mouse_move_observer)
        if self._left_press_observer:
            self._pl.iren.remove_observer(self._left_press_observer)
        if self._left_release_observer:
            self._pl.iren.remove_observer(self._left_release_observer)

    def remove(self):
        """Disable and delete all actors of this widget."""
        self.disable()
        for actor in self._circles + self._arrows:
            self._pl.remove_actor(actor)
        self._circles = []
        self._arrows = []