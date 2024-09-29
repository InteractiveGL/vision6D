import trimesh
import pyvista as pv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class AbstractData(ABC):
    path: str=None
    name: str=None
    source_obj: trimesh.Trimesh=None
    pv_obj: pv.PolyData=None
    actor: pv.Actor=None
    opacity: float=0.9
    previous_opacity: float=0.9
    opacity_spinbox: Optional[str]=None
    width: Optional[int] = None
    height: Optional[int] = None
    mirror_x: bool = False
    mirror_y: bool = False
    fx: Optional[float] = 572.4114
    fy: Optional[float] = 573.57043
    cx: Optional[float] = 325.2611
    cy: Optional[float] = 242.04899
    cam_viewup: Optional[Tuple[int, int, int]] = (0, -1, 0)

    @abstractmethod
    def clear_attributes(self):
        for field in self.__dataclass_fields__:
            setattr(self, field, None)
