# -*- coding: utf-8 -*-

from typing import Tuple, List, Sequence, Counter, ClassVar
from enum import IntEnum, auto, Enum

def quaternion_u(
    qw: float,
    qx: float,
    qy: float,
    qz: float
) -> Tuple[float, float, float]:
    ...

def quaternion_v(
    qw: float,
    qx: float,
    qy: float,
    qz: float
) -> Tuple[float, float, float]:
    ...

def quaternion_n(
    qw: float,
    qx: float,
    qy: float,
    qz: float
) -> Tuple[float, float, float]:
    ...

def make_quaternion(
    ux: float,
    uy: float,
    uz: float,
    vx: float,
    vy: float,
    vz: float
) -> Tuple[float, float, float, float]:
    ...

class Expression:
    class Op(Enum):
        PLUS = '+'
        MINUS = '-'
        TIMES = '*'
        DIVIDE = '/'
        EQUAL = '=='
        LTE = '<='
        ABS = '||'
        PARAM = 'p'
        CONST = 'c'
        MIN = 'min'
        MAX = 'max'
        NORM = 'norm'
        AND = 'and'
    
    def __init__(self, *val, **kwargs):
        ...

    def abs(self):
        ...
    
    def min(self, other):
        ...
    
    def max(self, other):
        ...
    
    def conjunction(self, other):
        ...

    def __invert__(self):
        ...
    
    def __neg__(self):
        ...
    
    def __pos__(self):
        ...
    
    def binary_op(self, other, op, right=False):
        ...

    def __le__(self, other):
        ...
    
    def __ge__(self, other):
        ...

    def __eq__(self, other):
        ...
    
    def __truediv__(self, other):
        ...
    
    def __add__(self, other):
        ...
    
    def __sub__(self, other):
        ...
    
    def __mul__(self, other):
        ...
    
    def __radd__(self, other):
        ...
    
    def __rsub__(self, other):
        ...
    
    def __rmul__(self, other):
        ...
    
    def __rtruediv__(self, other):
        ...
    
    def __repr__(self) -> str:
        ...
    
    def __str__(self) -> str:
        ...

def minimum(*exprs: List[Expression]):
    ...

def maximum(*exprs: List[Expression]):
    ...

class Constraint(IntEnum):
    """Symbol of the constraint types."""
    POINTS_COINCIDENT = 100000
    PT_PT_DISTANCE = auto()
    PT_PLANE_DISTANCE = auto()
    PT_LINE_DISTANCE = auto()
    PT_FACE_DISTANCE = auto()
    PT_IN_PLANE = auto()
    PT_ON_LINE = auto()
    PT_ON_FACE = auto()
    EQUAL_LENGTH_LINES = auto()
    LENGTH_RATIO = auto()
    EQ_LEN_PT_LINE_D = auto()
    EQ_PT_LN_DISTANCES = auto()
    EQUAL_ANGLE = auto()
    EQUAL_LINE_ARC_LEN = auto()
    SYMMETRIC = auto()
    SYMMETRIC_HORIZ = auto()
    SYMMETRIC_VERT = auto()
    SYMMETRIC_LINE = auto()
    AT_MIDPOINT = auto()
    HORIZONTAL = auto()
    VERTICAL = auto()
    DIAMETER = auto()
    PT_ON_CIRCLE = auto()
    SAME_ORIENTATION = auto()
    ANGLE = auto()
    PARALLEL = auto()
    PERPENDICULAR = auto()
    ARC_LINE_TANGENT = auto()
    CUBIC_LINE_TANGENT = auto()
    EQUAL_RADIUS = auto()
    PROJ_PT_DISTANCE = auto()
    WHERE_DRAGGED = auto()
    CURVE_CURVE_TANGENT = auto()
    LENGTH_DIFFERENCE = auto()
    EQUATIONS = auto()


class ResultFlag(IntEnum):
    """Symbol of the result flags."""
    OKAY = 0
    INCONSISTENT = auto()
    DIDNT_CONVERGE = auto()
    TOO_MANY_UNKNOWNS = auto()
    REDUNDANT_OKAY = auto()

class Params:
    pass

class Entity:

    FREE_IN_3D: ClassVar[Entity] = ...
    NONE: ClassVar[Entity] = ...
    params: Params

    def is_3d(self) -> bool:
        ...

    def is_none(self) -> bool:
        ...

    def is_point_2d(self) -> bool:
        ...

    def is_point_3d(self) -> bool:
        ...

    def is_point(self) -> bool:
        ...

    def is_normal_2d(self) -> bool:
        ...

    def is_normal_3d(self) -> bool:
        ...

    def is_normal(self) -> bool:
        ...

    def is_distance(self) -> bool:
        ...

    def is_work_plane(self) -> bool:
        ...

    def is_line_2d(self) -> bool:
        ...

    def is_line_3d(self) -> bool:
        ...

    def is_line(self) -> bool:
        ...

    def is_cubic(self) -> bool:
        ...

    def is_circle(self) -> bool:
        ...

    def is_arc(self) -> bool:
        ...

class SolverSystem:

    def __init__(self) -> None:
        """Create a solver system."""
        ...

    def __reduce__(self):
        ...

    def entity(self, i: int) -> Entity:
        ...

    def copy(self) -> SolverSystem:
        ...

    def clear(self) -> None:
        ...

    def set_group(self, g: int) -> None:
        ...

    def group(self) -> int:
        ...

    def set_params(self, p: Params, params: Sequence[float]) -> None:
        ...

    def params(self, p: Params) -> List[float]:
        ...

    def dof(self) -> int:
        ...

    def constraints(self) -> Counter[str]:
        ...

    def failures(self) -> List[int]:
        ...

    def solve(self) -> int:
        ...

    def param_len(self) -> int:
        ...

    def entity_len(self) -> int:
        ...

    def expr_len(self) -> int:
        ...

    def cons_len(self) -> int:
        ...

    def create_2d_base(self) -> Entity:
        ...

    def add_param(self, v: float) -> Params:
        ...

    def add_point_2d(self, u: float, v: float, wp: Entity) -> Entity:
        ...

    def add_point_3d(self, x: float, y: float, z: float) -> Entity:
        ...

    def add_normal_2d(self, wp: Entity) -> Entity:
        ...

    def add_normal_3d(self, qw: float, qx: float, qy: float, qz: float) -> Entity:
        ...

    def add_distance(self, d: float, wp: Entity) -> Entity:
        ...

    def add_line_2d(self, p1: Entity, p2: Entity, wp: Entity) -> Entity:
        ...

    def add_line_3d(self, p1: Entity, p2: Entity) -> Entity:
        ...

    def add_cubic(self, p1: Entity, p2: Entity, p3: Entity, p4: Entity, wp: Entity) -> Entity:
        ...

    def add_arc(self, nm: Entity, ct: Entity, start: Entity, end: Entity, wp: Entity) -> Entity:
        ...

    def add_circle(self, nm: Entity, ct: Entity, radius: Entity, wp: Entity) -> Entity:
        ...

    def add_work_plane(self, origin: Entity, nm: Entity) -> Entity:
        ...

    def add_constraint(
        self,
        c_type: int,
        wp: Entity,
        v: float,
        p1: Entity,
        p2: Entity,
        e1: Entity,
        e2: Entity,
        e3: Entity = Entity.NONE,
        e4: Entity = Entity.NONE,
        other: int = 0,
        other2: int = 0,
        equations: int = 0
    ) -> int:
        ...

    def coincident(self, e1: Entity, e2: Entity, wp: Entity = Entity.FREE_IN_3D) -> int:
        ...

    def distance(
        self,
        e1: Entity,
        e2: Entity,
        value: float,
        wp: Entity = Entity.FREE_IN_3D
    ) -> int:
        ...

    def equal(self, e1: Entity, e2: Entity, wp: Entity = Entity.FREE_IN_3D) -> int:
        ...

    def equal_angle(
        self,
        e1: Entity,
        e2: Entity,
        e3: Entity,
        e4: Entity,
        wp: Entity = Entity.FREE_IN_3D
    ) -> int:
        ...

    def equal_point_to_line(
        self,
        e1: Entity,
        e2: Entity,
        e3: Entity,
        e4: Entity,
        wp: Entity = Entity.FREE_IN_3D
    ) -> int:
        ...

    def ratio(self, e1: Entity, e2: Entity, value: float, wp: Entity = Entity.FREE_IN_3D) -> int:
        ...

    def symmetric(self, e1: Entity, e2: Entity, e3: Entity = Entity.NONE, wp: Entity = Entity.FREE_IN_3D) -> int:
        ...

    def symmetric_h(self, e1: Entity, e2: Entity, wp: Entity) -> int:
        ...

    def symmetric_v(self, e1: Entity, e2: Entity, wp: Entity) -> int:
        ...

    def midpoint(self, e1: Entity, e2: Entity, wp: Entity = Entity.FREE_IN_3D) -> int:
        ...

    def horizontal(self, e1: Entity, wp: Entity) -> int:
        ...

    def vertical(self, e1: Entity, wp: Entity) -> int:
        ...

    def diameter(self, e1: Entity, value: float) -> int:
        ...

    def same_orientation(self, e1: Entity, e2: Entity) -> int:
        ...

    def angle(self, e1: Entity, e2: Entity, value: float, wp: Entity = Entity.FREE_IN_3D, inverse: bool = False) -> int:
        ...

    def perpendicular(self, e1: Entity, e2: Entity, wp: Entity = Entity.FREE_IN_3D, inverse: bool = False) -> int:
        ...

    def parallel(self, e1: Entity, e2: Entity, wp: Entity = Entity.FREE_IN_3D) -> int:
        ...

    def tangent(self, e1: Entity, e2: Entity, wp: Entity = Entity.FREE_IN_3D) -> int:
        ...

    def distance_proj(self, e1: Entity, e2: Entity, value: float) -> int:
        ...

    def dragged(self, e1: Entity, wp: Entity = Entity.FREE_IN_3D) -> int:
        ...
