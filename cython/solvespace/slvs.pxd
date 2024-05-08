# -*- coding: utf-8 -*-
# cython: language_level=3

"""Wrapper header of Solvespace.

author: Yuan Chang
copyright: Copyright (C) 2016-2019
license: GPLv3+
email: pyslvs@gmail.com
"""

from libc.stdint cimport uint32_t
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector

cdef extern from "slvs.h" nogil:

    ctypedef uint32_t Slvs_hParam
    ctypedef uint32_t Slvs_hEntity
    ctypedef uint32_t Slvs_hConstraint
    ctypedef uint64_t Slvs_hGroup
    ctypedef uint32_t Slvs_hExpr

    # Virtual work plane entity
    Slvs_hEntity SLVS_FREE_IN_3D

    ctypedef struct Slvs_Param:
        Slvs_hParam h
        Slvs_hGroup group
        double val

    # Entity type
    int SLVS_E_POINT_IN_3D
    int SLVS_E_POINT_IN_2D
    int SLVS_E_NORMAL_IN_2D
    int SLVS_E_NORMAL_IN_3D
    int SLVS_E_DISTANCE
    int SLVS_E_WORKPLANE
    int SLVS_E_LINE_SEGMENT
    int SLVS_E_CUBIC
    int SLVS_E_CIRCLE
    int SLVS_E_ARC_OF_CIRCLE

    ctypedef struct Slvs_Entity:
        Slvs_hEntity h
        Slvs_hGroup group
        int type
        Slvs_hEntity wrkpl
        Slvs_hEntity point[4]
        Slvs_hEntity normal
        Slvs_hEntity distance
        Slvs_hParam param[4]
    
    ctypedef struct Slvs_Expr:
        Slvs_hExpr h
        int type
        Slvs_hParam param
        double val
        Slvs_hExpr arg1
        Slvs_hExpr arg2

    int SLVS_C_POINTS_COINCIDENT
    int SLVS_C_PT_PT_DISTANCE
    int SLVS_C_PT_PLANE_DISTANCE
    int SLVS_C_PT_LINE_DISTANCE
    int SLVS_C_PT_FACE_DISTANCE
    int SLVS_C_PT_IN_PLANE
    int SLVS_C_PT_ON_LINE
    int SLVS_C_PT_ON_FACE
    int SLVS_C_EQUAL_LENGTH_LINES
    int SLVS_C_LENGTH_RATIO
    int SLVS_C_EQ_LEN_PT_LINE_D
    int SLVS_C_EQ_PT_LN_DISTANCES
    int SLVS_C_EQUAL_ANGLE
    int SLVS_C_EQUAL_LINE_ARC_LEN
    int SLVS_C_SYMMETRIC
    int SLVS_C_SYMMETRIC_HORIZ
    int SLVS_C_SYMMETRIC_VERT
    int SLVS_C_SYMMETRIC_LINE
    int SLVS_C_AT_MIDPOINT
    int SLVS_C_HORIZONTAL
    int SLVS_C_VERTICAL
    int SLVS_C_DIAMETER
    int SLVS_C_PT_ON_CIRCLE
    int SLVS_C_SAME_ORIENTATION
    int SLVS_C_ANGLE
    int SLVS_C_PARALLEL
    int SLVS_C_PERPENDICULAR
    int SLVS_C_ARC_LINE_TANGENT
    int SLVS_C_CUBIC_LINE_TANGENT
    int SLVS_C_EQUAL_RADIUS
    int SLVS_C_PROJ_PT_DISTANCE
    int SLVS_C_WHERE_DRAGGED
    int SLVS_C_CURVE_CURVE_TANGENT
    int SLVS_C_LENGTH_DIFFERENCE
    int SLVS_C_EQUATIONS


    int SLVS_X_PARAM
    int SLVS_X_CONST
    int SLVS_X_PLUS
    int SLVS_X_MINUS
    int SLVS_X_TIMES
    int SLVS_X_DIV
    int SLVS_X_MIN
    int SLVS_X_MAX
    int SLVS_X_NEGATE
    int SLVS_X_SQRT
    int SLVS_X_SQUARE
    int SLVS_X_SIN
    int SLVS_X_COS
    int SLVS_X_ASIN
    int SLVS_X_ACOS
    int SLVS_X_ABS
    int SLVS_X_SGN
    int SLVS_X_NORM
    int SLVS_X_AND
    int SLVS_X_EQUAL
    int SLVS_X_LTE

    ctypedef struct Slvs_Constraint:
        Slvs_hConstraint h
        Slvs_hGroup group
        int type
        Slvs_hEntity wrkpl
        double valA
        Slvs_hEntity ptA
        Slvs_hEntity ptB
        Slvs_hEntity entityA
        Slvs_hEntity entityB
        Slvs_hEntity entityC
        Slvs_hEntity entityD
        Slvs_hExpr equations
        int other
        int other2

    ctypedef struct Slvs_System:
        Slvs_Param *param
        int params
        Slvs_Entity *entity
        int entities
        Slvs_Constraint *constraint
        int constraints
        Slvs_Expr *expr
        int exprs
        Slvs_hParam dragged[4]
        int calculateFaileds
        Slvs_hConstraint *failed
        int faileds
        int dof
        int iterations
        int result

    void Slvs_Solve(Slvs_System *sys, Slvs_hGroup hg)
    void Slvs_QuaternionU(
        double qw, double qx, double qy, double qz,
        double *x, double *y, double *z
    )
    void Slvs_QuaternionV(
        double qw, double qx, double qy, double qz,
        double *x, double *y, double *z
    )
    void Slvs_QuaternionN(
        double qw, double qx, double qy, double qz,
        double *x, double *y, double *z
    )
    void Slvs_MakeQuaternion(
        double ux, double uy, double uz,
        double vx, double vy, double vz,
        double *qw, double *qx, double *qy, double *qz
    )
    Slvs_Param Slvs_MakeParam(Slvs_hParam h, Slvs_hGroup group, double val)

    Slvs_Expr Slvs_MakeExpr(Slvs_hExpr h, int type, Slvs_hParam param, double val, Slvs_hExpr arg1, Slvs_hExpr arg2)
    Slvs_MakeExpr_Param(Slvs_hExpr h, Slvs_hParam p)
    Slvs_MakeExpr_Const(Slvs_hExpr h, double val)
    Slvs_MakeExpr_Plus(Slvs_hExpr h, Slvs_hExpr arg1, Slvs_hExpr arg2)
    Slvs_MakeExpr_Minus(Slvs_hExpr h, Slvs_hExpr arg1, Slvs_hExpr arg2) 
    Slvs_MakeExpr_Times(Slvs_hExpr h, Slvs_hExpr arg1, Slvs_hExpr arg2) 
    Slvs_MakeExpr_Div(Slvs_hExpr h, Slvs_hExpr arg1, Slvs_hExpr arg2) 
    Slvs_MakeExpr_Min(Slvs_hExpr h, Slvs_hExpr arg1, Slvs_hExpr arg2) 
    Slvs_MakeExpr_Max(Slvs_hExpr h, Slvs_hExpr arg1, Slvs_hExpr arg2) 
    Slvs_MakeExpr_Negate(Slvs_hExpr h, Slvs_hExpr arg1)
    Slvs_MakeExpr_Sqrt(Slvs_hExpr h, Slvs_hExpr arg1)
    Slvs_MakeExpr_Square(Slvs_hExpr h, Slvs_hExpr arg1)
    Slvs_MakeExpr_Sin(Slvs_hExpr h, Slvs_hExpr arg1) 
    Slvs_MakeExpr_Cos(Slvs_hExpr h, Slvs_hExpr arg1)
    Slvs_MakeExpr_ASin(Slvs_hExpr h, Slvs_hExpr arg1)
    Slvs_MakeExpr_ACos(Slvs_hExpr h, Slvs_hExpr arg1)
    Slvs_MakeExpr_Abs(Slvs_hExpr h, Slvs_hExpr arg1) 
    Slvs_MakeExpr_Sgn(Slvs_hExpr h, Slvs_hExpr arg1)
    Slvs_MakeExpr_Norm(Slvs_hExpr h, Slvs_hExpr arg1)
    Slvs_MakeExpr_And(Slvs_hExpr h, Slvs_hExpr arg1, Slvs_hExpr arg2)
    Slvs_MakeExpr_Equal(Slvs_hExpr h, Slvs_hExpr arg1, Slvs_hExpr arg2) 
    Slvs_MakeExpr_LTE(Slvs_hExpr h, Slvs_hExpr arg1, Slvs_hExpr arg2) 
    
    Slvs_Entity Slvs_MakePoint2d(
        Slvs_hEntity h, Slvs_hGroup group,
        Slvs_hEntity wrkpl,
        Slvs_hParam u, Slvs_hParam v
    )
    Slvs_Entity Slvs_MakePoint3d(
        Slvs_hEntity h, Slvs_hGroup group,
        Slvs_hParam x, Slvs_hParam y, Slvs_hParam z
    )
    Slvs_Entity Slvs_MakeNormal3d(
        Slvs_hEntity h, Slvs_hGroup group,
        Slvs_hParam qw, Slvs_hParam qx,
        Slvs_hParam qy, Slvs_hParam qz
    )
    Slvs_Entity Slvs_MakeNormal2d(
        Slvs_hEntity h, Slvs_hGroup group,
        Slvs_hEntity wrkpl
    )
    Slvs_Entity Slvs_MakeDistance(
        Slvs_hEntity h, Slvs_hGroup group,
        Slvs_hEntity wrkpl, Slvs_hParam d
    )
    Slvs_Entity Slvs_MakeLineSegment(
        Slvs_hEntity h, Slvs_hGroup group,
        Slvs_hEntity wrkpl,
        Slvs_hEntity ptA, Slvs_hEntity ptB
    )
    Slvs_Entity Slvs_MakeCubic(
        Slvs_hEntity h, Slvs_hGroup group,
        Slvs_hEntity wrkpl,
        Slvs_hEntity pt0, Slvs_hEntity pt1,
        Slvs_hEntity pt2, Slvs_hEntity pt3
    )
    Slvs_Entity Slvs_MakeArcOfCircle(
        Slvs_hEntity h, Slvs_hGroup group,
        Slvs_hEntity wrkpl,
        Slvs_hEntity normal,
        Slvs_hEntity center,
        Slvs_hEntity start, Slvs_hEntity end
    )
    Slvs_Entity Slvs_MakeCircle(
        Slvs_hEntity h, Slvs_hGroup group,
        Slvs_hEntity wrkpl,
        Slvs_hEntity center,
        Slvs_hEntity normal, Slvs_hEntity radius
    )
    Slvs_Entity Slvs_MakeWorkplane(
        Slvs_hEntity h, Slvs_hGroup group,
        Slvs_hEntity origin, Slvs_hEntity normal
    )
    Slvs_Constraint Slvs_MakeConstraint(
        Slvs_hConstraint h,
        Slvs_hGroup group,
        int type,
        Slvs_hEntity wrkpl,
        double valA,
        Slvs_hEntity ptA,
        Slvs_hEntity ptB,
        Slvs_hEntity entityA,
        Slvs_hEntity entityB
    )

cpdef tuple quaternion_u(double qw, double qx, double qy, double qz)
cpdef tuple quaternion_v(double qw, double qx, double qy, double qz)
cpdef tuple quaternion_n(double qw, double qx, double qy, double qz)
cpdef tuple make_quaternion(double ux, double uy, double uz, double vx, double vy, double vz)


cdef class Params:

    cdef vector[Slvs_hParam] param_list

    cpdef list expressions(self)

    @staticmethod
    cdef Params create(Slvs_hParam *p, size_t count)

cdef class Entity:

    cdef int t
    cdef Slvs_hEntity h, wp
    cdef Slvs_hGroup g
    cdef readonly Params params

    @staticmethod
    cdef Entity create(Slvs_Entity *e)

    cpdef bint is_3d(self)
    cpdef bint is_none(self)
    cpdef bint is_point_2d(self)
    cpdef bint is_point_3d(self)
    cpdef bint is_point(self)
    cpdef bint is_normal_2d(self)
    cpdef bint is_normal_3d(self)
    cpdef bint is_normal(self)
    cpdef bint is_distance(self)
    cpdef bint is_work_plane(self)
    cpdef bint is_line_2d(self)
    cpdef bint is_line_3d(self)
    cpdef bint is_line(self)
    cpdef bint is_cubic(self)
    cpdef bint is_circle(self)
    cpdef bint is_arc(self)


cdef class SolverSystem:

    cdef int dof_v
    cdef int iterations_v
    cdef Slvs_hGroup g
    cdef vector[Slvs_Param] param_list
    cdef vector[Slvs_Entity] entity_list
    cdef vector[Slvs_Expr] expr_list
    cdef vector[Slvs_Constraint] cons_list
    cdef vector[Slvs_hConstraint] failed_list

    cpdef SolverSystem copy(self)
    cpdef void clear(self)
    cpdef void set_group(self, size_t g)
    cpdef int group(self)
    cpdef void set_params(self, Params p, object params)
    cpdef list params(self, Params p)
    cpdef int dof(self)
    cpdef int iterations(self)
    cpdef object constraints(self)
    cpdef list failures(self)
    cdef int solve_c(self) nogil

    cpdef size_t param_len(self)
    cpdef size_t entity_len(self)
    cpdef size_t expr_len(self)
    cpdef size_t cons_len(self)

    cpdef Entity create_2d_base(self)
    cdef Slvs_hParam new_param(self, double val) nogil
    cpdef Params add_param(self, double val)
    cdef Slvs_hEntity eh(self) nogil

    cpdef Entity add_point_2d(self, double u, double v, Entity wp)
    cpdef Entity add_point_3d(self, double x, double y, double z)
    cpdef Entity add_normal_2d(self, Entity wp)
    cpdef Entity add_normal_3d(self, double qw, double qx, double qy, double qz)
    cpdef Entity add_distance(self, double d, Entity wp)
    cpdef Entity add_line_2d(self, Entity p1, Entity p2, Entity wp)
    cpdef Entity add_line_3d(self, Entity p1, Entity p2)
    cpdef Entity add_cubic(self, Entity p1, Entity p2, Entity p3, Entity p4, Entity wp)
    cpdef Entity add_arc(self, Entity nm, Entity ct, Entity start, Entity end, Entity wp)
    cpdef Entity add_circle(self, Entity nm, Entity ct, Entity radius, Entity wp)
    cpdef Entity add_work_plane(self, Entity origin, Entity nm)
    cpdef int add_constraint(
        self,
        int c_type,
        Entity wp,
        double v,
        Entity p1,
        Entity p2,
        Entity e1,
        Entity e2,
        Entity e3 = *,
        Entity e4 = *,
        int other = *,
        int other2 = *,
        int equations = *
    )

    cpdef int describe_system(self)

    cpdef int add_expression_node(self, int op, Slvs_hParam param, double val, Slvs_hExpr arg1, Slvs_hExpr arg2)

    cpdef int coincident(self, Entity e1, Entity e2, Entity wp = *)
    cpdef int distance(self, Entity e1, Entity e2, double value, Entity wp = *)
    cpdef int equal(self, Entity e1, Entity e2, Entity wp = *)
    cpdef int equal_angle(self, Entity e1, Entity e2, Entity e3, Entity e4, Entity wp = *)
    cpdef int equal_point_to_line(self, Entity e1, Entity e2, Entity e3, Entity e4, Entity wp = *)
    cpdef int ratio(self, Entity e1, Entity e2, double value, Entity wp = *)
    cpdef int symmetric(self, Entity e1, Entity e2, Entity e3 = *, Entity wp = *)
    cpdef int symmetric_h(self, Entity e1, Entity e2, Entity wp)
    cpdef int symmetric_v(self, Entity e1, Entity e2, Entity wp)
    cpdef int midpoint(self, Entity e1, Entity e2, Entity wp = *)
    cpdef int horizontal(self, Entity e1, Entity wp)
    cpdef int vertical(self, Entity e1, Entity wp)
    cpdef int diameter(self, Entity e1, double value)
    cpdef int same_orientation(self, Entity e1, Entity e2)
    cpdef int angle(self, Entity e1, Entity e2, double value, Entity wp = *, bint inverse = *)
    cpdef int perpendicular(self, Entity e1, Entity e2, Entity wp = *, bint inverse = *)
    cpdef int parallel(self, Entity e1, Entity e2, Entity wp = *)
    cpdef int tangent(self, Entity e1, Entity e2, Entity wp = *)
    cpdef int distance_proj(self, Entity e1, Entity e2, double value)
    cpdef int dragged(self, Entity e1, Entity wp = *)
    cpdef int length_diff(self, Entity e1, Entity e2, double value, Entity wp = *)
