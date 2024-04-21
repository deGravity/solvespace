//-----------------------------------------------------------------------------
// A library wrapper around SolveSpace, to permit someone to use its constraint
// solver without coupling their program too much to SolveSpace's internals.
//
// Copyright 2008-2013 Jonathan Westhues.
//-----------------------------------------------------------------------------
#include "solvespace.h"
#define EXPORT_DLL
#include <slvs.h>
#include <iostream>
#include <vector>

Sketch SolveSpace::SK = {};
static System SYS;

void SolveSpace::Platform::FatalError(const std::string &message) {
    fprintf(stderr, "%s", message.c_str());
    abort();
}

void Group::GenerateEquations(IdList<Equation,hEquation> *) {
    // Nothing to do for now.
}

extern "C" {

void Slvs_QuaternionU(double qw, double qx, double qy, double qz,
                         double *x, double *y, double *z)
{
    Quaternion q = Quaternion::From(qw, qx, qy, qz);
    Vector v = q.RotationU();
    *x = v.x;
    *y = v.y;
    *z = v.z;
}

void Slvs_QuaternionV(double qw, double qx, double qy, double qz,
                         double *x, double *y, double *z)
{
    Quaternion q = Quaternion::From(qw, qx, qy, qz);
    Vector v = q.RotationV();
    *x = v.x;
    *y = v.y;
    *z = v.z;
}

void Slvs_QuaternionN(double qw, double qx, double qy, double qz,
                         double *x, double *y, double *z)
{
    Quaternion q = Quaternion::From(qw, qx, qy, qz);
    Vector v = q.RotationN();
    *x = v.x;
    *y = v.y;
    *z = v.z;
}

void Slvs_MakeQuaternion(double ux, double uy, double uz,
                         double vx, double vy, double vz,
                         double *qw, double *qx, double *qy, double *qz)
{
    Vector u = Vector::From(ux, uy, uz),
           v = Vector::From(vx, vy, vz);
    Quaternion q = Quaternion::From(u, v);
    *qw = q.w;
    *qx = q.vx;
    *qy = q.vy;
    *qz = q.vz;
}

int param_handle = 1;
class SParam {
public:
    Param p;
    Expr *e;

    SParam(double value) {
        p     = Param();
        p.h.v = param_handle++;
        p.val = value;
        SK.param.Add(&p);
        SYS.param.Add(&p);
        e = Expr::From(p.h);
    }

    double v() {
        return SK.GetParam(p.h)->val;
    }
};

void Test_Ineq()
{
    std::cout << "Running Test_Ineq" << std::endl;

   
    ConstraintBase c = {};


    // Triangle Points
    auto x1 = SParam(1);
    auto x2 = SParam(3);
    auto x3 = SParam(2);

    auto y1 = SParam(0);
    auto y2 = SParam(0);
    auto y3 = SParam(1);

    // Constrain Equilateral
    auto d12 = x1.e->Minus(x2.e)->Square()->Plus(y1.e->Minus(y2.e)->Square())->Sqrt();
    auto d13 = x1.e->Minus(x3.e)->Square()->Plus(y1.e->Minus(y3.e)->Square())->Sqrt();
    auto d23 = x3.e->Minus(x2.e)->Square()->Plus(y3.e->Minus(y2.e)->Square())->Sqrt();

    c.AddEq(&SYS.eq, d12->Minus(Expr::From(1.0)), 0);
    c.AddEq(&SYS.eq, d13->Minus(Expr::From(1.0)), 1);
    c.AddEq(&SYS.eq, d23->Minus(Expr::From(1.0)), 2);

    // Bounding Box
    auto bottom = y1.e->Min(y2.e->Min(y3.e));
    auto top    = y1.e->Max(y2.e->Max(y3.e));
    auto left   = x1.e->Min(x2.e->Min(x3.e));
    auto right  = x1.e->Max(x2.e->Max(x3.e));

    // Constrain Bounding Box to QIII
    // top + s_top == 0, right + s_right == 0, s_top = -top, s_right = -right
    auto s_top = SParam(-top->Eval());
    c.AddEq(&SYS.eq, top->Minus(Expr::From(0.0))->Plus(s_top.e), 3);
    c.AddEq(&SYS.eq, s_top.e->Minus(s_top.e->Abs()), 4);

    auto s_right = SParam(-right->Eval());
    c.AddEq(&SYS.eq, right->Minus(Expr::From(0.0))->Plus(s_right.e), 5);
    c.AddEq(&SYS.eq, s_right.e->Minus(s_right.e->Abs()), 6);
    
    
    Group g = {};
    g.h.v   = 2;

    List<hConstraint> bad = {};

    // Now we're finally ready to solve!
    bool andFindBad = false; // ssys->calculateFaileds ? true : false;
    int dof         = 0;

    SolveResult how = SYS.Solve(&g, NULL, &dof, &bad, andFindBad, false);
    
    std::string success_message;
    switch(how) {
    case SolveResult::OKAY: success_message = "OKAY"; break;
    case SolveResult::DIDNT_CONVERGE: success_message = "DIDNT_CONVERGE"; break;
    case SolveResult::REDUNDANT_OKAY: success_message = "REDUNDANT_OKAY"; break;
    case SolveResult::REDUNDANT_DIDNT_CONVERGE: success_message = "REDUNDANT_DIDNT_CONVERGE"; break;
    case SolveResult::TOO_MANY_UNKNOWNS: success_message = "TOO_MANY_UNKNOWNS"; break;
    }

    std::cout << "Solve Success = " << success_message << std::endl;
    std::cout << "\tp1 = (" << x1.v() << " , " << y1.v() << ")" << std::endl;
    std::cout << "\tp2 = (" << x2.v() << " , " << y2.v() << ")" << std::endl;
    std::cout << "\tp3 = (" << x3.v() << " , " << y3.v() << ")" << std::endl << std::endl;
    std::cout << "\ttop = " << top->Eval() << "  s_top = " << s_top.v() << std::endl;
    std::cout << "\tright = " << right->Eval() << "  s_right = " << s_right.v() << std::endl;
}

Expr *GetExpression(Slvs_System *ssys, Slvs_Expr *sexpr, ConstraintBase *c) {
    Expr *arg1;
    Expr *arg2;

    // Base Cases, and populate the second argument if binary operation    
    switch(sexpr->type) {
    case SLVS_X_PARAM: {
        hParam hp;
        hp.v = sexpr->param;
        return Expr::From(hp); // TODO -- do we need to go through the sketch first?
    }
    case SLVS_X_CONST: return Expr::From(sexpr->val);
    case SLVS_X_PLUS:
    case SLVS_X_MINUS:
    case SLVS_X_TIMES:
    case SLVS_X_DIV:
    case SLVS_X_MIN:
    case SLVS_X_MAX: arg2 = GetExpression(ssys, &ssys->expr[sexpr->arg2], c); break;
    default: break;
    }
    // All non-base cases have an arg1
    arg1 = GetExpression(ssys, &ssys->expr[sexpr->arg1], c);

    switch(sexpr->type) {
    case SLVS_X_PLUS: return arg1->Plus(arg2);
    case SLVS_X_MINUS: return arg1->Minus(arg2);
    case SLVS_X_TIMES: return arg1->Times(arg2);
    case SLVS_X_DIV: return arg1->Div(arg2);
    case SLVS_X_MIN: return arg1->Min(arg2);
    case SLVS_X_MAX: return arg1->Max(arg2);
    case SLVS_X_NEGATE: return arg1->Negate();
    case SLVS_X_SQRT: return arg1->Sqrt();
    case SLVS_X_SQUARE: return arg1->Square();
    case SLVS_X_SIN: return arg1->Sin();
    case SLVS_X_COS: return arg1->Cos();
    case SLVS_X_ASIN: return arg1->ASin();
    case SLVS_X_ACOS: return arg1->ACos();
    case SLVS_X_ABS: return arg1->Abs();
    case SLVS_X_SGN: return arg1->Sgn();
    case SLVS_X_NORM: return arg1->Norm();
    default: dbp("not an arithmetic operator, param, or const %d", sexpr->type); return Expr::From(1); // TODO - better default?
    }
}

void AddEquality(Slvs_System *ssys, Slvs_Expr *sexpr, IdList<Param, hParam> *genParams,
                 ConstraintBase *c, int idx) {
    auto lhs = GetExpression(ssys, &ssys->expr[sexpr->arg1], c);
    auto rhs = GetExpression(ssys, &ssys->expr[sexpr->arg2], c);
    c->AddEq(&SYS.eq, lhs->Minus(rhs), idx);
}

void AddInequality(Slvs_System *ssys, Slvs_Expr *sexpr, IdList<Param, hParam> *genParams,
                 ConstraintBase *c, int idx) {
    auto lhs = GetExpression(ssys, &ssys->expr[sexpr->arg1], c);
    auto rhs = GetExpression(ssys, &ssys->expr[sexpr->arg2], c);
    Param s  = {};
    s.val    = rhs->Minus(lhs)->Eval(); // Initialize slack var so original ineq holds
    SK.param.AddAndAssignId(&s);
    SYS.param.Add(&s);
    auto slack = Expr::From(s.h);
    c->AddEq(&SYS.eq, lhs->Minus(rhs)->Plus(slack), idx); // lhs <= rhs
    c->AddEq(&SYS.eq, slack->Minus(slack->Abs()), idx + 1); // slack >= 0
}

void AddRelation(Slvs_System *ssys, Slvs_Rel *srel, IdList<Param, hParam> *genParams, ConstraintBase *c) {
    // Get a list of expression pointers (one for each AND-ed equality or inequality
    Slvs_Expr *expr;
    std::vector<Slvs_hExpr> equations;
    std::vector<Slvs_hExpr> to_explore;
    to_explore.push_back(srel->expr);
    while(!to_explore.empty()) {
        Slvs_hExpr curr = to_explore.back();
        to_explore.pop_back();
        expr = &ssys->expr[curr];
        switch(expr->type) {
        case SLVS_X_AND: 
            to_explore.push_back(expr->arg1); 
            to_explore.push_back(expr->arg2);
            break;
        case SLVS_X_EQUAL:
        case SLVS_X_LTE: 
            equations.push_back(curr); 
            break;
        default: dbp("not an equivalence relation %d", expr->type); return;
        }
    }
    int idx = 0;
    for(auto eq : equations) {
        expr = &ssys->expr[eq];
        switch(expr->type) {
        case SLVS_X_EQUAL:
            AddEquality(ssys, expr, genParams, c, idx);
            idx += 1;
            break;
        case SLVS_X_LTE: AddInequality(ssys, expr, genParams, c, idx); 
            idx += 2; // Inequalities have a secondary slack variable constraint
            break;
            
        }
    }
}

void Slvs_Solve(Slvs_System *ssys, Slvs_hGroup shg)
{
    int i;
    for(i = 0; i < ssys->params; i++) {
        Slvs_Param *sp = &(ssys->param[i]);
        Param p = {};

        p.h.v = sp->h;
        p.val = sp->val;
        SK.param.Add(&p);
        if(shg % sp->group == 0) { // Hack - Factorizable Groups
            SYS.param.Add(&p);
        }
    }

    for(i = 0; i < ssys->entities; i++) {
        Slvs_Entity *se = &(ssys->entity[i]);
        EntityBase e = {};

        switch(se->type) {
case SLVS_E_POINT_IN_3D:        e.type = Entity::Type::POINT_IN_3D; break;
case SLVS_E_POINT_IN_2D:        e.type = Entity::Type::POINT_IN_2D; break;
case SLVS_E_NORMAL_IN_3D:       e.type = Entity::Type::NORMAL_IN_3D; break;
case SLVS_E_NORMAL_IN_2D:       e.type = Entity::Type::NORMAL_IN_2D; break;
case SLVS_E_DISTANCE:           e.type = Entity::Type::DISTANCE; break;
case SLVS_E_WORKPLANE:          e.type = Entity::Type::WORKPLANE; break;
case SLVS_E_LINE_SEGMENT:       e.type = Entity::Type::LINE_SEGMENT; break;
case SLVS_E_CUBIC:              e.type = Entity::Type::CUBIC; break;
case SLVS_E_CIRCLE:             e.type = Entity::Type::CIRCLE; break;
case SLVS_E_ARC_OF_CIRCLE:      e.type = Entity::Type::ARC_OF_CIRCLE; break;

default: dbp("bad entity type %d", se->type); return;
        }
        e.h.v           = se->h;
        e.group.v       = se->group;
        e.workplane.v   = se->wrkpl;
        e.point[0].v    = se->point[0];
        e.point[1].v    = se->point[1];
        e.point[2].v    = se->point[2];
        e.point[3].v    = se->point[3];
        e.normal.v      = se->normal;
        e.distance.v    = se->distance;
        e.param[0].v    = se->param[0];
        e.param[1].v    = se->param[1];
        e.param[2].v    = se->param[2];
        e.param[3].v    = se->param[3];

        SK.entity.Add(&e);
    }
    IdList<Param, hParam> params = {};
    for(i = 0; i < ssys->constraints; i++) {
        Slvs_Constraint *sc = &(ssys->constraint[i]);
        ConstraintBase c = {};

        Constraint::Type t;
        switch(sc->type) {
case SLVS_C_POINTS_COINCIDENT:  t = Constraint::Type::POINTS_COINCIDENT; break;
case SLVS_C_PT_PT_DISTANCE:     t = Constraint::Type::PT_PT_DISTANCE; break;
case SLVS_C_PT_PLANE_DISTANCE:  t = Constraint::Type::PT_PLANE_DISTANCE; break;
case SLVS_C_PT_LINE_DISTANCE:   t = Constraint::Type::PT_LINE_DISTANCE; break;
case SLVS_C_PT_FACE_DISTANCE:   t = Constraint::Type::PT_FACE_DISTANCE; break;
case SLVS_C_PT_IN_PLANE:        t = Constraint::Type::PT_IN_PLANE; break;
case SLVS_C_PT_ON_LINE:         t = Constraint::Type::PT_ON_LINE; break;
case SLVS_C_PT_ON_FACE:         t = Constraint::Type::PT_ON_FACE; break;
case SLVS_C_EQUAL_LENGTH_LINES: t = Constraint::Type::EQUAL_LENGTH_LINES; break;
case SLVS_C_LENGTH_RATIO:       t = Constraint::Type::LENGTH_RATIO; break;
case SLVS_C_ARC_ARC_LEN_RATIO:  t = Constraint::Type::ARC_ARC_LEN_RATIO; break;
case SLVS_C_ARC_LINE_LEN_RATIO: t = Constraint::Type::ARC_LINE_LEN_RATIO; break;
case SLVS_C_EQ_LEN_PT_LINE_D:   t = Constraint::Type::EQ_LEN_PT_LINE_D; break;
case SLVS_C_EQ_PT_LN_DISTANCES: t = Constraint::Type::EQ_PT_LN_DISTANCES; break;
case SLVS_C_EQUAL_ANGLE:        t = Constraint::Type::EQUAL_ANGLE; break;
case SLVS_C_EQUAL_LINE_ARC_LEN: t = Constraint::Type::EQUAL_LINE_ARC_LEN; break;
case SLVS_C_LENGTH_DIFFERENCE:  t = Constraint::Type::LENGTH_DIFFERENCE; break;
case SLVS_C_ARC_ARC_DIFFERENCE: t = Constraint::Type::ARC_ARC_DIFFERENCE; break;
case SLVS_C_ARC_LINE_DIFFERENCE:t = Constraint::Type::ARC_LINE_DIFFERENCE; break;
case SLVS_C_SYMMETRIC:          t = Constraint::Type::SYMMETRIC; break;
case SLVS_C_SYMMETRIC_HORIZ:    t = Constraint::Type::SYMMETRIC_HORIZ; break;
case SLVS_C_SYMMETRIC_VERT:     t = Constraint::Type::SYMMETRIC_VERT; break;
case SLVS_C_SYMMETRIC_LINE:     t = Constraint::Type::SYMMETRIC_LINE; break;
case SLVS_C_AT_MIDPOINT:        t = Constraint::Type::AT_MIDPOINT; break;
case SLVS_C_HORIZONTAL:         t = Constraint::Type::HORIZONTAL; break;
case SLVS_C_VERTICAL:           t = Constraint::Type::VERTICAL; break;
case SLVS_C_DIAMETER:           t = Constraint::Type::DIAMETER; break;
case SLVS_C_PT_ON_CIRCLE:       t = Constraint::Type::PT_ON_CIRCLE; break;
case SLVS_C_SAME_ORIENTATION:   t = Constraint::Type::SAME_ORIENTATION; break;
case SLVS_C_ANGLE:              t = Constraint::Type::ANGLE; break;
case SLVS_C_PARALLEL:           t = Constraint::Type::PARALLEL; break;
case SLVS_C_PERPENDICULAR:      t = Constraint::Type::PERPENDICULAR; break;
case SLVS_C_ARC_LINE_TANGENT:   t = Constraint::Type::ARC_LINE_TANGENT; break;
case SLVS_C_CUBIC_LINE_TANGENT: t = Constraint::Type::CUBIC_LINE_TANGENT; break;
case SLVS_C_EQUAL_RADIUS:       t = Constraint::Type::EQUAL_RADIUS; break;
case SLVS_C_PROJ_PT_DISTANCE:   t = Constraint::Type::PROJ_PT_DISTANCE; break;
case SLVS_C_WHERE_DRAGGED:      t = Constraint::Type::WHERE_DRAGGED; break;
case SLVS_C_CURVE_CURVE_TANGENT:t = Constraint::Type::CURVE_CURVE_TANGENT; break;

default: dbp("bad constraint type %d", sc->type); return;
        }

        c.type = t;

        c.h.v           = sc->h;
        c.group.v       = sc->group;
        c.workplane.v   = sc->wrkpl;
        c.valA          = sc->valA;
        c.ptA.v         = sc->ptA;
        c.ptB.v         = sc->ptB;
        c.entityA.v     = sc->entityA;
        c.entityB.v     = sc->entityB;
        c.entityC.v     = sc->entityC;
        c.entityD.v     = sc->entityD;
        c.other         = (sc->other) ? true : false;
        c.other2        = (sc->other2) ? true : false;

        c.Generate(&params);
        if(!params.IsEmpty()) {
            for(Param &p : params) {
                p.h = SK.param.AddAndAssignId(&p);
                c.valP = p.h;
                SYS.param.Add(&p);
            }
            params.Clear();
            c.ModifyToSatisfy();
        }

        SK.constraint.Add(&c);
    }

    for(i = 0; i < (int)arraylen(ssys->dragged); i++) {
        if(ssys->dragged[i]) {
            hParam hp = { ssys->dragged[i] };
            SYS.dragged.Add(&hp);
        }
    }

    // Additional Logic to add in Directly Specified Constraint Equations
    {
        
        
        for(i = 0; i < ssys->rels; ++i) {
            ConstraintBase c = {}; // Dummy Constraint Base to access generation methods
            c.h.v            = SK.constraint.MaximumId() + 30*i + 1; // unique constraint id to avoid collisions
            Slvs_Rel *srel = &(ssys->rel[i]);
            if(shg % srel->group == 0) { // HACK - Factorizable Groups
                AddRelation(ssys, srel, &params, &c);
                // Copied from constraint loop.
                if(!params.IsEmpty()) {
                    for(Param &p : params) {
                        //p.h    = SK.param.AddAndAssignId(&p);
                        //SYS.param.Add(&p);
                        //SK.param.Add(&p); // TODO - Should we add slack sparams to the sketch?
                    }
                    params.Clear();
                }
            }
        }
    }

    Group g = {};
    g.h.v = shg;

    List<hConstraint> bad = {};

    // Now we're finally ready to solve!
        bool andFindBad = ssys->calculateFaileds ? true : false;
    SolveResult how = SYS.Solve(&g, NULL, &(ssys->dof), &bad, andFindBad, /*andFindFree=*/false);

    switch(how) {
        case SolveResult::OKAY:
            ssys->result = SLVS_RESULT_OKAY;
            break;

        case SolveResult::DIDNT_CONVERGE:
            ssys->result = SLVS_RESULT_DIDNT_CONVERGE;
            break;

        case SolveResult::REDUNDANT_DIDNT_CONVERGE: 
            ssys->result = SLVS_RESULT_INCONSISTENT;
            break;
        case SolveResult::REDUNDANT_OKAY: 
            ssys->result = SLVS_RESULT_REDUNDANT_OKAY;
            break;

        case SolveResult::TOO_MANY_UNKNOWNS:
            ssys->result = SLVS_RESULT_TOO_MANY_UNKNOWNS;
            break;
    }

    // Write the new parameter values back to our caller.
    for(i = 0; i < ssys->params; i++) {
        Slvs_Param *sp = &(ssys->param[i]);
        hParam hp = { sp->h };
        sp->val = SK.GetParam(hp)->val;
    }

    if(ssys->failed) {
        // Copy over any the list of problematic constraints.
        for(i = 0; i < ssys->faileds && i < bad.n; i++) {
            ssys->failed[i] = bad[i].v;
        }
        ssys->faileds = bad.n;
    }

    bad.Clear();
    SYS.param.Clear();
    SYS.entity.Clear();
    SYS.eq.Clear();
    SYS.dragged.Clear();

    SK.param.Clear();
    SK.entity.Clear();
    SK.constraint.Clear();

    FreeAllTemporary();
}

} /* extern "C" */
