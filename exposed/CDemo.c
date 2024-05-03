/*-----------------------------------------------------------------------------
 * Some sample code for slvs.dll. We draw some geometric entities, provide
 * initial guesses for their positions, and then constrain them. The solver
 * calculates their new positions, in order to satisfy the constraints.
 *
 * Copyright 2008-2013 Jonathan Westhues.
 *---------------------------------------------------------------------------*/
#ifdef WIN32
#   include <windows.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <slvs.h>

static Slvs_System sys;

static void *CheckMalloc(size_t n)
{
    void *r = malloc(n);
    if(!r) {
        printf("out of memory!\n");
        exit(-1);
    }
    return r;
}

void CrashRepro() {
    Slvs_hGroup g;
    g = 5;
    sys.param[sys.params++] = Slvs_MakeParam(1, g, 3.0);
    g = 3;
    sys.expr[sys.exprs++]   = Slvs_MakeExpr_Param(0, 1);
    sys.expr[sys.exprs++]   = Slvs_MakeExpr_Const(1, 2.0);
    sys.expr[sys.exprs++]   = Slvs_MakeExpr_LTE(2, 0, 1);

    sys.constraint[sys.constraints++] =
        Slvs_MakeConstraint(1, g, SLVS_C_EQUATIONS, 200, 0, 0, 0, 0, 0, 2);



    sys.calculateFaileds = 0;
    
    g = 15;
    /* And solve. */
    Slvs_Solve(&sys, g);

    printf("\nFirst Solve Result: ");
    switch(sys.result) {
    case SLVS_RESULT_OKAY: printf("OKAY"); break;
    case SLVS_RESULT_DIDNT_CONVERGE: printf("DIDNT_CONVERGE"); break;
    case SLVS_RESULT_REDUNDANT_OKAY: printf("REDUNDANT_OKAY"); break;
    case SLVS_RESULT_INCONSISTENT: printf("REDUNDANT_DIDNT_CONVERGE"); break;
    case SLVS_RESULT_TOO_MANY_UNKNOWNS: printf("TOO_MANY_UNKNOWNS"); break;
    }

    g = 15;
    /* And solve. */
    Slvs_Solve(&sys, g);

    printf("\nSecond Solve Result: ");
    switch(sys.result) {
    case SLVS_RESULT_OKAY: printf("OKAY"); break;
    case SLVS_RESULT_DIDNT_CONVERGE: printf("DIDNT_CONVERGE"); break;
    case SLVS_RESULT_REDUNDANT_OKAY: printf("REDUNDANT_OKAY"); break;
    case SLVS_RESULT_INCONSISTENT: printf("REDUNDANT_DIDNT_CONVERGE"); break;
    case SLVS_RESULT_TOO_MANY_UNKNOWNS: printf("TOO_MANY_UNKNOWNS"); break;
    }
}

void ExampleIneq() {
    Slvs_hGroup g;
    double qw, qx, qy, qz;

    g = 3;
    /* First, we create our workplane. Its origin corresponds to the origin
     * of our base frame (x y z) = (0 0 0) */
    sys.param[sys.params++]    = Slvs_MakeParam(1, g, 0.0);
    sys.param[sys.params++]    = Slvs_MakeParam(2, g, 0.0);
    sys.param[sys.params++]    = Slvs_MakeParam(3, g, 0.0);
    sys.entity[sys.entities++] = Slvs_MakePoint3d(101, g, 1, 2, 3);
    /* and it is parallel to the xy plane, so it has basis vectors (1 0 0)
     * and (0 1 0). */
    Slvs_MakeQuaternion(1, 0, 0, 0, 1, 0, &qw, &qx, &qy, &qz);
    sys.param[sys.params++]    = Slvs_MakeParam(4, g, qw);
    sys.param[sys.params++]    = Slvs_MakeParam(5, g, qx);
    sys.param[sys.params++]    = Slvs_MakeParam(6, g, qy);
    sys.param[sys.params++]    = Slvs_MakeParam(7, g, qz);
    sys.entity[sys.entities++] = Slvs_MakeNormal3d(102, g, 4, 5, 6, 7);

    sys.entity[sys.entities++] = Slvs_MakeWorkplane(200, g, 101, 102);

    /* Now create a second group. We'll solve group 2, while leaving group 1
     * constant; so the workplane that we've created will be locked down,
     * and the solver can't move it. */
    g = 2;
    /* Three points of a triangle */
    sys.param[sys.params++]    = Slvs_MakeParam(11, g, 1.0);
    sys.param[sys.params++]    = Slvs_MakeParam(12, g, 0.0);
    sys.entity[sys.entities++] = Slvs_MakePoint2d(301, g, 200, 11, 12);

    sys.param[sys.params++]    = Slvs_MakeParam(13, g, 3.0);
    sys.param[sys.params++]    = Slvs_MakeParam(14, g, 0.0);
    sys.entity[sys.entities++] = Slvs_MakePoint2d(302, g, 200, 13, 14);

    sys.param[sys.params++]    = Slvs_MakeParam(15, g, 2.0);
    sys.param[sys.params++]    = Slvs_MakeParam(16, g, 1.0);
    sys.entity[sys.entities++] = Slvs_MakePoint2d(303, g, 200, 15, 16);


    /* Three line segments. */
    sys.entity[sys.entities++] = Slvs_MakeLineSegment(400, g, 200, 301, 302);
    sys.entity[sys.entities++] = Slvs_MakeLineSegment(401, g, 200, 302, 303);
    sys.entity[sys.entities++] = Slvs_MakeLineSegment(402, g, 200, 303, 301);

    /* Equilateral Triangle Constraints */
    
    sys.constraint[sys.constraints++] =
        Slvs_MakeConstraint(1, g, SLVS_C_PT_PT_DISTANCE, 200, 1.0, 301, 302, 0, 0, 0);
    sys.constraint[sys.constraints++] =
        Slvs_MakeConstraint(2, g, SLVS_C_PT_PT_DISTANCE, 200, 1.0, 302, 303, 0, 0, 0);
    sys.constraint[sys.constraints++] =
        Slvs_MakeConstraint(3, g, SLVS_C_PT_PT_DISTANCE, 200, 1.0, 303, 301, 0, 0, 0);
    

    /* Add the Bounding Box Expressions
    * bottom = min(12, 14, 16)
    * top = max(12 ,14, 16)
    * left = min(11, 13, 15)
    * right = max(11, 13, 15)
    */

    sys.expr[sys.exprs++] = Slvs_MakeExpr_Param(0, 11); // x1
    sys.expr[sys.exprs++] = Slvs_MakeExpr_Param(1, 12); // y1

    sys.expr[sys.exprs++] = Slvs_MakeExpr_Param(2, 13); // x2
    sys.expr[sys.exprs++] = Slvs_MakeExpr_Param(3, 14); // y2

    sys.expr[sys.exprs++] = Slvs_MakeExpr_Param(4, 15); // x3
    sys.expr[sys.exprs++] = Slvs_MakeExpr_Param(5, 16); // y3

    sys.expr[sys.exprs++] = Slvs_MakeExpr_Min(6, 0, 2);
    sys.expr[sys.exprs++] = Slvs_MakeExpr_Min(7, 6, 4); // left
    sys.expr[sys.exprs++] = Slvs_MakeExpr_Max(8, 2, 4);
    sys.expr[sys.exprs++] = Slvs_MakeExpr_Max(9, 0, 8); // right
    sys.expr[sys.exprs++] = Slvs_MakeExpr_Min(10, 1, 3);
    sys.expr[sys.exprs++] = Slvs_MakeExpr_Min(11, 5, 10); // bottom
    sys.expr[sys.exprs++] = Slvs_MakeExpr_Max(12, 3, 5);
    sys.expr[sys.exprs++] = Slvs_MakeExpr_Max(13, 1, 12); // top
    
    sys.expr[sys.exprs++] = Slvs_MakeExpr_Const(14, 0.0); // 0.0

    sys.expr[sys.exprs++] = Slvs_MakeExpr_LTE(15, 9, 14); // right <= 0
    sys.expr[sys.exprs++] = Slvs_MakeExpr_LTE(16, 13, 14); // top <= 0

    sys.constraint[sys.constraints++] =
        Slvs_MakeConstraint(4, g, SLVS_C_EQUATIONS, 200, 0, 0, 0, 0, 0, 15);
    
    sys.constraint[sys.constraints++] =
        Slvs_MakeConstraint(5, g, SLVS_C_EQUATIONS, 200, 0, 0, 0, 0, 0, 16);

   
    /* If the solver fails, then ask it to report which constraints caused
     * the problem. */
    sys.calculateFaileds = 0;

    /* And solve. */
    Slvs_Solve(&sys, g);

    
    printf("Solve Result: ");
    switch(sys.result) {
    case SLVS_RESULT_OKAY: printf("OKAY"); break;
    case SLVS_RESULT_DIDNT_CONVERGE: printf("DIDNT_CONVERGE"); break;
    case SLVS_RESULT_REDUNDANT_OKAY: printf("REDUNDANT_OKAY"); break;
    case SLVS_RESULT_INCONSISTENT: printf("REDUNDANT_DIDNT_CONVERGE"); break;
    case SLVS_RESULT_TOO_MANY_UNKNOWNS: printf("TOO_MANY_UNKNOWNS"); break;
    }

    double x1 = sys.param[7].val;
    double y1 = sys.param[8].val;
    double x2 = sys.param[9].val;
    double y2 = sys.param[10].val;
    double x3 = sys.param[11].val;
    double y3 = sys.param[12].val;

    printf("\n\tp1 = (%f , %f)\n", x1, y1);
    printf("\tp2 = (%f , %f)\n", x2, y2);
    printf("\tp3 = (%f , %f)\n", x3, y3);
    //std::cout << "\ttop = " << top->Eval() << "  s_top = " << s_top.v() << std::endl;
    //std::cout << "\tright = " << right->Eval() << "  s_right = " << s_right.v() << std::endl;


    if(sys.result != SLVS_RESULT_OKAY) {
        int i;
        printf("solve failed: problematic constraints are:");
        for(i = 0; i < sys.faileds; i++) {
            printf(" %d", sys.failed[i]);
        }
        printf("\n");
        if(sys.result == SLVS_RESULT_INCONSISTENT) {
            printf("system inconsistent\n");
        } else {
            printf("system nonconvergent\n");
        }
    }
}

int main()
{

    //Test_Ineq();
    
    sys.param      = CheckMalloc(50*sizeof(sys.param[0]));
    sys.entity     = CheckMalloc(50*sizeof(sys.entity[0]));
    sys.constraint = CheckMalloc(50*sizeof(sys.constraint[0]));
    sys.expr       = CheckMalloc(100 * sizeof(sys.expr[0]));

    sys.failed  = CheckMalloc(50*sizeof(sys.failed[0]));
    sys.faileds = 50;
    
    CrashRepro();
    

    /*Example3d();*/
    
    /*
    for(;;) {
        Example2d();
        sys.params = sys.constraints = sys.entities = 0;
        break;
    }
    */
    
    return 0;
}

