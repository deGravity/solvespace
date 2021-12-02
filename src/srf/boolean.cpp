//-----------------------------------------------------------------------------
// Top-level functions to compute the Boolean union, difference or intersection
// between two shells of rational polynomial surfaces.
//
// Copyright 2008-2013 Jonathan Westhues.
//-----------------------------------------------------------------------------
#include "solvespace.h"

static int I;

void SShell::MakeFromUnionOf(SShell *a, SShell *b) {
    MakeFromBoolean(a, b, SSurface::CombineAs::UNION);
}

void SShell::MakeFromDifferenceOf(SShell *a, SShell *b) {
    MakeFromBoolean(a, b, SSurface::CombineAs::DIFFERENCE);
}

void SShell::MakeFromIntersectionOf(SShell *a, SShell *b) {
    MakeFromBoolean(a, b, SSurface::CombineAs::INTERSECTION);
}

void SCurve::GetAxisAlignedBounding(Vector *ptMax, Vector *ptMin) const {
    *ptMax = {VERY_NEGATIVE, VERY_NEGATIVE, VERY_NEGATIVE};
    *ptMin = {VERY_POSITIVE, VERY_POSITIVE, VERY_POSITIVE};

    for(int i = 0; i <= exact.deg; i++) {
        exact.ctrl[i].MakeMaxMin(ptMax, ptMin);
    }
}

// We will be inserting other curve vertices into our curves to split them.
// This is helpful when curved surfaces become tangent along a trim and the
// usual tests for curve-surface intersection don't split the curve at a vertex.
// This is faster than the previous version that split at surface corners and
// handles more buggy cases. It's not clear this is the best way but it works ok.
static void FindVertsOnCurve(List<SInter> *l, const SCurve *curve, SShell *sh) {

    Vector amax, amin;
    curve->GetAxisAlignedBounding(&amax, &amin);

    for(auto sc : sh->curve) {
        if(!sc.isExact) continue;
        
        Vector cmax, cmin;
        sc.GetAxisAlignedBounding(&cmax, &cmin);

        if(Vector::BoundingBoxesDisjoint(amax, amin, cmax, cmin)) {
            // They cannot possibly intersect, no curves to generate
            continue;
        }
        
        for(int i=0; i<2; i++) {
            Vector pt = sc.exact.ctrl[ i==0 ? 0 : sc.exact.deg ];
            double t;
            curve->exact.ClosestPointTo(pt, &t, /*must converge=*/ false);
            double d = pt.Minus(curve->exact.PointAt(t)).Magnitude();
            if((t>LENGTH_EPS) && (t<(1.0-LENGTH_EPS)) && (d < LENGTH_EPS)) {
                SInter inter;
                inter.p = pt;
                l->Add(&inter);
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Take our original pwl curve. Wherever an edge intersects a surface within
// either agnstA or agnstB, split the piecewise linear element. Then refine
// the intersection so that it lies on all three relevant surfaces: the
// intersecting surface, srfA, and srfB. (So the pwl curve should lie at
// the intersection of srfA and srfB.) Return a new pwl curve with everything
// split.
//-----------------------------------------------------------------------------
SCurve SCurve::MakeCopySplitAgainst(SShell *agnstA, SShell *agnstB,
                                    SSurface *srfA, SSurface *srfB) const
{
    SCurve ret;
    ret = *this;
    ret.pts = {};

    // First find any vertex that lies on our curve.
    List<SInter> vertpts = {};
    if(isExact) {
        if(agnstA)
            FindVertsOnCurve(&vertpts, this, agnstA);
        if(agnstB)
            FindVertsOnCurve(&vertpts, this, agnstB);
    }
    
    const SCurvePt *p = pts.First();
    ssassert(p != NULL, "Cannot split an empty curve");
    SCurvePt prev = *p;
    ret.pts.Add(p);
    p = pts.NextAfter(p);
            
    for(; p; p = pts.NextAfter(p)) {
        List<SInter> il = {};

        // Find all the intersections with the two passed shells
        if(agnstA)
            agnstA->AllPointsIntersecting(prev.p, p->p, &il,
                /*asSegment=*/true, /*trimmed=*/false, /*inclTangent=*/true);
        if(agnstB)
            agnstB->AllPointsIntersecting(prev.p, p->p, &il,
                /*asSegment=*/true, /*trimmed=*/false, /*inclTangent=*/true);

        if(!il.IsEmpty()) {
            // The intersections were generated by intersecting the pwl
            // edge against a surface; so they must be refined to lie
            // exactly on the original curve.
            il.ClearTags();
            SInter *pi;
            for(pi = il.First(); pi; pi = il.NextAfter(pi)) {
                if(pi->srf == srfA || pi->srf == srfB) {
                    // The edge certainly intersects the surfaces that it
                    // trims (at its endpoints), but those ones don't count.
                    // They are culled later, but no sense calculating them
                    // and they will cause numerical problems (since two
                    // of the three surfaces they're refined to lie on will
                    // be identical, so the matrix will be singular).
                    pi->tag = 1;
                    continue;
                }

                Point2d puv;
                (pi->srf)->ClosestPointTo(pi->p, &puv, /*mustConverge=*/false);

                // Split the edge if the intersection lies within the surface's
                // trim curves, or within the chord tol of the trim curve; want
                // some slop if points are close to edge and pwl is too coarse,
                // and it doesn't hurt to split unnecessarily.
                Point2d dummy = { 0, 0 };
                SBspUv::Class c = (pi->srf->bsp) ? pi->srf->bsp->ClassifyPoint(puv, dummy, pi->srf) : SBspUv::Class::OUTSIDE;
                if(c == SBspUv::Class::OUTSIDE) {
                    double d = VERY_POSITIVE;
                    if(pi->srf->bsp) d = pi->srf->bsp->MinimumDistanceToEdge(puv, pi->srf);
                    if(d > SS.ChordTolMm()) {
                        pi->tag = 1;
                        continue;
                    }
                }

                // We're keeping the intersection, so actually refine it. Finding the intersection
                // to within EPS is important to match the ends of different chopped trim curves.
                // The general 3-surface intersection fails to refine for trims where surfaces
                // are tangent at the curve, but those trims are usually exact, so…
                if(isExact) {
                    (pi->srf)->PointOnCurve(&exact, &(puv.x), &(puv.y));
                } else {
                    (pi->srf)->PointOnSurfaces(srfA, srfB, &(puv.x), &(puv.y));
                }
                pi->p = (pi->srf)->PointAt(puv);
            }
            il.RemoveTagged();
        }
        // Now add any vertex that is on this segment
        const Vector lineStart     = prev.p;
        const Vector lineDirection = (p->p).Minus(prev.p);
        for(auto vtx : vertpts) {
            double t = (vtx.p.Minus(lineStart)).DivProjected(lineDirection);
            if((0.0 < t) && (t < 1.0)) {
                il.Add(&vtx);
            }
        }
        if(!il.IsEmpty()) {
            SInter *pi;

            // And now sort them in order along the line. Note that we must
            // do that after refining, in case the refining would make two
            // points switch places.
            std::sort(il.begin(), il.end(), [&](const SInter &a, const SInter &b) {
                double ta = (a.p.Minus(lineStart)).DivProjected(lineDirection);
                double tb = (b.p.Minus(lineStart)).DivProjected(lineDirection);

                return (ta < tb);
            });

            // And now uses the intersections to generate our split pwl edge(s)
            Vector prev = Vector::From(VERY_POSITIVE, 0, 0);
            for(pi = il.First(); pi; pi = il.NextAfter(pi)) {
                // On-edge intersection will generate same split point for
                // both surfaces, so don't create zero-length edge.
                if(!prev.Equals(pi->p)) {
                    SCurvePt scpt;
                    scpt.tag    = 0;
                    scpt.p      = pi->p;
                    scpt.vertex = true;
                    ret.pts.Add(&scpt);
                }
                prev = pi->p;
            }
        }

        il.Clear();
        ret.pts.Add(p);
        prev = *p;
    }
    vertpts.Clear();
    return ret;
}

void SShell::CopyCurvesSplitAgainst(bool opA, SShell *agnst, SShell *into) {
#pragma omp parallel for
    for(int i=0; i<curve.n; i++) {
        SCurve *sc = &curve[i];
        SCurve scn = sc->MakeCopySplitAgainst(agnst, NULL,
                                surface.FindById(sc->surfA),
                                surface.FindById(sc->surfB));
        scn.source = opA ? SCurve::Source::A : SCurve::Source::B;
#pragma omp critical
        {
            hSCurve hsc = into->curve.AddAndAssignId(&scn);
            // And note the new ID so that we can rewrite the trims appropriately
            sc->newH = hsc;
        }
    }
}

void SSurface::TrimFromEdgeList(SEdgeList *el, bool asUv) {
    el->l.ClearTags();

    STrimBy stb = {};
    for(;;) {
        // Find an edge, any edge; we'll start from there.
        SEdge *se;
        for(se = el->l.First(); se; se = el->l.NextAfter(se)) {
            if(se->tag) continue;
            break;
        }
        if(!se) break;
        se->tag = 1;
        stb.start = se->a;
        stb.finish = se->b;
        stb.curve.v = se->auxA;
        stb.backwards = se->auxB ? true : false;

        // Find adjoining edges from the same curve; those should be
        // merged into a single trim.
        bool merged;
        do {
            merged = false;
            for(se = el->l.First(); se; se = el->l.NextAfter(se)) {
                if(se->tag)                         continue;
                if(se->auxA != (int)stb.curve.v)    continue;
                if(( se->auxB && !stb.backwards) ||
                   (!se->auxB &&  stb.backwards))   continue;

                if((se->a).Equals(stb.finish)) {
                    stb.finish = se->b;
                    se->tag = 1;
                    merged = true;
                } else if((se->b).Equals(stb.start)) {
                    stb.start = se->a;
                    se->tag = 1;
                    merged = true;
                }
            }
        } while(merged);

        if(asUv) {
            stb.start  = PointAt(stb.start.x,  stb.start.y);
            stb.finish = PointAt(stb.finish.x, stb.finish.y);
        }

        // And add the merged trim, with xyz (not uv like the polygon) pts
        trim.Add(&stb);
    }
}

static bool KeepRegion(SSurface::CombineAs type, bool opA, SShell::Class shell, SShell::Class orig)
{
    bool inShell = (shell == SShell::Class::INSIDE),
         outSide = (shell == SShell::Class::OUTSIDE),
         inSame  = (shell == SShell::Class::COINC_SAME),
         inOrig  = (orig == SShell::Class::INSIDE);

    if(!inOrig) return false;
    switch(type) {
        case SSurface::CombineAs::UNION:
            if(opA) {
                return outSide;
            } else {
                return outSide || inSame;
            }

        case SSurface::CombineAs::DIFFERENCE:
            if(opA) {
                return outSide;
            } else {
                return inShell || inSame;
            }

        case SSurface::CombineAs::INTERSECTION:
            if(opA) {
                return inShell;
            } else {
                return inShell || inSame;
            }

        default: ssassert(false, "Unexpected combine type");
    }
}
static bool KeepEdge(SSurface::CombineAs type, bool opA,
                     SShell::Class indir_shell, SShell::Class outdir_shell,
                     SShell::Class indir_orig, SShell::Class outdir_orig)
{
    bool keepIn  = KeepRegion(type, opA, indir_shell,  indir_orig),
         keepOut = KeepRegion(type, opA, outdir_shell, outdir_orig);

    // If the regions to the left and right of this edge are both in or both
    // out, then this edge is not useful and should be discarded.
    if(keepIn && !keepOut) return true;
    return false;
}

static void TagByClassifiedEdge(SBspUv::Class bspclass, SShell::Class *indir, SShell::Class *outdir)
{
    switch(bspclass) {
        case SBspUv::Class::INSIDE:
            *indir  = SShell::Class::INSIDE;
            *outdir = SShell::Class::INSIDE;
            break;

        case SBspUv::Class::OUTSIDE:
            *indir  = SShell::Class::OUTSIDE;
            *outdir = SShell::Class::OUTSIDE;
            break;

        case SBspUv::Class::EDGE_PARALLEL:
            *indir  = SShell::Class::INSIDE;
            *outdir = SShell::Class::OUTSIDE;
            break;

        case SBspUv::Class::EDGE_ANTIPARALLEL:
            *indir  = SShell::Class::OUTSIDE;
            *outdir = SShell::Class::INSIDE;
            break;

        default:
            dbp("TagByClassifiedEdge: fail!");
            *indir  = SShell::Class::OUTSIDE;
            *outdir = SShell::Class::OUTSIDE;
            break;
    }
}

static void DEBUGEDGELIST(SEdgeList *sel, SSurface *surf) {
    dbp("print %d edges", sel->l.n);
    SEdge *se;
    for(se = sel->l.First(); se; se = sel->l.NextAfter(se)) {
        Vector mid = (se->a).Plus(se->b).ScaledBy(0.5);
        Vector arrow = (se->b).Minus(se->a);
        swap(arrow.x, arrow.y);
        arrow.x *= -1;
        arrow = arrow.WithMagnitude(0.01);
        arrow = arrow.Plus(mid);

        SS.nakedEdges.AddEdge(surf->PointAt(se->a.x, se->a.y),
                              surf->PointAt(se->b.x, se->b.y));
        SS.nakedEdges.AddEdge(surf->PointAt(mid.x, mid.y),
                              surf->PointAt(arrow.x, arrow.y));
    }
}

//-----------------------------------------------------------------------------
// We are given src, with at least one edge, and avoid, a list of points to
// avoid. We return a chain of edges (that share endpoints), such that no
// point within the avoid list ever occurs in the middle of a chain. And we
// delete the edges in that chain from our source list.
//-----------------------------------------------------------------------------
void SSurface::FindChainAvoiding(SEdgeList *src, SEdgeList *dest,
                                 SPointList *avoid)
{
    ssassert(!src->l.IsEmpty(), "Need at least one edge");
    // Start with an arbitrary edge.
    dest->l.Add(src->l.First());
    src->l.ClearTags();
    src->l.First()->tag = 1;

    bool added;
    do {
        added = false;
        // The start and finish of the current edge chain
        Vector s = dest->l.First()->a,
               f = dest->l.Last()->b;

        // We can attach a new edge at the start or finish, as long as that
        // start or finish point isn't in the list of points to avoid.
        bool startOkay  = !avoid->ContainsPoint(s),
             finishOkay = !avoid->ContainsPoint(f);

        // Now look for an unused edge that joins at the start or finish of
        // our chain (if permitted by the avoid list).
        SEdge *se;
        for(se = src->l.First(); se; se = src->l.NextAfter(se)) {
            if(se->tag) continue;
            if(startOkay && s.Equals(se->b)) {
                dest->l.AddToBeginning(se);
                s = se->a;
                se->tag = 1;
                startOkay = !avoid->ContainsPoint(s);
            } else if(finishOkay && f.Equals(se->a)) {
                dest->l.Add(se);
                f = se->b;
                se->tag = 1;
                finishOkay = !avoid->ContainsPoint(f);
            } else {
                continue;
            }

            added = true;
        }
    } while(added);

    src->l.RemoveTagged();
}

void SSurface::EdgeNormalsWithinSurface(Point2d auv, Point2d buv,
                                        Vector *pt,
                                        Vector *enin, Vector *enout,
                                        Vector *surfn,
                                        uint32_t auxA,
                                        SShell *shell, SShell *sha, SShell *shb)
{
    // the midpoint of the edge
    Point2d muv  = (auv.Plus(buv)).ScaledBy(0.5);

    *pt    = PointAt(muv);

    // If this edge just approximates a curve, then refine our midpoint so
    // so that it actually lies on that curve too. Otherwise stuff like
    // point-on-face tests will fail, since the point won't actually lie
    // on the other face.
    hSCurve hc = { auxA };
    SCurve *sc = shell->curve.FindById(hc);
    if(sc->isExact && sc->exact.deg != 1) {
        double t;
        sc->exact.ClosestPointTo(*pt, &t, /*mustConverge=*/false);
        *pt = sc->exact.PointAt(t);
        ClosestPointTo(*pt, &muv);
    } else if(!sc->isExact) {
        SSurface *trimmedA = sc->GetSurfaceA(sha, shb),
                       *trimmedB = sc->GetSurfaceB(sha, shb);
        *pt = trimmedA->ClosestPointOnThisAndSurface(trimmedB, *pt);
        ClosestPointTo(*pt, &muv);
    }

    *surfn = NormalAt(muv.x, muv.y);

    // Compute the edge's inner normal in xyz space.
    Vector ab    = (PointAt(auv)).Minus(PointAt(buv)),
           enxyz = (ab.Cross(*surfn)).WithMagnitude(SS.ChordTolMm());
    // And based on that, compute the edge's inner normal in uv space. This
    // vector is perpendicular to the edge in xyz, but not necessarily in uv.
    Vector tu, tv, tx, ty;
    TangentsAt(muv.x, muv.y, &tu, &tv);
    Vector n = tu.Cross(tv);
    // since tu and tv may not be orthogonal, use y in place of v, x in place of u.
    // |y| = |v|sin(theta) where theta is the angle between tu and tv.
    ty = n.Cross(tu).ScaledBy(1.0/tu.MagSquared());
    tx = tv.Cross(n).ScaledBy(1.0/tv.MagSquared());

    Point2d enuv;
    enuv.x = enxyz.Dot(tx) / tx.MagSquared();
    enuv.y = enxyz.Dot(ty) / ty.MagSquared();

    // Compute the inner and outer normals of this edge (within the srf),
    // in xyz space. These are not necessarily antiparallel, if the
    // surface is curved.
    Vector pin   = PointAt(muv.Minus(enuv)),
           pout  = PointAt(muv.Plus(enuv));
    *enin  = pin.Minus(*pt),
    *enout = pout.Minus(*pt);
}

//-----------------------------------------------------------------------------
// Trim this surface against the specified shell, in the way that's appropriate
// for the specified Boolean operation type (and which operand we are). We
// also need a pointer to the shell that contains our own surface, since that
// contains our original trim curves.
//-----------------------------------------------------------------------------
SSurface SSurface::MakeCopyTrimAgainst(SShell *parent,
                                       SShell *sha, SShell *shb,
                                       SShell *into,
                                       SSurface::CombineAs type,
                                       int dbg_index)
{
    bool opA = (parent == sha);
    SShell *agnst = opA ? shb : sha;

    SSurface ret;
    // The returned surface is identical, just the trim curves change
    ret = *this;
    ret.trim = {};

    // First, build a list of the existing trim curves; update them to use
    // the split curves.
    STrimBy *stb;
    for(stb = trim.First(); stb; stb = trim.NextAfter(stb)) {
        STrimBy stn = *stb;
        stn.curve = (parent->curve.FindById(stn.curve))->newH;
        ret.trim.Add(&stn);
    }

    if(type == SSurface::CombineAs::DIFFERENCE && !opA) {
        // The second operand of a Boolean difference gets turned inside out
        ret.Reverse();
    }

    // Build up our original trim polygon; remember the coordinates could
    // be changed if we just flipped the surface normal, and we are using
    // the split curves (not the original curves).
    SEdgeList orig = {};
    ret.MakeEdgesInto(into, &orig, MakeAs::UV);
    ret.trim.Clear();
    // which means that we can't necessarily use the old BSP...
    SBspUv *origBsp = SBspUv::From(&orig, &ret);

    // And now intersect the other shell against us
    SEdgeList inter = {};

    SSurface *ss;
    for(SCurve &sc : into->curve) {
        if(sc.source != SCurve::Source::INTERSECTION) continue;
        if(opA) {
            if(sc.surfA != h) continue;
            ss = shb->surface.FindById(sc.surfB);
        } else {
            if(sc.surfB != h) continue;
            ss = sha->surface.FindById(sc.surfA);
        }
        int i;
        for(i = 1; i < sc.pts.n; i++) {
            Vector a = sc.pts[i-1].p,
                   b = sc.pts[i].p;

            Point2d auv, buv;
            ss->ClosestPointTo(a, &(auv.x), &(auv.y));
            ss->ClosestPointTo(b, &(buv.x), &(buv.y));

            SBspUv::Class c = (ss->bsp) ? ss->bsp->ClassifyEdge(auv, buv, ss) : SBspUv::Class::OUTSIDE;
            if(c != SBspUv::Class::OUTSIDE) {
                Vector ta = Vector::From(0, 0, 0);
                Vector tb = Vector::From(0, 0, 0);
                ret.ClosestPointTo(a, &(ta.x), &(ta.y));
                ret.ClosestPointTo(b, &(tb.x), &(tb.y));

                Vector tn = ret.NormalAt(ta.x, ta.y);
                Vector sn = ss->NormalAt(auv.x, auv.y);

                // We are subtracting the portion of our surface that
                // lies in the shell, so the in-plane edge normal should
                // point opposite to the surface normal.
                bool bkwds = true;
                if((tn.Cross(b.Minus(a))).Dot(sn) < 0) bkwds = !bkwds;
                if((type == SSurface::CombineAs::DIFFERENCE && !opA) ||
                   (type == SSurface::CombineAs::INTERSECTION)) { // Invert all newly created edges for intersection
                    bkwds = !bkwds;
                }
                if(bkwds) {
                    inter.AddEdge(tb, ta, sc.h.v, 1);
                } else {
                    inter.AddEdge(ta, tb, sc.h.v, 0);
                }
            }
        }
    }

    // Record all the points where more than two edges join, which I will call
    // the choosing points. If two edges join at a non-choosing point, then
    // they must either both be kept or both be discarded (since that would
    // otherwise create an open contour).
    SPointList choosing = {};
    SEdge *se;
    for(se = orig.l.First(); se; se = orig.l.NextAfter(se)) {
        choosing.IncrementTagFor(se->a);
        choosing.IncrementTagFor(se->b);
    }
    for(se = inter.l.First(); se; se = inter.l.NextAfter(se)) {
        choosing.IncrementTagFor(se->a);
        choosing.IncrementTagFor(se->b);
    }
    SPoint *sp;
    for(sp = choosing.l.First(); sp; sp = choosing.l.NextAfter(sp)) {
        if(sp->tag == 2) {
            sp->tag = 1;
        } else {
            sp->tag = 0;
        }
    }
    choosing.l.RemoveTagged();

    // The list of edges to trim our new surface, a combination of edges from
    // our original and intersecting edge lists.
    SEdgeList final = {};

    while(!orig.l.IsEmpty()) {
        SEdgeList chain = {};
        FindChainAvoiding(&orig, &chain, &choosing);

        // Arbitrarily choose an edge within the chain to classify; they
        // should all be the same, though.
        se = &(chain.l[chain.l.n/2]);

        Point2d auv  = (se->a).ProjectXy(),
                buv  = (se->b).ProjectXy();

        Vector pt, enin, enout, surfn;
        ret.EdgeNormalsWithinSurface(auv, buv, &pt, &enin, &enout, &surfn,
                                        se->auxA, into, sha, shb);

        SShell::Class indir_shell, outdir_shell, indir_orig, outdir_orig;

        indir_orig  = SShell::Class::INSIDE;
        outdir_orig = SShell::Class::OUTSIDE;

        agnst->ClassifyEdge(&indir_shell, &outdir_shell,
                            ret.PointAt(auv), ret.PointAt(buv), pt,
                            enin, enout, surfn);

        if(KeepEdge(type, opA, indir_shell, outdir_shell,
                               indir_orig,  outdir_orig))
        {
            for(se = chain.l.First(); se; se = chain.l.NextAfter(se)) {
                final.AddEdge(se->a, se->b, se->auxA, se->auxB);
            }
        }
        chain.Clear();
    }

    while(!inter.l.IsEmpty()) {
        SEdgeList chain = {};
        FindChainAvoiding(&inter, &chain, &choosing);

        // Any edge in the chain, same as above.
        se = &(chain.l[chain.l.n/2]);

        Point2d auv = (se->a).ProjectXy(),
                buv = (se->b).ProjectXy();

        Vector pt, enin, enout, surfn;
        ret.EdgeNormalsWithinSurface(auv, buv, &pt, &enin, &enout, &surfn,
                                        se->auxA, into, sha, shb);

        SShell::Class indir_shell, outdir_shell, indir_orig, outdir_orig;

        SBspUv::Class c_this = (origBsp) ? origBsp->ClassifyEdge(auv, buv, &ret) : SBspUv::Class::OUTSIDE;
        TagByClassifiedEdge(c_this, &indir_orig, &outdir_orig);

        agnst->ClassifyEdge(&indir_shell, &outdir_shell,
                            ret.PointAt(auv), ret.PointAt(buv), pt,
                            enin, enout, surfn);

        if(KeepEdge(type, opA, indir_shell, outdir_shell,
                               indir_orig,  outdir_orig))
        {
            for(se = chain.l.First(); se; se = chain.l.NextAfter(se)) {
                final.AddEdge(se->a, se->b, se->auxA, se->auxB);
            }
        }
        chain.Clear();
    }

    // Cull extraneous edges; duplicates or anti-parallel pairs. In particular,
    // we can get duplicate edges if our surface intersects the other shell
    // at an edge, so that both surfaces intersect coincident (and both
    // generate an intersection edge).
    final.CullExtraneousEdges(/*both=*/true);

    // Use our reassembled edges to trim the new surface.
    ret.TrimFromEdgeList(&final, /*asUv=*/true);

    SPolygon poly = {};
    final.l.ClearTags();
    if(!final.AssemblePolygon(&poly, NULL, /*keepDir=*/true))
#pragma omp critical
    {
        into->booleanFailed = true;
        dbp("failed: I=%d, avoid=%d", I+dbg_index, choosing.l.n);
        DEBUGEDGELIST(&final, &ret);
    }
    poly.Clear();

    choosing.Clear();
    final.Clear();
    inter.Clear();
    orig.Clear();
    return ret;
}

void SShell::CopySurfacesTrimAgainst(SShell *sha, SShell *shb, SShell *into, SSurface::CombineAs type) {
    std::vector <SSurface> ssn(surface.n);
#pragma omp parallel for
    for (int i = 0; i < surface.n; i++)
    {
        SSurface *ss = &surface[i];
        ssn[i] = ss->MakeCopyTrimAgainst(this, sha, shb, into, type, i);
    }

    for (int i = 0; i < surface.n; i++)
    {
        surface[i].newH = into->surface.AddAndAssignId(&ssn[i]);
    }
    I += surface.n;
}

void SShell::MakeIntersectionCurvesAgainst(SShell *agnst, SShell *into) {
#pragma omp parallel for
    for(int i = 0; i< surface.n; i++) {
        SSurface *sa = &surface[i];

        for(SSurface &sb : agnst->surface){
            // Intersect every surface from our shell against every surface
            // from agnst; this will add zero or more curves to the curve
            // list for into.
            sa->IntersectAgainst(&sb, this, agnst, into);
        }
    }
}

void SShell::CleanupAfterBoolean() {
    for(SSurface &ss : surface) {
        ss.edges.Clear();
    }
}

//-----------------------------------------------------------------------------
// All curves contain handles to the two surfaces that they trim. After a
// Boolean or assembly, we must rewrite those handles to refer to the curves
// by their new IDs.
//-----------------------------------------------------------------------------
void SShell::RewriteSurfaceHandlesForCurves(SShell *a, SShell *b) {
    for(SCurve &sc : curve) {
        sc.surfA = sc.GetSurfaceA(a, b)->newH,
        sc.surfB = sc.GetSurfaceB(a, b)->newH;
    }
}

//-----------------------------------------------------------------------------
// Copy all the surfaces and curves from two different shells into a single
// shell. The only difficulty is to rewrite all of their handles; we don't
// look for any surface intersections, so if two objects interfere then the
// result is just self-intersecting. This is used for assembly, since it's
// much faster than merging as union.
//-----------------------------------------------------------------------------
void SShell::MakeFromAssemblyOf(SShell *a, SShell *b) {
    booleanFailed = false;

    Vector t = Vector::From(0, 0, 0);
    Quaternion q = Quaternion::IDENTITY;
    int i = 0;
    SShell *ab;

    // First, copy over all the curves. Note which shell (a or b) each curve
    // came from, but assign it a new ID.
    curve.ReserveMore(a->curve.n + b->curve.n);
    SCurve cn;
    for(i = 0; i < 2; i++) {
        ab = (i == 0) ? a : b;
        for(SCurve &c : ab->curve) {
            cn = SCurve::FromTransformationOf(&c, t, q, 1.0);
            cn.source = (i == 0) ? SCurve::Source::A : SCurve::Source::B;
            // surfA and surfB are wrong now, and we can't fix them until
            // we've assigned IDs to the surfaces. So we'll get that later.
            c.newH = curve.AddAndAssignId(&cn);
        }
    }

    // Likewise copy over all the surfaces.
    surface.ReserveMore(a->surface.n + b->surface.n);
    SSurface sn;
    for(i = 0; i < 2; i++) {
        ab = (i == 0) ? a : b;
        for(SSurface &s : ab->surface) {
            sn = SSurface::FromTransformationOf(&s, t, q, 1.0, /*includingTrims=*/true);
            // All the trim curve IDs get rewritten; we know the new handles
            // to the curves since we recorded them in the previous step.
            STrimBy *stb;
            for(stb = sn.trim.First(); stb; stb = sn.trim.NextAfter(stb)) {
                stb->curve = ab->curve.FindById(stb->curve)->newH;
            }
            s.newH = surface.AddAndAssignId(&sn);
        }
    }

    // Finally, rewrite the surfaces associated with each curve to use the
    // new handles.
    RewriteSurfaceHandlesForCurves(a, b);
}

void SShell::MakeFromBoolean(SShell *a, SShell *b, SSurface::CombineAs type) {
    booleanFailed = false;

    a->MakeClassifyingBsps(NULL);
    b->MakeClassifyingBsps(NULL);

    // Copy over all the original curves, splitting them so that a
    // piecewise linear segment never crosses a surface from the other
    // shell.
    a->CopyCurvesSplitAgainst(/*opA=*/true,  b, this);
    b->CopyCurvesSplitAgainst(/*opA=*/false, a, this);

    // Generate the intersection curves for each surface in A against all
    // the surfaces in B (which is all of the intersection curves).
    a->MakeIntersectionCurvesAgainst(b, this);

    for(SCurve &sc : curve) {
        SSurface *srfA = sc.GetSurfaceA(a, b),
                 *srfB = sc.GetSurfaceB(a, b);

        sc.RemoveShortSegments(srfA, srfB);
    }

    // And clean up the piecewise linear things we made as a calculation aid
    a->CleanupAfterBoolean();
    b->CleanupAfterBoolean();
    // Remake the classifying BSPs with the split (and short-segment-removed)
    // curves
    a->MakeClassifyingBsps(this);
    b->MakeClassifyingBsps(this);

    if(b->surface.IsEmpty() || a->surface.IsEmpty()) {
        I = 1000000;
    } else {
        I = 0;
    }
    // Then trim and copy the surfaces
    a->CopySurfacesTrimAgainst(a, b, this, type);
    b->CopySurfacesTrimAgainst(a, b, this, type);

    // Now that we've copied the surfaces, we know their new hSurfaces, so
    // rewrite the curves to refer to the surfaces by their handles in the
    // result.
    RewriteSurfaceHandlesForCurves(a, b);

    // And clean up the piecewise linear things we made as a calculation aid
    a->CleanupAfterBoolean();
    b->CleanupAfterBoolean();
}

//-----------------------------------------------------------------------------
// All of the BSP routines that we use to perform and accelerate polygon ops.
//-----------------------------------------------------------------------------
void SShell::MakeClassifyingBsps(SShell *useCurvesFrom) {
#pragma omp parallel for
    for(int i = 0; i<surface.n; i++) {
        surface[i].MakeClassifyingBsp(this, useCurvesFrom);
    }
}

void SSurface::MakeClassifyingBsp(SShell *shell, SShell *useCurvesFrom) {
    SEdgeList el = {};

    MakeEdgesInto(shell, &el, MakeAs::UV, useCurvesFrom);
    bsp = SBspUv::From(&el, this);
    el.Clear();

    edges = {};
    MakeEdgesInto(shell, &edges, MakeAs::XYZ, useCurvesFrom);
}

SBspUv *SBspUv::Alloc() {
    return (SBspUv *)AllocTemporary(sizeof(SBspUv));
}

SBspUv *SBspUv::From(SEdgeList *el, SSurface *srf) {
    SEdgeList work = {};

    SEdge *se;
    for(se = el->l.First(); se; se = el->l.NextAfter(se)) {
        work.AddEdge(se->a, se->b, se->auxA, se->auxB);
    }
    std::sort(work.l.begin(), work.l.end(), [](SEdge const &a, SEdge const &b) {
        double la = (a.a).Minus(a.b).Magnitude(), lb = (b.a).Minus(b.b).Magnitude();
        // Sort in descending order, longest first. This improves numerical
        // stability for the normals.
        return la > lb;
    });
    SBspUv *bsp = NULL;
    for(se = work.l.First(); se; se = work.l.NextAfter(se)) {
        bsp = InsertOrCreateEdge(bsp, (se->a).ProjectXy(), (se->b).ProjectXy(), srf);
    }

    work.Clear();
    return bsp;
}

//-----------------------------------------------------------------------------
// The points in this BSP are in uv space, but we want to apply our tolerances
// consistently in xyz (i.e., we want to say a point is on-edge if its xyz
// distance to that edge is less than LENGTH_EPS, irrespective of its distance
// in uv). So we linearize the surface about the point we're considering and
// then do the test. That preserves point-on-line relationships, and the only
// time we care about exact correctness is when we're very close to the line,
// which is when the linearization is accurate.
//-----------------------------------------------------------------------------

void SBspUv::ScalePoints(Point2d *pt, Point2d *a, Point2d *b, SSurface *srf) const {
    Vector tu, tv;
    srf->TangentsAt(pt->x, pt->y, &tu, &tv);
    double mu = tu.Magnitude(), mv = tv.Magnitude();

    pt->x *= mu; pt->y *= mv;
    a ->x *= mu; a ->y *= mv;
    b ->x *= mu; b ->y *= mv;
}

double SBspUv::ScaledSignedDistanceToLine(Point2d pt, Point2d a, Point2d b,
                                          SSurface *srf) const
{
    ScalePoints(&pt, &a, &b, srf);

    Point2d n = ((b.Minus(a)).Normal()).WithMagnitude(1);
    double d = a.Dot(n);

    return pt.Dot(n) - d;
}

double SBspUv::ScaledDistanceToLine(Point2d pt, Point2d a, Point2d b, bool asSegment,
                                    SSurface *srf) const
{
    ScalePoints(&pt, &a, &b, srf);

    return pt.DistanceToLine(a, b, asSegment);
}

SBspUv *SBspUv::InsertOrCreateEdge(SBspUv *where, Point2d ea, Point2d eb, SSurface *srf) {
    if(where == NULL) {
        SBspUv *ret = Alloc();
        ret->a = ea;
        ret->b = eb;
        return ret;
    }
    where->InsertEdge(ea, eb, srf);
    return where;
}

void SBspUv::InsertEdge(Point2d ea, Point2d eb, SSurface *srf) {
    double dea = ScaledSignedDistanceToLine(ea, a, b, srf),
           deb = ScaledSignedDistanceToLine(eb, a, b, srf);

    if(fabs(dea) < LENGTH_EPS && fabs(deb) < LENGTH_EPS) {
        // Line segment is coincident with this one, store in same node
        SBspUv *m = Alloc();
        m->a = ea;
        m->b = eb;
        m->more = more;
        more = m;
    } else if(fabs(dea) < LENGTH_EPS) {
        // Point A lies on this line, but point B does not
        if(deb > 0) {
            pos = InsertOrCreateEdge(pos, ea, eb, srf);
        } else {
            neg = InsertOrCreateEdge(neg, ea, eb, srf);
        }
    } else if(fabs(deb) < LENGTH_EPS) {
        // Point B lies on this line, but point A does not
        if(dea > 0) {
            pos = InsertOrCreateEdge(pos, ea, eb, srf);
        } else {
            neg = InsertOrCreateEdge(neg, ea, eb, srf);
        }
    } else if(dea > 0 && deb > 0) {
        pos = InsertOrCreateEdge(pos, ea, eb, srf);
    } else if(dea < 0 && deb < 0) {
        neg = InsertOrCreateEdge(neg, ea, eb, srf);
    } else {
        // New edge crosses this one; we need to split.
        Point2d n = ((b.Minus(a)).Normal()).WithMagnitude(1);
        double d = a.Dot(n);
        double t = (d - n.Dot(ea)) / (n.Dot(eb.Minus(ea)));
        Point2d pi = ea.Plus((eb.Minus(ea)).ScaledBy(t));
        if(dea > 0) {
            pos = InsertOrCreateEdge(pos, ea, pi, srf);
            neg = InsertOrCreateEdge(neg, pi, eb, srf);
        } else {
            neg = InsertOrCreateEdge(neg, ea, pi, srf);
            pos = InsertOrCreateEdge(pos, pi, eb, srf);
        }
    }
    return;
}

SBspUv::Class SBspUv::ClassifyPoint(Point2d p, Point2d eb, SSurface *srf) const {
    double dp = ScaledSignedDistanceToLine(p, a, b, srf);

    if(fabs(dp) < LENGTH_EPS) {
        const SBspUv *f = this;
        while(f) {
            Point2d ba = (f->b).Minus(f->a);
            if(ScaledDistanceToLine(p, f->a, ba, /*asSegment=*/true, srf) < LENGTH_EPS) {
                if(ScaledDistanceToLine(eb, f->a, ba, /*asSegment=*/false, srf) < LENGTH_EPS){
                    if(ba.Dot(eb.Minus(p)) > 0) {
                        return Class::EDGE_PARALLEL;
                    } else {
                        return Class::EDGE_ANTIPARALLEL;
                    }
                } else {
                    return Class::EDGE_OTHER;
                }
            }
            f = f->more;
        }
        // Pick arbitrarily which side to send it down, doesn't matter
        Class c1 =  neg ? neg->ClassifyPoint(p, eb, srf) : Class::OUTSIDE;
        Class c2 =  pos ? pos->ClassifyPoint(p, eb, srf) : Class::INSIDE;
        if(c1 != c2) {
            dbp("MISMATCH: %d %d %08x %08x", c1, c2, neg, pos);
        }
        return c1;
    } else if(dp > 0) {
        return pos ? pos->ClassifyPoint(p, eb, srf) : Class::INSIDE;
    } else {
        return neg ? neg->ClassifyPoint(p, eb, srf) : Class::OUTSIDE;
    }
}

SBspUv::Class SBspUv::ClassifyEdge(Point2d ea, Point2d eb, SSurface *srf) const {
    SBspUv::Class ret = ClassifyPoint((ea.Plus(eb)).ScaledBy(0.5), eb, srf);
    if(ret == Class::EDGE_OTHER) {
        // Perhaps the edge is tangent at its midpoint (and we screwed up
        // somewhere earlier and failed to split it); try a different
        // point on the edge.
        ret = ClassifyPoint(ea.Plus((eb.Minus(ea)).ScaledBy(0.294)), eb, srf);
    }
    return ret;
}

double SBspUv::MinimumDistanceToEdge(Point2d p, SSurface *srf) const {

    double dn = (neg) ? neg->MinimumDistanceToEdge(p, srf) : VERY_POSITIVE;
    double dp = (pos) ? pos->MinimumDistanceToEdge(p, srf) : VERY_POSITIVE;

    Point2d as = a, bs = b;
    ScalePoints(&p, &as, &bs, srf);
    double d = p.DistanceToLine(as, bs.Minus(as), /*asSegment=*/true);

    return min(d, min(dn, dp));
}

