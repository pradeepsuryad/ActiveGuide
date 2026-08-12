# ActiveGuide Phase 3 results

One organ, three models of it. Identical fixtures, margin, damping,
velocity box and integrator; only the controller's model of the
forbidden region changes.

| anatomy model          |       goal | TRUE clear | organ viol |  believed |  RCM max |     p99 |
|--------------------------|--------------|--------------|--------------|-------------|------------|-----------|
| circumscribed sphere   |    0.50 mm |   10.53 mm |    0.00 mm |  -0.22 mm |  0.50 mm |  556 us |
| inscribed sphere       |    0.50 mm |   -0.68 mm |    0.68 mm |   0.00 mm |  0.50 mm |  495 us |
| baked mesh SDF (ours)  |    0.50 mm |    4.40 mm |    0.00 mm |   0.00 mm |  0.50 mm |  635 us |

`TRUE clear` is the smallest exact distance from any sampled point on
the instrument (tip and shaft) to the real mesh surface, measured with
`anatomy.MeshSdf`. Negative means the instrument was inside the organ.

`believed` is clearance past the 4 mm margin, as the controller's own model
reported it. All three sit at essentially zero: each rode its safety margin
and thought it was doing exactly that. (Slightly negative is normal -- a VFI
brakes proportionally to the room left rather than stopping dead at the
boundary, so a small transient into the margin is expected.) None of them
could tell that its own model of the anatomy was wrong; that is the point.

## The sphere trade-off, independently of this trajectory

A single run only shows what happened along one path. Integrating the
disagreement between each sphere and the organ over the whole baked
field gives the path-free version:

| organ volume | inscribed leaves unprotected | circumscribed denies |
|---|---|---|
| 44.6 cm^3 | 25.7 cm^3 (57%) | 55.2 cm^3 |

Inscribed radius 16.6 mm, circumscribed 28.8 mm: no sphere about this centre is within
12.2 mm of the surface everywhere, so shrinking the radius only trades
tissue left unprotected against workspace needlessly denied. The measured
penetration above is whatever that error happened to cost on this path.
