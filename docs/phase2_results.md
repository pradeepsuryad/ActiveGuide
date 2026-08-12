# ActiveGuide Phase 2 results

Identical task, fixtures, damping, velocity box and integrator.
The only difference is where enforcement happens: Cartesian
output space vs joint-velocity space.

| robot  | controller           |      goal |      wall |  joint lim |   RCM max |  min sig |      p50 |      p99 |
|----------|------------------------|-------------|-------------|--------------|-------------|------------|------------|------------|
| panda  | Cartesian proj.      |   0.50 mm |  0.000 mm |  0.0000 rad |  27.35 mm |   0.3035 |    28 us |   109 us |
| panda  | joint-space QP       |   0.50 mm |  0.000 mm |  0.0000 rad |   0.50 mm |   0.2988 |   165 us |   515 us |
| psm    | Cartesian proj.      |   0.50 mm |  0.000 mm |  0.0000 rad |   0.01 mm |   0.3527 |    79 us |   261 us |
| psm    | joint-space QP       |   0.50 mm |  0.000 mm |  0.0000 rad |   0.01 mm |   0.3527 |   209 us |   532 us |

`wall` and `joint lim` are worst *violations* (0 is perfect).
`min sig` is the smallest singular value of J reached -- lower
means the controller drove the arm nearer a singularity.
