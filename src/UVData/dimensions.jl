using DimensionalData: @dim, TimeDim, Ti
@dim Pol "Polarization correlation product (e.g. RR, LL, RL, LR) — xradio MSv4 `polarization`"
@dim Frequency "Frequency channel (Hz) — xradio MSv4 `frequency`"
@dim Ant "Antenna index in the array's antenna table"
@dim Baseline "Baseline label `\"ant1-ant2\"` (xradio MSv4 `baseline_id`)"
@dim Integration TimeDim "Legacy fused (time × baseline) row index — to be retired in favour of `Ti × Baseline`"
@dim UVW "UVW component label (one of `\"U\"`, `\"V\"`, `\"W\"`)"
