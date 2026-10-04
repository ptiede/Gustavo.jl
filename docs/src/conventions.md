```@meta
CurrentModule = Gustavo
```

# [Conventions](@id conventions)

This page fixes the conventions the rest of the package assumes: the sense of
the visibility phase, the sign of the parameters fitted against it, the units
of each stored quantity, and the labels used for feeds and correlation
products. They are stated once here and not repeated elsewhere.

## Visibility phase

The internal convention is the MSv4 (casacore) sense, the conjugate of the
FITS-IDI convention ``V = \langle E_{a_1} \overline{E_{a_2}} \rangle`` of AIPS
Memo 114r §2.1. It is also the convention of CASA, of the Python `xradio`
writer, and of AIPS random-groups UVFITS.

Conversion happens at the format boundary, not in the solver.
[`load_uvfits`](@ref) assumes a standard AIPS file and reads its visibilities
verbatim; a FITS-IDI file is converted, conjugation included, by
`XRadio.fitsidi2msv4`.

## Fitted parameters

A delay ``\tau``, rate ``\dot r`` and phase ``\varphi`` are defined by the
model they are fitted with,

```math
V \propto \exp\!\big(i[\varphi + 2\pi\tau(\nu - \nu_\mathrm{ref}) + 2\pi\dot r (t - t_\mathrm{ref})]\big),
```

against visibilities in the sense above. CASA's `KJones` defines its delay by
the same relation, so a delay fitted here agrees numerically with one CASA fits
for the same observation. fourfit works in the FITS-IDI sense, so a delay
fitted here is the negative of one fourfit fits. Phases and differential TEC
follow the same convention.

Each parameter is referred to an explicit origin: ``\nu_\mathrm{ref}`` is the
reference frequency of the observation, and ``t_\mathrm{ref}`` is the origin of
the time segment the parameter is solved on, which for a per-scan term is that
scan's own mean epoch. A phase is meaningful only with the epoch it was
measured at, since quoting it a lever arm away from that epoch costs
``2\pi\sigma_{\dot r}\Delta t``.

Gains compose multiplicatively, and a station with no solution carries
``\theta = 0``, which is the identity gain rather than a flag. Such stations
are reported as uncovered by the solve and are to be flagged on that basis.

## Units

| Quantity | Unit |
|:---------|:-----|
| Time axis (`Ti` lookup, `obs_time`) | seconds since the Unix epoch, as MSv4's `time` in `unix` format |
| Frequency axis (`Frequency` lookup, `channel_freqs`) | Hz |
| Delay | seconds |
| Fringe rate | Hz |
| Phase, differential TEC coefficient | radians |
| `:weight` | inverse variance, ``1/\sigma^2`` |
| `:uvw` | meters, as MSv4's `UVW`; FITS-IDI and AIPS UVFITS store seconds of light travel, converted on reading and by `write_uvfits` on writing |

Times are absolute. A `Float64` resolves about 0.24 µs at present epochs, and
every delay and rate term is evaluated on ``t - t_0`` about a segment-local
origin, where the resolution is picoseconds.

## Flags and weights

`:flag` is a `Bool` layer on the axes of `:visibility`, and is `true` where the
datum must not be used. `:weight` states how good the datum would have been. The two
are independent, matching MSv4's `FLAG`/`WEIGHT` pair: a flagged sample may
carry a positive weight, and a zero weight does not by itself flag a sample.
Solvers require both conditions, so a sample contributes when its flag is unset
and its weight is finite and positive.

UVFITS has no flag table, and a negative weight is the only means the format
has of recording a flag. [`load_uvfits`](@ref) therefore flags a sample whose
weight has its sign bit set (`-0.0` included) and keeps its magnitude as the
weight; `+0.0` is read as an unflagged sample of zero weight, and a (time,
baseline) slot no record fills is flagged with zero weight.
[`write_uvfits`](@ref) is the inverse: it negates a flagged sample's weight,
and leaves out a record whose every sample is flagged with zero weight.

## Feeds and correlation products

Inside the solver a correlation product is the pair of feed indices it
relates: product ``(f_a, f_b)`` relates ``V[a, b, p]`` to antenna ``a``'s feed
``f_a`` and antenna ``b``'s feed ``f_b``. A solver cube's `Polarization` axis
holds these pairs, [`feed_pairs`](@ref Gustavo.UVData.feed_pairs) returns them,
and products are selected by index or by pair (`pol = (1, 1)`). No label is
read: a feed's nominal polarization is recorded per antenna
(`polarization_type` in a Measurement Set) and matters only where a feed is
mapped back to a physical receptor.

A Measurement Set stores products as receptor labels (`"RR"`, `"XY"`, …).
[`feed_pairs`](@ref feed_pairs(::XRadio.MeasurementSet)) resolves each letter
through its antenna's `polarization_type`, so one stored product can relate
different feed pairs on different baselines; each baseline's products are
reordered into one shared order when data becomes solver input.

## Baseline coordinates

FITS-IDI and AIPS UVFITS share one baseline-coordinate convention and one
antenna ordering, so ``(u, v, w)`` is never negated; it is only converted between
light-seconds and meters. The
conjugation applied to the visibilities on the FITS-IDI boundary does not
extend to it.
