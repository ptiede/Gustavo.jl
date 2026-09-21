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

Conversion happens at the format boundary, not in the solver:

| Operation | Conjugated |
|:----------|:-----------|
| `load_fitsidi`, `write_fitsidi` | yes |
| `load_uvfits` | no |
| `write_uvfits(...; convention = :aips)` | no |
| `write_uvfits(...; convention = :fitsidi)` | yes |

`load_uvfits` assumes a standard AIPS file, so only `convention = :aips`
round-trips through Gustavo as the identity.

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
| `:weights` | inverse variance, ``1/\sigma^2`` |
| `:uvw` | as the source file stores it, seconds of light travel for FITS-IDI and AIPS UVFITS |

Times are absolute. A `Float64` resolves about 0.24 µs at present epochs, and
every delay and rate term is evaluated on ``t - t_0`` about a segment-local
origin, where the resolution is picoseconds.

## Flags and weights

`:flags` is a `Bool` layer on the axes of `:vis`, and is `true` where the datum
must not be used. `:weights` states how good the datum would have been. The two
are independent, matching MSv4's `FLAG`/`WEIGHT` pair: a flagged sample may
carry a positive weight, and a zero weight does not by itself flag a sample.
Solvers require both conditions, so a sample contributes when its flag is unset
and its weight is finite and positive.

UVFITS has no flag table, and a negative weight is the only means the format
has of recording a flag. `write_uvfits` therefore negates the weight of a
flagged sample, preserving its magnitude, and `load_uvfits` reads the sign back
as the flag.

## Feeds and correlation products

Correlation products are labelled by two generic feed letters, `"P"` for feed 1
and `"Q"` for feed 2, so that one vocabulary covers circular and linear feeds.
`"PQ"` relates ``V[a, b, p]`` to antenna ``a``'s feed 1 and antenna ``b``'s
feed 2; [`correlation_feed_pair`](@ref Gustavo.UVData.correlation_feed_pair)
maps a label to that index pair. `"PP"` and `"QQ"` are the parallel hands and
`"PQ"`, `"QP"` the cross hands.

The nominal basis each station's feeds are in is a property of the antenna
table, not of the label, and the array need not share one basis.

## Baseline coordinates

FITS-IDI and AIPS UVFITS share one baseline-coordinate convention and one
antenna ordering, so ``(u, v, w)`` is read and written verbatim by both
readers and both writers, and is never negated. The conjugation applied to the
visibilities on the FITS-IDI boundary does not extend to it.
