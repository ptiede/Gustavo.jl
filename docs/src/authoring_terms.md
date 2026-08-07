```@meta
CurrentModule = Gustavo.Calibration
```

# Authoring a new gain term

A gain [`term`](@ref AbstractGainTerm) is one physical contribution to a
station's phase or log-amplitude response — a delay, a rate, a polynomial
bandpass shape. Gustavo's built-in terms (`ConstantTerm`, `Delay`,
`Dispersion`, `Rate`, `Polynomial`) all implement the same narrow interface,
and a third-party term implements it too: no other seam is required to plug a
new physical effect into the forward map.

A term never sees the global parameter vector, the layout that addresses it,
or an index into either — it is handed its own named parameters and the
coordinates it asked for, and returns a scalar contribution for one
(channel, time) cell. That isolation is what keeps the forward map
allocation-free, type-stable, and AD-generic: adding a term never touches the
evaluation loop.

## The interface

A new term is a `struct` subtyping [`AbstractGainTerm`](@ref) plus five
methods (a sixth, [`term_label`](@ref), is optional):

  - [`term_axes`](@ref) — which coordinate axes the term reads.
  - [`param_shapes`](@ref) — the term's own parameter names and shapes.
  - a coordinate builder ([`freq_coordinate`](@ref) / [`time_coordinate`](@ref))
    and its resolved state ([`freq_coord_state`](@ref) /
    [`time_coord_state`](@ref)) for each axis declared.
  - [`term_eval`](@ref) — the scalar evaluation itself.
  - [`term_label`](@ref) — optional; defaults to the type name.

### `term_axes`

Names the coordinate axes a term reads, drawn from `(:Frequency, :Ti)`, both,
or neither:

```julia
term_axes(::Quadratic) = (:Frequency,)
```

`term_eval` is handed exactly these axes, under these names, as a
`NamedTuple` — a term reading both writes `x.Frequency` and `x.Ti`. The names
are checked at plan time, so a typo in `term_axes` fails there rather than
silently evaluating against the wrong axis.

### `param_shapes`

Names the term's parameters for one (time-segment, frequency-segment) block
and gives each a shape: `()` for a scalar, `(n,)` for a vector of length `n`.
`nchan_seg` — the segment's channel count — is available for terms whose
arity the data sets (`Polynomial`'s degree is fixed by construction and
ignores it):

```julia
param_shapes(::Quadratic, nchan_seg) = (quad = (),)
```

`term_eval` receives these as a `NamedTuple` too, so a term author writes
`p.quad` rather than a position into a block whose length it would otherwise
have to know. How finely a term varies in frequency or time is said by its
segmentation (`ChannelBlocks`, `TimeBlocks`, …), never by `param_shapes` —
a value that is free per channel is `ConstantTerm` paired with
`ChannelBlocks(1)`.

### Coordinate builders

One pair per axis declared, because a term may read both. The builder is a
scalar function of one coordinate and the segment it lies in, so the same
definition serves the solve grid and any grid a solution is later applied to:

```julia
freq_coordinate(::Quadratic, f, f0, seg) = f - f0
```

Its third argument is the term's RESOLVED STATE — the constants its coordinate
reads, whatever they are. They are not parameters (nothing fits them) and not
fields of the term (a term is a declaration, written before any geometry
exists), so a term resolves its own against the solve geometry once at plan
time:

```julia
freq_coord_state(::Quadratic, geom, fseg_id, nfseg) = geom.f0
```

`Delay` and `Dispersion` resolve a reference frequency this way, `Rate` a
reference epoch, and `Polynomial` a per-segment centre and scale (which is why
the builder is given `seg`). A term needing nothing stores whatever it likes,
including `nothing`.

There is deliberately no generic fallback for either: a term that declares an
axis in `term_axes` but defines no matching builder or state errors at plan
time (a `MethodError`), rather than silently evaluating that axis at zero.

### `term_eval`

The term's contribution to phase or log-amplitude at one (channel, time)
cell, `term_eval(term, p, x)`. `p` holds the parameters `param_shapes`
declared — a scalar per `()` name, a vector view per `(n,)` name; `x` holds
the coordinates `term_axes` declared. This is the hot path of
`evaluate_gains`, so it must stay allocation-free and type-stable: read `p`
and `x` by field name, never by iterating the `NamedTuple`.

```julia
@inline term_eval(::Quadratic, p, x) = p.quad * x.Frequency^2
```

### `term_label`

A short diagnostic label, used in `show` and summaries. Optional — the
default is the type name:

```julia
term_label(t::AbstractGainTerm) = string(nameof(typeof(t)))
```

Override it only when the type name is not evocative enough on its own (the
built-in terms do, e.g. `term_label(::Delay) = "delay"`,
`term_label(t::Polynomial{:Frequency}) = "polyf$(t.degree)"`).

## Worked example: a quadratic frequency term

A term whose phase grows as the square of the offset from `f0` — not
physical, but a compact illustration of every hook:

```julia
using Gustavo.Calibration

"Quadratic phase in frequency: phase = `quad`·(f − f0)², `quad` in rad/Hz²."
struct Quadratic <: AbstractGainTerm end

term_axes(::Quadratic) = (:Frequency,)
param_shapes(::Quadratic, nchan_seg) = (quad = (),)
freq_coord_state(::Quadratic, geom, fseg_id, nfseg) = geom.f0
freq_coordinate(::Quadratic, f, f0, seg) = f - f0
@inline term_eval(::Quadratic, p, x) = p.quad * x.Frequency^2
term_label(::Quadratic) = "quad"
```

A term is not fit on its own — it is wrapped in a [`GainComponent`](@ref)
(which pins its time/frequency segmentation) and a [`TiedComponent`](@ref)
(which pins how it ties across feeds), then placed in a
[`StationGainModel`](@ref) alongside the other components:

```julia
model = StationGainModel(
    phase = (
        d = TiedComponent(GainComponent(Delay(), GlobalTime(), GlobalFrequency()), SharedFeeds()),
        q = TiedComponent(GainComponent(Quadratic(), GlobalTime(), GlobalFrequency()), SharedFeeds()),
    ),
)
```

From here `Quadratic` participates in `plan_parameters` and `evaluate_gains`
exactly as the built-in terms do — nothing downstream of the six hooks above
is aware that it is a third-party addition.
