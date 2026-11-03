# ── Bandpass diagnostics ─────────────────────────────────────────────────────
#
# Goodness of fit of corrected visibilities against a channel-constant source,
# and per-station tests for a bandpass that changes between two scans.

# ── Normalized residuals ─────────────────────────────────────────────────────

"""
    normalized_residuals(spectra; dims = Frequency) -> DimStack

Every cut of `spectra` along `dims` divided by its weighted mean along `dims`.
`spectra` holds layers `vis` and `weight`, with the weight the inverse
variance of each real component of `vis`: a [`baseline_spectra`](@ref) result,
or a Measurement Set's visibilities with the weights of flagged samples set to
zero. Layers:

- `r = v / m`, with `m = Σ w v / Σ w` over the cut. Along `Frequency` this
  removes each spectrum's phase, amplitude scale and band-mean source
  visibility, leaving whatever structure across the band the correction did not
  remove, a residual delay included. Along `Ti` it shows each channel's change
  about its mean over time.
- `var = 1 / (w |m|²)`, the variance of each real component of `r`, ignoring
  the noise in `m`.

A sample with zero weight or non-finite data is `NaN` in both layers, and so
is every sample of a cut with no usable sample.
"""
function normalized_residuals(spectra::AbstractDimStack; dims = Frequency)
    vis, weight = spectra.vis, spectra.weight
    R = float(promote_type(real(eltype(vis)), eltype(weight)))
    r = fill!(similar(vis, Complex{R}), Complex{R}(NaN, NaN))
    var = fill!(similar(weight, R), R(NaN))
    for cell in DimensionalData.DimIndices(DimensionalData.otherdims(vis, dims))
        v, w = view(vis, cell...), view(weight, cell...)
        ok = [k for k in eachindex(v, w) if _shape_usable(v[k], w[k])]
        isempty(ok) && continue
        m = sum(k -> w[k] * v[k], ok) / sum(k -> w[k], ok)
        rc, varc = view(r, cell...), view(var, cell...)
        for k in ok
            rc[k] = v[k] / m
            varc[k] = inv(w[k] * abs2(m))
        end
    end
    return DimStack((; r, var))
end

# ── Scan time spans ──────────────────────────────────────────────────────────

"""
    scan_intervals(ps) -> Dict{String, Tuple{Float64, Float64}}

The first and last sample time of each scan of `ps`, in seconds as
`XRadio.times` gives them, keyed by scan name. Read from `DataGeometry(ps)`.
"""
function scan_intervals(ps)
    geom = DataGeometry(ps)
    out = Dict{String, Tuple{Float64, Float64}}()
    for (t, s) in zip(geom.times, geom.scan_of_time)
        lo, hi = get(out, geom.scan_names[s], (t, t))
        out[geom.scan_names[s]] = (min(lo, t), max(hi, t))
    end
    return out
end

# ── Bandpass step tests ──────────────────────────────────────────────────────

"""
    step_tests(residuals, intervals, station) -> Vector{NamedTuple}

An F test for a change in `station`'s bandpass at each gap between two of its
consecutive scans. `residuals` is a [`normalized_residuals`](@ref) result
along `Frequency` of scan averages, and `intervals` a [`scan_intervals`](@ref)
result of the same data.

Every series of residual spectra `r` on a baseline to `station` and a parallel
feed pair (both feeds the same index) is split at the gap and fitted by one
weighted mean spectrum per side, channel by channel, using the channels with at
least two scans on each side. Summed over the series,

    F = (Δχ² / ν₁) / (χ²_split / ν₂),

with `Δχ²` the χ² that two means remove relative to one, `ν₁ = 2` per channel,
`χ²_split` the χ² about the two means and `ν₂ = 2(n - 2)` per channel of `n`
scans. Scatter between scans that is not a step (residual delays, phase noise)
raises `χ²_split` along with `Δχ²`, so `F` stays near 1 without a step. A
residual delay that differs from scan to scan is such scatter, and one that
changes across the gap is a step. A step
at one gap also raises `F` at nearby gaps, where the one-sided means mix the
two bandpasses; it is largest at the step.

One row per gap, in time order, with fields `time` (mid-gap, s), `gap` (its
length, s), `F`, `chi2_split`, `dof_split` (`ν₂`) and `nbaselines`, the number
of (baseline, parallel feed pair) series to `station` holding data. The scans
are those where any such series has data; gaps across which those scans overlap
and gaps with no channel to split are left out.
"""
function step_tests(residuals::AbstractDimStack, intervals::AbstractDict, station)
    r, var = residuals.r, residuals.var
    scan_names, antenna_pairs, feeds = lookup(r, Scan), lookup(r, AntennaPair), lookup(r, FeedPair)
    cells = [
        (j, q) for j in axes(r, AntennaPair), q in axes(r, FeedPair)
            if station in antenna_pairs[j] && feeds[q][1] == feeds[q][2]
    ]
    series = map(cells) do (j, q)
        Rs = parent(permutedims(r[AntennaPair(j), FeedPair(q)], (Frequency, Scan)))
        Vs = parent(permutedims(var[AntennaPair(j), FeedPair(q)], (Frequency, Scan)))
        (Rs, Vs, [s for s in axes(Vs, 2) if any(isfinite, view(Vs, :, s))])
    end
    filter!(x -> !isempty(x[3]), series)
    spans = sort!(unique!([intervals[scan_names[s]] for (_, _, scans) in series for s in scans]))
    starts = Dict(s => first(intervals[scan_names[s]]) for (_, _, scans) in series for s in scans)
    Row = @NamedTuple{time::Float64, gap::Float64, F::Float64, chi2_split::Float64, dof_split::Int, nbaselines::Int}
    rows = Row[]
    last_end = -Inf
    for j in firstindex(spans):(lastindex(spans) - 1)
        last_end = max(last_end, last(spans[j]))
        next_start = first(spans[j + 1])
        next_start > last_end || continue
        T = (last_end + next_start) / 2
        Δ, ν₁, χ², ν₂ = 0.0, 0, 0.0, 0
        for (Rs, Vs, scans) in series
            d = _split_terms(Rs, Vs, scans, s -> starts[s] < T)
            Δ += d[1]; ν₁ += d[2]; χ² += d[3]; ν₂ += d[4]
        end
        (ν₁ > 0 && ν₂ > 0) || continue
        push!(rows, Row((T, next_start - last_end, (Δ / ν₁) / (χ² / ν₂), χ², ν₂, length(series))))
    end
    return rows
end

# Δχ², ν₁, χ²_split and ν₂ of one series `R` (channel × scan) with per-component
# variances `V`, over the scans `scans`, split by `before`.
function _split_terms(R, V, scans, before)
    Δ, ν₁, χ², ν₂ = 0.0, 0, 0.0, 0
    for c in axes(R, 1)
        Wb = Wa = 0.0
        Sb = Sa = zero(ComplexF64)
        nb = na = 0
        for s in scans
            isfinite(V[c, s]) || continue
            w = inv(V[c, s])
            if before(s)
                Wb += w; Sb += w * R[c, s]; nb += 1
            else
                Wa += w; Sa += w * R[c, s]; na += 1
            end
        end
        (nb >= 2 && na >= 2) || continue
        μb, μa = Sb / Wb, Sa / Wa
        Δ += abs2(μb - μa) / (inv(Wb) + inv(Wa))
        for s in scans
            isfinite(V[c, s]) || continue
            χ² += abs2(R[c, s] - (before(s) ? μb : μa)) / V[c, s]
        end
        ν₁ += 2
        ν₂ += 2 * (nb + na - 2)
    end
    return Δ, ν₁, χ², ν₂
end

# ── χ² of corrected visibilities ─────────────────────────────────────────────

"""
    Chi2Tally{T}

Residual χ² and degrees of freedom about a channel-constant (`chi2_const`,
`dof_const`) and a linear-in-frequency (`chi2_lin`, `dof_lin`) complex model
of each spectrum, summed over spectra.
"""
struct Chi2Tally{T <: Real}
    chi2_const::T
    dof_const::Int
    chi2_lin::T
    dof_lin::Int
end

Base.:+(a::Chi2Tally, b::Chi2Tally) = Chi2Tally(
    a.chi2_const + b.chi2_const, a.dof_const + b.dof_const, a.chi2_lin + b.chi2_lin, a.dof_lin + b.dof_lin,
)
Base.zero(::Type{Chi2Tally{T}}) where {T} = Chi2Tally(zero(T), 0, zero(T), 0)
Base.zero(t::Chi2Tally) = zero(typeof(t))

"""
    Chi2Level{T}

The [`Chi2Tally`](@ref)s of one time resolution: `total`, `per_station`
(station name ⇒ tally of its baselines) and `per_product` (feed pair ⇒ tally).
"""
struct Chi2Level{T <: Real}
    total::Chi2Tally{T}
    per_station::Dict{String, Chi2Tally{T}}
    per_product::Dict{Tuple{Int, Int}, Chi2Tally{T}}
end

"""
    BandpassChi2{A, S}

The result of [`bandpass_chi2`](@ref): the [`Chi2Level`](@ref)s `ap`
(per integration, or `nothing`) and `scan` (per scan average). Read it with
[`total_chi2`](@ref) and [`station_chi2`](@ref), and print it with
[`print_chi2`](@ref).
"""
struct BandpassChi2{A <: Union{Nothing, Chi2Level}, S <: Chi2Level}
    ap::A
    scan::S
end

"""
    bandpass_chi2(ps; min_channels = 8, per_integration = true) -> BandpassChi2

The χ² of the corrected processing set `ps` against a source that is constant
(`const`) or linear (`lin`) in frequency, over cross baselines and parallel
feed pairs (both feeds the same index). Once a bandpass is divided out, each
spectrum should be the source alone; channel structure left over is bandpass
error. The linear model also absorbs a residual delay or a source spectral
index, which are close to linear over a narrow band, so it is the one to
compare bandpass models with.

A spectrum is the channels of one Measurement Set on one baseline, feed pair
and time sample: each integration of `ps` for the `ap` resolution, and each
sample of `XRadio.average(ps, XRadio.ByScan())` for the `scan` resolution. A
bandpass error is constant over a scan while the noise averages down, so the
`scan` χ² shows it more clearly. A residual delay drifting through a scan also
decoheres the average, more in the outer channels, which raises the `scan` χ²
by the same amount for every model compared on the same data.

Each spectrum with at least `min_channels` usable channels (unflagged, finite,
positive weight) contributes `2(n - 1)` degrees of freedom to `const` and
`2(n - 2)` to `lin`. `per_station` counts each baseline under both its
stations, so the station sums exceed the total. `per_integration = false`
skips the `ap` resolution, which is most of the cost, and leaves `ap` as
`nothing`.

The weights must be inverse variances of each real component of the
visibilities, which `scale_weights!(ms, weight_scales(ms))` on each Measurement
Set makes them. `calibrate` keeps them so.
"""
function bandpass_chi2(ps; min_channels = 8, per_integration = true)
    ap = per_integration ? _level_chi2(ps, min_channels) : nothing
    scan = _level_chi2(XRadio.average(XRadio.transform(_without_integration_time, ps), XRadio.ByScan()), min_channels)
    return BandpassChi2(ap, scan)
end

# `average` cannot average `effective_integration_time`, which the χ² does not read.
function _without_integration_time(ms)
    haskey(ms, :effective_integration_time) || return ms
    return delete!(copy(ms), :effective_integration_time)
end

function _level_chi2(ps, min_channels)
    total = zero(Chi2Tally{Float64})
    per_station = Dict{String, Chi2Tally{Float64}}()
    per_product = Dict{Tuple{Int, Int}, Chi2Tally{Float64}}()
    for ms in values(ps)
        V, W, F = _member_layers(ms)
        ν = lookup(V, Frequency)
        antennas = _member_antenna_pairs(ms)
        feeds = feed_pairs(ms)
        for bi in axes(V, BaselineID), p in axes(V, Polarization)
            a, b = antennas[bi]
            fp = feeds[p, bi]
            (a != b && fp[1] == fp[2]) || continue
            for t in axes(V, Ti)
                cell = (Polarization(p), BaselineID(bi), Ti(t))
                x = _spectrum_chi2(view(V, cell...), view(W, cell...), view(F, cell...), ν, min_channels)
                isnothing(x) && continue
                total += x
                per_station[a] = get(per_station, a, zero(total)) + x
                per_station[b] = get(per_station, b, zero(total)) + x
                per_product[fp] = get(per_product, fp, zero(total)) + x
            end
        end
    end
    return Chi2Level(total, per_station, per_product)
end

# Weighted least squares of one complex spectrum against a constant and against
# a line in frequency centered on its weighted mean frequency, where the two
# basis columns are orthogonal; `nothing` when fewer than `min_channels`
# channels are usable. Frequencies are taken from the first usable channel, so
# the weighted sums stay small enough for single precision.
function _spectrum_chi2(v, w, f, ν, min_channels)
    ks = [k for k in eachindex(v, w, f, ν) if _usable(f[k], w[k], v[k])]
    n = length(ks)
    n >= min_channels || return nothing
    x = [ν[k] - ν[first(ks)] for k in ks]
    ws = [w[k] for k in ks]
    vs = [v[k] for k in ks]
    W = sum(ws)
    m = sum(ws .* vs) / W
    x̄ = sum(ws .* x) / W
    slope = sum(ws .* (x .- x̄) .* vs) / sum(ws .* (x .- x̄) .^ 2)
    χ²c = sum(ws .* abs2.(vs .- m))
    χ²l = sum(ws .* abs2.(vs .- m .- slope .* (x .- x̄)))
    return Chi2Tally{Float64}(χ²c, 2 * (n - 1), χ²l, 2 * (n - 2))
end

_tally_model(t::Chi2Tally, model::Symbol) =
    model === :lin ? (; chi2 = t.chi2_lin, dof = t.dof_lin) :
    model === :const ? (; chi2 = t.chi2_const, dof = t.dof_const) :
    throw(ArgumentError("model must be :const or :lin, got $(repr(model))"))

"""
    total_chi2(res::BandpassChi2; model = :lin) -> (; chi2, dof)

The χ² and degrees of freedom of the whole set at the `scan` resolution, about
the `:lin` or `:const` model.
"""
total_chi2(res::BandpassChi2; model = :lin) = _tally_model(res.scan.total, model)

"""
    station_chi2(res::BandpassChi2; model = :lin) -> Dict{String, NamedTuple}

The χ² and degrees of freedom `(; chi2, dof)` of each station's baselines at
the `scan` resolution, about the `:lin` or `:const` model. A baseline counts
under both its stations.
"""
station_chi2(res::BandpassChi2; model = :lin) =
    Dict(st => _tally_model(t, model) for (st, t) in res.scan.per_station)

_reduced(χ², dof) = χ² / dof

"""
    print_chi2([io], res::BandpassChi2; nparams = 0, station_params = Dict{String, Int}())

Print `res` as tables of reduced χ²: the totals of each resolution, then per
station and per feed pair at the `scan` resolution. The totals' degrees of
freedom are charged the `nparams` free parameters of the solution, and each
station's are charged its own, `station_params[station]` (0 when absent), so
that models with different numbers of parameters compare: a model must lower
χ² by more than the parameters it adds. Per feed pair, nothing is charged.
"""
function print_chi2(io::IO, res::BandpassChi2; nparams = 0, station_params = Dict{String, Int}())
    println(io, "bandpass χ²/dof   free parameters $nparams")
    for (name, level) in (("ap  ", res.ap), ("scan", res.scan))
        isnothing(level) && continue
        t = level.total
        @printf(
            io, "  %s total   const %9.4f   lin %9.4f   (dof %d / %d)\n",
            name, _reduced(t.chi2_const, t.dof_const - nparams), _reduced(t.chi2_lin, t.dof_lin - nparams),
            t.dof_const - nparams, t.dof_lin - nparams,
        )
    end
    println(io, "  per station, scan resolution (a baseline counts under both stations)")
    @printf(io, "    %-8s %8s %10s %10s %10s\n", "station", "params", "dof", "const", "lin")
    for st in sort!(collect(keys(res.scan.per_station)))
        t = res.scan.per_station[st]
        k = get(station_params, st, 0)
        @printf(
            io, "    %-8s %8d %10d %10.4f %10.4f\n", st, k, t.dof_lin - k,
            _reduced(t.chi2_const, t.dof_const - k), _reduced(t.chi2_lin, t.dof_lin - k),
        )
    end
    println(io, "  per feed pair, scan resolution")
    for fp in sort!(collect(keys(res.scan.per_product)))
        t = res.scan.per_product[fp]
        @printf(io, "    %-8s %10.4f %10.4f\n", "$(fp[1])-$(fp[2])", _reduced(t.chi2_const, t.dof_const), _reduced(t.chi2_lin, t.dof_lin))
    end
    return nothing
end
print_chi2(res::BandpassChi2; kwargs...) = print_chi2(stdout, res; kwargs...)

Base.show(io::IO, ::MIME"text/plain", res::BandpassChi2) = print_chi2(io, res)
function Base.show(io::IO, res::BandpassChi2)
    t = res.scan.total
    return print(io, "BandpassChi2(scan χ²/dof lin ", round(_reduced(t.chi2_lin, t.dof_lin); sigdigits = 4), ")")
end
