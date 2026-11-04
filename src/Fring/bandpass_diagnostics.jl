# ── Bandpass diagnostics ─────────────────────────────────────────────────────
#
# Goodness of fit of corrected visibilities against a channel-constant source,
# and per-station tests for a bandpass that changes between two scans.

# ── Normalized residuals ─────────────────────────────────────────────────────

const VisibilityData = Union{XRadio.ProcessingSet, XRadio.MeasurementSet}

"""
    normalized_residuals(data; dims = Frequency) -> DimStack

Every cut of `data` along `dims` divided by its weighted mean along `dims`.
`data` is a processing set, a Measurement Set or a [`baseline_spectra`](@ref)
result, laid out as `baseline_spectra` lays it out, with the weight the
inverse variance of each real component of the visibilities. Layers, over the
dimensions of [`baseline_spectra`](@ref):

- `r = v / m`, with `m = Σ w v / Σ w` over the cut. Along `Frequency` this
  removes each spectrum's phase, amplitude scale and band-mean source
  visibility, leaving whatever structure across the band the correction did not
  remove, a residual delay included. Along `Ti` it shows each channel's change
  about its mean over time.
- `var = 1 / (w |m|²)`, the variance of each real component of `r`, ignoring
  the noise in `m`.
- `scan_name`, over `Ti`.

A flagged sample, or one with zero weight or non-finite data, is `NaN` in `r`
and `var`, and so is every sample of a cut with no usable sample.
"""
normalized_residuals(data::VisibilityData; dims = Frequency) = normalized_residuals(baseline_spectra(data); dims)

function normalized_residuals(spectra::AbstractDimStack; dims = Frequency)
    vis, weight, flag = spectra.visibility, spectra.weight, spectra.flag
    R = float(promote_type(real(eltype(vis)), eltype(weight)))
    r = fill!(similar(vis, Complex{R}), Complex{R}(NaN, NaN))
    var = fill!(similar(weight, R), R(NaN))
    for cell in DimensionalData.DimIndices(DimensionalData.otherdims(vis, dims))
        v, w, f = view(vis, cell...), view(weight, cell...), view(flag, cell...)
        ok = [k for k in eachindex(v, w, f) if _usable(f[k], w[k], v[k])]
        isempty(ok) && continue
        m = sum(k -> w[k] * v[k], ok) / sum(k -> w[k], ok)
        rc, varc = view(r, cell...), view(var, cell...)
        for k in ok
            rc[k] = v[k] / m
            varc[k] = inv(w[k] * abs2(m))
        end
    end
    return DimStack((; r, var, spectra.scan_name))
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
along `Frequency`, usually of scan averages (`average(ps, ByScan())`), and
`intervals` the [`scan_intervals`](@ref) of the data before averaging.

Every series of residual spectra `r` on a baseline to `station` and a parallel
feed pair (both feeds the same index) is split at the gap and fitted by one
weighted mean spectrum per side, channel by channel, using the channels with at
least two samples on each side. Summed over the series,

    F = (Δχ² / ν₁) / (χ²_split / ν₂),

with `Δχ²` the χ² that two means remove relative to one, `ν₁ = 2` per channel,
`χ²_split` the χ² about the two means and `ν₂ = 2(n - 2)` per channel of `n`
samples. Scatter between scans that is not a step (residual delays, phase noise)
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
    scan_names, antenna_pairs, feeds = parent(residuals.scan_name), lookup(r, AntennaPair), lookup(r, FeedPair)
    cells = [
        (j, q) for j in axes(r, AntennaPair), q in axes(r, FeedPair)
            if station in antenna_pairs[j] && feeds[q][1] == feeds[q][2]
    ]
    series = map(cells) do (j, q)
        Rs = parent(permutedims(r[AntennaPair(j), FeedPair(q)], (Frequency, Ti)))
        Vs = parent(permutedims(var[AntennaPair(j), FeedPair(q)], (Frequency, Ti)))
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

# Δχ², ν₁, χ²_split and ν₂ of one series `R` (channel × time) with per-component
# variances `V`, over the time samples `scans`, split by `before`.
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

# ── Goodness of fit ──────────────────────────────────────────────────────────

"""
    chi2(data, model; dims) -> DimStack

The χ² of `data` against `model`, `Σ w |v - model|²`, summed over `dims`.
`data` is a processing set, a Measurement Set or a [`baseline_spectra`](@ref)
result, with the weight the inverse variance of each real component of the
visibilities. `model` is a `DimArray` of model visibilities over some of the
dimensions of [`baseline_spectra`](@ref), used for every sample along the
dimensions it lacks.

Returns layers `chi2` and `dof` over the dimensions not in `dims`. Each usable
sample (unflagged, positive weight, finite data) counts 2 degrees of freedom.
The model's own free parameters are not charged; subtract them from `dof`.
"""
chi2(data::VisibilityData, model::AbstractDimArray; dims) = chi2(baseline_spectra(data), model; dims)

function chi2(spectra::AbstractDimStack, model::AbstractDimArray; dims)
    vis, weight, flag = spectra.visibility, spectra.weight, spectra.flag
    usable = _usable.(flag, weight, vis)
    terms = @d ifelse.(usable, weight .* abs2.(vis .- model), zero(real(eltype(vis))))
    samples = ifelse.(usable, 2, 0)
    return DimStack((; chi2 = _sum_over(terms, dims), dof = _sum_over(samples, dims)))
end

"""
    tss(data; dims = Frequency) -> DimStack

The weighted sum of squares of `data` about its weighted mean along `dims`.
For each cut along `dims`, with mean `m = Σ w v / Σ w`, this is
`Σ w |v - m|²`. `data` is a processing set, a Measurement Set or a
[`baseline_spectra`](@ref) result, with the weight the inverse variance of
each real component of the visibilities.

Returns layers `tss` and `dof` over the dimensions of
[`baseline_spectra`](@ref) not in `dims`. A cut with `n` usable samples
(unflagged, positive weight, finite data) has `2(n - 1)` degrees of freedom; a
cut with none has 0 of both.
"""
tss(data::VisibilityData; dims = Frequency) = tss(baseline_spectra(data); dims)

function tss(spectra::AbstractDimStack; dims = Frequency)
    vis, weight, flag = spectra.visibility, spectra.weight, spectra.flag
    cuts = DimensionalData.otherdims(vis, dims)
    q = zeros(Float64, cuts)
    ν = zeros(Int, cuts)
    for cell in DimensionalData.DimIndices(cuts)
        v, w, f = view(vis, cell...), view(weight, cell...), view(flag, cell...)
        ok = [k for k in eachindex(v, w, f) if _usable(f[k], w[k], v[k])]
        isempty(ok) && continue
        m = sum(k -> w[k] * v[k], ok) / sum(k -> w[k], ok)
        q[cell...] = sum(k -> w[k] * abs2(v[k] - m), ok)
        ν[cell...] = 2 * (length(ok) - 1)
    end
    return DimStack((; tss = q, dof = ν))
end

_sum_over(A, dims) = isempty(dims) ? A : dropdims(sum(A; dims); dims)
_sum_over(A, dims::Union{Type, DimensionalData.Dimension}) = _sum_over(A, (dims,))
