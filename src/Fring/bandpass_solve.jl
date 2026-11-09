# ── The bandpass solve ───────────────────────────────────────────────────────

"""
    solve_joint_bandpass!(scans, geom::DataGeometry, components; gauge, flat_span = 0.01,
                          eltype = nothing, kw...) -> (; phase, amp)

Solve the bandpass components `components` (`(; phase, logamp)`, each a vector
of components, each the station blocks [`antenna_blocks`](@ref) returns) on the
scans `scans`, the per-scan `(rl, wl)` accumulator pairs from
`accumulate_bandpass` with each scan's first sample `ti`, and write their
parameters into the blocks' `θ`.

The scans are reduced onto the common refinement of every component's
frequency segments, each entry covering its scan's first sample and its
cell's first channel, and solved by [`solve_joint_gains!`](@ref), to which `kw`
pass: a per-scan, per-station-pair, per-feed-pair source term is profiled out,
so the solve suits a resolved or polarized calibrator, and the gauge fixes the
phase each connected group of station gains leaves free. Solved phases are
comparable only within such a group: where the graph splits (disjoint
sub-arrays, or every station segmented at the same epoch) each piece carries
its own constant.

Returns `(; phase, amp, status)`: per observable the chains
`solve_joint_gains!` returns, each with its prior's hyperparameters resolved;
and per observable `(:phase` and `:logamp`) the [`TrackStatus`](@ref) of each
(component path, station block, station position, feed node, time segment,
spectral window) its parameters cover, from the station-gain values they
reach: no data, flat where they span less than `flat_span`, or solved. The data are reduced in
the accumulators' types, or in the real type `eltype`; the steps are solved in
`Float64`.
"""
function solve_joint_bandpass!(scans, geom::DataGeometry, components; flat_span::Real = 0.01, eltype = nothing, kw...)
    isempty(scans) && return (; phase = [], amp = [], status = (; phase = Dict(), logamp = Dict()))
    cells = _frequency_cells(components, length(geom.channel_freqs))
    r, w = _reduce_all_scans(scans, cells; eltype)
    cover = [(sc.ti, first(chans)) for sc in scans, chans in cells]
    fits = solve_joint_gains!(components, (; r, w), geom; cover, kw...)
    (; sg) = fits.gains
    status = (;
        phase = _window_status(components.phase, sg.params.phase, sg.J.phase, fits.gains.phase, geom, flat_span),
        logamp = _window_status(components.logamp, sg.params.logamp, sg.J.logamp, fits.gains.amp, geom, flat_span),
    )
    return (; fits.phase, fits.amp, status)
end

# The status of each (component path, block, station position, feed node, time
# segment, spectral window) the parameters `params` cover, from the station-gain
# values `x` (`NaN` without data) each reaches through `J`.
function _window_status(components, params, J, x, geom::DataGeometry, flat_span)
    values = Dict{Tuple{Any, Int, Int, Int, Int, Int}, Vector{Float64}}()
    window = Dict{Tuple{Int, Int, Int}, Int}()
    for (u, (k, j, i)) in pairs(params)
        b = components[k][j]
        _, node, fs, ts, ai = Tuple(CartesianIndices(b.θ)[i])
        spw = get!(() -> geom.spw_of_chan[findfirst(==(fs), b.plan.fseg_id)], window, (k, j, fs))
        v = get!(Vector{Float64}, values, (b.plan.path, j, ai, node, ts, spw))
        foreach(r -> push!(v, x[rowvals(J)[r]]), nzrange(J, u))
    end
    return Dict(key => _band_track_status(v, flat_span) for (key, v) in values)
end

# The channels of each cell of the common refinement of every component's
# frequency segments, over every station block, in channel order.
function _frequency_cells(components, nchan)
    key = [Int[] for _ in 1:nchan]
    for cs in components, c in cs, b in c, ch in 1:nchan
        push!(key[ch], b.plan.fseg_id[ch])
    end
    cells = Vector{Int}[]
    for ch in 1:nchan
        ch > 1 && key[ch] == key[ch - 1] ? push!(last(cells), ch) : push!(cells, [ch])
    end
    return cells
end

# Record a joint bandpass solve's fits (`solve_joint_bandpass!`) of the
# component at `path` into one array per station block of their status
# (`_block_status_array`, by spectral window) from `status`, and of their
# chains' priors (`_block_prior_array`) from `fits`, each array `nothing` to
# skip.
function _record_track_fits!(status_arrays, prior_arrays, fits, status, path, geom::DataGeometry, blocks)
    if !isnothing(status_arrays)
        for ((p, j, ai, node, ts, spw), st) in status
            p == path || continue
            windows = [geom.spw_of_chan[first(chans)] for chans in _shape_segments(blocks[j].plan, geom).piece_chans]
            status_arrays[j][ai, node, findfirst(==(spw), windows), ts] = st
        end
    end
    isnothing(prior_arrays) && return nothing
    for f in fits
        f.path == path && (prior_arrays[f.block][f.station, f.node, f.across] = f.prior)
    end
    return nothing
end

# Whether `x′` differs from `x` by no more than `x`'s own rounding.
_within_rounding(x′, x) = all(i -> abs(x′[i] - x[i]) <= eps(abs(x[i])), eachindex(x′, x))

# The largest change of any station's log-gain, `(Δℓ, Δφ)` with the phase change
# wrapped, in standard errors of its weight `W` from the data, and that gain.
function _largest_weighted_change(Δℓ, Δφ, W)
    worst, at = 0.0, 0
    for n in eachindex(Δℓ, Δφ, W)
        W[n] > 0 || continue
        c = sqrt(W[n]) * abs(complex(Δℓ[n], rem2pi(Δφ[n], RoundNearest)))
        c > worst && ((worst, at) = (c, n))
    end
    return worst, at
end

# The data leave each connected component's common phase free, and the complex
# start leaves it wherever its rounds settle, so on a track it shows as steps
# between neighboring components. Taking the components in order of their first
# gain along the tracks, this rotates each one whose gains are all `free` by the
# weighted circular mean of its gains' differences from their predecessors on
# the same track, which removes those steps for the branch choice. A component
# holding a gain that is not `free` keeps its phase.
function _align_component_phases!(φ, compid, free, weight, tracks)
    members = [Int[] for _ in 1:maximum(compid; init = 0)]
    foreach(n -> iszero(compid[n]) || push!(members[compid[n]], n), eachindex(compid))
    prev = zeros(Int, length(φ))
    first_at = fill(typemax(Int), length(members))
    for t in tracks
        last = 0
        for (i, n) in pairs(t.gains)
            c = compid[n]
            iszero(c) && continue
            first_at[c] = min(first_at[c], i)
            prev[n] = last
            last = n
        end
    end
    for c in sortperm(first_at)
        all(n -> free[n], members[c]) || continue
        acc = zero(ComplexF64)
        for n in members[c]
            p = prev[n]
            (iszero(p) || compid[p] == c) && continue
            acc += weight[n] * weight[p] / (weight[n] + weight[p]) * cis(φ[p] - φ[n])
        end
        iszero(acc) && continue
        δ = angle(acc)
        foreach(n -> φ[n] += δ, members[c])
    end
    return φ
end

# The false-alarm probability under which `_align_feed_delays!` takes a feed's
# delay as detected.
const _FEED_DELAY_PFA = 1.0e-4

# Adding one phase spectrum `ρ(ν)` to feed `f` of every station leaves every
# same-feed product unchanged and turns each product pairing `f` with another
# feed by `±ρ`, so only those products measure it. For each feed `f` after the
# first, in turn, this searches one delay `τ`, `ρ = 2πτ(ν − ν̄)` over the gains'
# frequencies `ν`, against the products pairing `f` with an earlier feed. With
# `S` profiled, a group `J` lowers the χ² by `|Σ_c e^{∓iρ(ν_c)} conj(u_c) r_c|²
# / Σ_c w_c|u_c|²`, `u = g_a·conj(g_b)`, which on noise alone is χ² with two
# degrees of freedom, since the weights are inverse variances per real
# component. The trial delays span `±1/(2δν)` in steps of `1/(4B)`, over the
# band `B` and the smallest spacing `δν` of the frequencies, so the best one is
# within `π/8` of the peak's phase at the band's edges. A peak whose
# false-alarm probability over those trials is below `_FEED_DELAY_PFA` is
# applied to `φ`; otherwise the feed's delay is left where it is, with a
# warning.
function _align_feed_delays!(φ, ℓ, cells::BandpassCells, feed, ν)
    for f in 2:maximum(feed)
        freqs = sort!(unique(ν[n] for n in eachindex(ν, feed) if feed[n] == f))
        length(freqs) > 1 || continue
        column = Dict(v => i for (i, v) in pairs(freqs))
        g = exp.(complex.(ℓ, φ))
        groups = Vector{ComplexF64}[]
        for rng in cells.ranges
            fa, fb = feed[cells.na[first(rng)]], feed[cells.nb[first(rng)]]
            ((fa == f && fb < f) || (fb == f && fa < f)) || continue
            y = zeros(ComplexF64, length(freqs))
            den = 0.0
            for k in rng
                a, b = cells.na[k], cells.nb[k]
                u = g[a] * conj(g[b])
                v = conj(u) * cells.r[k]
                fa == f ? (y[column[ν[a]]] += v) : (y[column[ν[b]]] += conj(v))
                den += cells.w[k] * abs2(u)
            end
            den > 0 && push!(groups, y ./ sqrt(den))
        end
        isempty(groups) && continue
        Y = reduce(hcat, groups)
        ν̄ = sum(freqs) / length(freqs)
        spacing = 1 / (4 * (last(freqs) - first(freqs)))
        n = ceil(Int, 1 / (2 * minimum(diff(freqs))) / spacing)
        τs = (-n:n) .* spacing
        power = zeros(length(τs))
        for trials in Iterators.partition(eachindex(τs), 256)
            power[trials] .= vec(sum(abs2, cis.(-2π .* τs[trials] .* (freqs .- ν̄)') * Y; dims = 2))
        end
        best = argmax(power)
        pfa = -expm1(length(τs) * log1p(-ccdf(Chisq(2 * size(Y, 2)), power[best])))
        if pfa > _FEED_DELAY_PFA
            @warn "the joint solve found no delay of feed $f common to every station in the " *
                "$(size(Y, 2)) groups of products pairing it with an earlier feed: the best of " *
                "$(length(τs)) trial delays has false-alarm probability $pfa. The solve keeps the " *
                "delay the data start with, and may stall short of the right one if that is far off."
            continue
        end
        for m in eachindex(φ, ν, feed)
            feed[m] == f && (φ[m] += 2π * τs[best] * (ν[m] - ν̄))
        end
    end
    return φ
end
