# ── Fringe fitting: per-scan search, then closure over the detections ────────

"""
    fit(search::FringeSearch, ps::ProcessingSet; at = nothing, exec = ExecutionConfig()) -> Vector{DimStack}

Fringe-search every scan of `ps`: per scan, every cross (antenna pair, feed
pair) cell is searched over `search`'s delay and rate windows, across all the
scan's spectral windows. Returns one `DimStack` per scan, in scan order, over
`AntennaPair × FeedPair` with the layers `delay` (s), `rate` (Hz), `phase`
(rad, at the scan's mean time), `amp`, `snr`, `pfa` and `valid`. No cell is
gated: [`FringeClosure`](@ref)'s `pfa_max` decides which are real fringes.

Each stack's metadata records the scan: `scan` (its name), `epoch` (its mean
time), `geom` (its [`DataGeometry`](@ref)), `search`, `ncells` (the trial count
of the recorded `pfa`), `freq_mean`, and `freq_rms` and `time_rms` (the RMS
spreads of its channels and times, which set the uncertainty of a delay and a
rate).

With `at`, a [`CalibrationSolution`](@ref) holding delay and rate terms, every
cell is instead measured at the delay and rate `at` predicts for it
(`τ_{a,fa} − τ_{b,fb}`), so a fringe too weak for the blind search is still
measured. Its `pfa` is over `search.steer_cells` trials. A cell is `NaN` where
`at` holds no constrained value for either station's feed in the scan.
"""
function StatsAPI.fit(
        search::Fring.FringeSearch, ps::XRadio.ProcessingSet;
        at::Union{Nothing, CalibrationSolution} = nothing, exec::ExecutionConfig = ExecutionConfig(),
    )
    _check_blas_threads()
    groups = groupby(ps, XRadio.ByScan())
    sizes = [_group_bytes(g) for g in values(groups)]
    return _map_groups(groups, sizes, exec; stage = :fringe_search) do group
        _search_one_scan(search, _read_group(group, inner_executor(exec)), at, inner_executor(exec))
    end
end

StatsAPI.fit(search::Fring.FringeSearch, ms::XRadio.MeasurementSet; kwargs...) =
    fit(search, _one_member(ms); kwargs...)

function _search_one_scan(search::Fring.FringeSearch, group::XRadio.ProcessingSet, at, executor)
    geom = DataGeometry(group)
    gc = Fring._GroupCells(group, geom)
    epoch = sum(gc.times) / length(gc.times)
    res = Fring.search_scan(gc, geom, search; executor, t0 = epoch)
    scan = only(geom.scan_names)
    meta = (;
        scan, epoch, geom, search, ncells = Fring._family_cells(gc, search),
        freq_mean = sum(gc.freqs) / length(gc.freqs),
        freq_rms = Fring._rms_spread(gc.freqs), time_rms = Fring._rms_spread(gc.times),
    )
    isnothing(at) && return DimensionalData.rebuild(res; metadata = meta)
    delay, rate = _predicted_terms(at, scan, geom.antennas, geom.nfeed)
    s = Fring.steer_scan(gc, res, geom.f0, epoch, delay, rate; cells = search.steer_cells, executor)
    valid = res.valid .& .!isnan.(s.snr)
    return DimensionalData.DimStack((; s..., valid); metadata = merge(meta, (; ncells = search.steer_cells)))
end

# The station delay and rate `sol` predicts in scan `name`, over
# `AntennaName(antennas) × Feed(1:nfeed)`: `NaN` for a station or feed the
# solution does not hold, or holds but left unconstrained in that scan.
function _predicted_terms(sol::CalibrationSolution, name, antennas, nfeed)
    s = findfirst(==(name), sol.geom.scan_names)
    isnothing(s) && throw(
        ArgumentError(
            "scan $(repr(name)) is not in the solution `at`, which covers " *
                "$(join(map(repr, sol.geom.scan_names), ", "))",
        ),
    )
    ti = findfirst(==(s), sol.geom.scan_of_time)
    delay = fill(NaN, AntennaName(collect(antennas)), Feed(1:nfeed))
    rate = fill(NaN, AntennaName(collect(antennas)), Feed(1:nfeed))
    for g in Calibration._applied(sol).groups
        names = sol.geom.antennas[g.antennas]
        d, r = Fring.scan_antenna_terms(g.model, g.layout, g.θ, ti, names)
        for a in names, f in lookup(d, Feed)
            (a in antennas && f <= nfeed) || continue
            delay[AntennaName(At(a)), Feed(At(f))] = d[AntennaName(At(a)), Feed(At(f))]
            rate[AntennaName(At(a)), Feed(At(f))] = r[AntennaName(At(a)), Feed(At(f))]
        end
    end
    # A dense θ reads an unconstrained station as an identity 0, which would
    # put a prediction where the solution has none.
    flagged = _solution_flag_sets(sol)
    isnothing(flagged) && return delay, rate
    for (a, f, k) in flagged
        (k == s && a in antennas && f <= nfeed) || continue
        delay[AntennaName(At(a)), Feed(At(f))] = NaN
        rate[AntennaName(At(a)), Feed(At(f))] = NaN
    end
    return delay, rate
end

"""
    FringeClosure(; model = default_fringe_model(), closure = Stationization(), gauge = ZeroSumPhase())

The station solve of a fringe fit: from the per-baseline detections of
`fit(::FringeSearch, ps)`, the per-(station, feed) delays and rates of
`model`, a phase-only [`GainModel`](@ref) (see [`Fring.default_fringe_model`](@ref)).
`closure`, a [`Fring.Stationization`](@ref), holds the detection threshold
`pfa_max`, which decides which detections are real fringes and so which
stations are calibrated, and how inconsistent detections are weighted.
`gauge`, an [`AbstractGauge`](@ref), fixes what the detections leave
undetermined; a [`ByComponent`](@ref) names components of `model`.

`fit(FringeClosure(...), dets)` solves the scans of `dets`, a vector of
detection stacks, together, and returns a [`CalibrationSolution`](@ref) over
their joint geometry. A model whose components are all per scan solves each
scan on its own; a component spanning scans, such as a `GlobalTime` inter-feed
delay, couples them, so close one scan at a time inside [`mapsets`](@ref), or
every scan at once:

```julia
dets = reduce(vcat, values(mapsets(g -> fit(FringeSearch(), g), groupby(ps, ByScan()))))
sol = fit(FringeClosure(; model = default_fringe_model(; rel_time = GlobalTime())), dets)
```

The models it fits: a `Delay` or `Rate` spanning the whole band
(`Frequency = GlobalFrequency()`), under any feed scope, with a time
segmentation no finer than a scan, and with no prior or a `RandomWalkPrior`,
`OUPrior` or `CARMAPrior` along `Ti` with numbers for hyperparameters. Such a
prior ties the component's values across its time segments, and so couples the
scans: a segment no accepted detection constrains takes the prior's
interpolation of its neighbors rather than being flagged. A drifting
inter-feed delay is a flat level beside a mean-zero drift:

```julia
rel_delay = GainComponent(Delay(); Ti = GlobalTime(), Feed = ExceptFeed(1))
rel_drift = GainComponent(Delay(); Ti = PerScan(), Feed = ExceptFeed(1),
                          prior = OUPrior(; scale = 3600.0, σ = 3.0e-11))
model = merge(default_fringe_model(); phase = (; rel_delay, rel_drift))
```
 A segmentation that splits a scan (a
`TimeBlocks` shorter than the scans, `PerIntegration`) asks for values the
search does not measure and is rejected. It fits no constant phase, no
dispersion (`Dispersion`) and no per-band-group delay (a `FreqGroups`-segmented
`Delay`).

The solution's info holds `ncomp`, the number of fringe groups the detections
form, and the `(station, feed, scan)` triples no accepted detection constrains
(`flagged_ant`, `flagged_feed`, `flagged_scan`), which [`calibrate!`](@ref)
flags.
"""
Base.@kwdef struct FringeClosure{M <: GainModel, C <: Fring.Stationization, G <: AbstractGauge}
    model::M = Fring.default_fringe_model()
    closure::C = Fring.Stationization()
    gauge::G = ZeroSumPhase()
    function FringeClosure(model::M, closure::C, gauge::G) where {M, C, G}
        _check_solver_gauge(gauge, model)
        return new{M, C, G}(model, closure, gauge)
    end
end

# One search per scan yields one delay and rate per scan, written to the
# column holding the scan's first epoch. A term whose time segmentation splits
# a scan would leave the scan's other columns at identity, so `TimeBlocks`
# coarser than the scans is fitted and finer is not.
Fring.can_fit(::FringeClosure, tc, geom) =
    !isnothing(Fring.matched_kind(tc)) && !Fring.splits_a_scan(tc.Ti, geom)
Fring.validate_model(::FringeClosure, model) = Fring.validate_fringe_model(model)

function StatsAPI.fit(c::FringeClosure, dets::AbstractVector; exec::ExecutionConfig = ExecutionConfig())
    isempty(dets) && throw(ArgumentError("there are no detections to close"))
    geom = DataGeometry([metadata(d).geom for d in dets])
    gauge = resolve_gauge(c.gauge, geom.antennas)
    model = Calibration.resolve(_vetted_model(c, geom), geom.antennas, geom)
    Calibration.require_antenna_uniform(model, geom.antennas, "FringeClosure")
    layout = plan_parameters(model, geom.antennas, geom; require_nonempty = false)
    θ = zeros(layout.nθ)
    stageB = Fring.fringe_stage_components(model, layout)
    has_rate = any(k -> last(k) === :rate, stageB)
    scans = map(d -> _closure_scan(d, geom, has_rate), dets)
    # A mixture's floors and outlier fractions are shared by every scan.
    per_scan = _closes_per_scan(c.model) && !(c.closure.loss isa Fring.OutlierMixture)
    blocks = per_scan ? [[k] for k in eachindex(scans)] : [collect(eachindex(scans))]
    results = _scheduled_map(blocks, ones(Int, length(blocks)); executor = outer_executor(exec)) do ks
        block = scans[ks]
        ncomp, covered, unlinked, fits = Fring.solve_antenna_systems!(θ, block, stageB, geom.antennas; gauge, opts = c.closure, geom.times)
        scan_name(si) = geom.scan_names[geom.scan_of_time[Fring._scan_ti(block[si])]]
        (;
            ncomp, fits, flags = Fring.unconstrained_flags(block, covered, geom),
            unlinked = Tuple{Symbol, String}[(kind, scan_name(si)) for (kind, si) in unlinked],
        )
    end
    _warn_unlinked(reduce(vcat, (r.unlinked for r in results)))
    flags = reduce(vcat, (r.flags for r in results))
    info = (; nscan = length(dets), ncomp = sum(r.ncomp for r in results), Fring.flag_table(flags)...)
    em = mapreduce(r -> r.fits, merge, results)
    if !isempty(em)
        _warn_em(em, geom, dets)
        info = (; info..., _em_info(em, geom)...)
    end
    return CalibrationSolution(model, layout, geom, θ, info; name = :fringe, spec = sprint(show, c; context = :limit => true))
end

# The closure's EM results for the solution info: the EM iterations per kind;
# per component whose prior gives hyperparameters as densities, its resolved
# priors over `(AntennaName, FeedNode)`; and per kind under `OutlierMixture`,
# the floor and outlier fraction of the same-feed and different-feed
# detections.
function _em_info(em, geom)
    priors = Dict{Symbol, Any}()
    mixture = Dict{Symbol, Any}()
    iterations = Dict{Symbol, Int}()
    for (kind, fit) in em
        iterations[kind] = fit.iterations
        isnothing(fit.mixture) || (mixture[kind] = (; same_feed = (; floor = fit.mixture.floor[1], outlier_fraction = fit.mixture.outlier_fraction[1]),
            different_feed = (; floor = fit.mixture.floor[2], outlier_fraction = fit.mixture.outlier_fraction[2])))
        for path in unique(ch.path for ch in fit.chains)
            idx = [i for (i, ch) in pairs(fit.chains) if ch.path == path]
            names = unique(geom.antennas[fit.chains[i].antenna] for i in idx)
            nodes = sort!(unique(fit.chains[i].node for i in idx))
            out = Array{Any}(undef, length(names), length(nodes))
            for i in idx
                ch = fit.chains[i]
                out[findfirst(==(geom.antennas[ch.antenna]), names), findfirst(==(ch.node), nodes)] = fit.priors[i]
            end
            priors[last(path)] = DimensionalData.DimArray(map(identity, out), (AntennaName(names), FeedNode(nodes)))
        end
    end
    out = (; em_iterations = NamedTuple(iterations))
    isempty(priors) || (out = (; out..., priors = NamedTuple(priors)))
    isempty(mixture) || (out = (; out..., mixture = NamedTuple(mixture)))
    return out
end

# Warn of an EM that ran out of iterations, and of the detections each
# (station, feed) has that the station values more likely do not explain, with
# their median offset from the solution as that station sees it.
function _warn_em(em, geom, dets)
    for (kind, fit) in em
        fit.converged || @warn "FringeClosure: the $kind solve's EM did not converge; raise " *
            "`Stationization(; max_em_iterations)`"
        isempty(fit.outliers) && continue
        unit, scale = kind === :delay ? ("ns", 1.0e9) : ("mHz", 1.0e3)
        offsets = Dict{Tuple{String, Int, Bool}, Vector{Float64}}()
        for r in fit.outliers, (st, f, sign) in ((r.sta_a, r.feed_a, 1), (r.sta_b, r.feed_b, -1))
            push!(get!(offsets, (geom.antennas[st], f, r.feed_a == r.feed_b), Float64[]), sign * scale * r.residual)
        end
        totals = _detection_counts(dets)
        lines = [
            "$name feed $f, $(same ? "same" : "different")-feed detections: $(length(v)) of " *
                "$(get(totals, (name, f, same), 0)) unexplained, median offset $(round(median(v); sigdigits = 3)) $unit"
                for ((name, f, same), v) in sort!(collect(offsets); by = p -> -length(last(p)))
        ]
        @warn "FringeClosure: detections the $kind solution does not explain:\n" * join(lines, "\n")
    end
    return nothing
end

# Each (station, feed, same-feed?) triple's count of measured detections.
function _detection_counts(dets)
    counts = Dict{Tuple{String, Int, Bool}, Int}()
    for d in dets, p in lookup(d, AntennaPair), f in lookup(d, FeedPair)
        d.valid[AntennaPair(At(p)), FeedPair(At(f))] || continue
        for (name, fd) in ((p[1], f[1]), (p[2], f[2]))
            key = (name, fd, f[1] == f[2])
            counts[key] = get(counts, key, 0) + 1
        end
    end
    return counts
end

# `c`'s model, once every component is one the closure fits on `geom`.
_vetted_model(c::FringeClosure, geom::DataGeometry) = _vet_model(
    c, c.model,
    "It fits a `Delay` or `Rate` on `Frequency = GlobalFrequency()` with a time " *
        "segmentation no finer than the data's scans: the search measures one value per " *
        "scan, so a segmentation that splits a scan (`PerIntegration`, or `TimeBlocks` " *
        "shorter than a scan) leaves columns unmeasured. It fits no dispersion (`Dispersion`) " *
        "or per-band-group delay (a `FreqGroups`-segmented `Delay`). See `FringeClosure`.",
    (; geom),
)

# A scan's detections as the station solve reads them: placed at the closure
# geometry's time index of the scan's first epoch, with the lever arms that set
# the delay and rate uncertainties.
function _closure_scan(d, geom::DataGeometry, has_rate::Bool)
    meta = metadata(d)
    (has_rate && length(meta.geom.times) < 2) && throw(
        ArgumentError(
            "FringeClosure: scan $(repr(meta.scan)) has one time, and a rate needs at least two. " *
                "Data averaged to one sample per scan cannot fit a `Rate`; search unaveraged " *
                "data, or use a model without a `Rate` term.",
        ),
    )
    ti = Calibration._time_index(geom, first(meta.geom.times))
    return Fring._with_ti(d, ti; meta.freq_mean, meta.freq_rms, meta.time_rms, meta.search.delay_window, meta.search.rate_window)
end

# A model of per-scan components without priors holds no column or prior row two
# scans share, so each scan's station systems are independent.
function _closes_per_scan(m::GainModel)
    trees = (m.phase, (e.phase for e in values(m.antennas) if haskey(e, :phase))...)
    per_scan(tc) = Calibration.component_is_per_scan(tc) && isnothing(tc.prior)
    return all(t -> all(per_scan, Calibration._flatten_components(t)), trees)
end

# A gauge sets what the detections leave free. One zero point per fringe group
# is expected; a second means the group's feeds are not linked, so the offset
# between them is the gauge's choice rather than a measurement.
function _warn_unlinked(unlinked)
    isempty(unlinked) && return nothing
    byscan = Dict{String, Vector{Symbol}}()
    for (kind, scan) in unlinked
        push!(get!(byscan, scan, Symbol[]), kind)
    end
    lines = ["scan $scan: $(join(sort!(unique(kinds)), ", "))" for (scan, kinds) in sort!(collect(byscan); by = first)]
    @warn "FringeClosure: no accepted detection relates different feeds of a fringe group, so the " *
        "offset between its feeds is not measured; the gauge sets it to zero at its preferred antenna. " *
        "Detections relating different feeds (`fringe_detections`) or `closure.pfa_max` decide this.\n" *
        join(lines, "\n")
    return nothing
end
