# ── solve_fringes: the end-to-end fringe-fitting pipeline ────────────────────
#
# Tie the tested building blocks together into one solve over a whole `UVSet`:
#
#   1. build a shared per-feed fringe `StationGainModel` (per-scan constant phase
#      + delay + rate, plus a per-integration adhoc-phase component);
#   2. for each (source, scan) group of sibling band leaves: concatenate bands
#      along frequency, fringe-search every (baseline, product), stationize to
#      per-(station, feed) delay/rate/phase, pack into θ;
#   3. divide out the Stage-B station gains, coherently frequency-average the
#      residual per (baseline, product, AP), solve the globally-closing adhoc
#      phase track, pack it into θ.
#
# The output is a `CalibrationSolution` that `apply_calibration` flattens the
# data with. Time units: the search/stationize layer works in SECONDS, so
# scan times (hours) and t0 (hours) are multiplied by 3600 on the way in; the
# Rate term stores its parameter in Hz, matching what `stationize_scan` returns.

import DimensionalData
using DimensionalData: lookup, dims, Ti
using ..UVData: Frequency
using Statistics: mean
using LinearAlgebra: BLAS

# Run `f` with BLAS pinned to a single thread, restoring the prior setting after.
# The fringe solve threads over scan groups with OhMyThreads; if BLAS also
# spawns threads, every task's WLS/QR solve fans out `BLAS.get_num_threads()`
# threads, so `ntasks × blas_threads` (e.g. 8 × 8 = 64) oversubscribe the cores
# and contend — threads sit "runnable" while only ~1 core makes progress. One
# BLAS thread per task is the correct split when the parallelism is across tasks.
function _with_single_blas_thread(f)
    old = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        return f()
    finally
        BLAS.set_num_threads(old)
    end
end

# Run `f` with FFTW using `n` threads per transform, restoring 1 (FFTW's default)
# after. The fringe SEARCH is ~96% FFT, but the memory cap pins group-parallelism
# (`ntasks`) well below the core count on a big file — leaving cores idle DURING
# the search. Giving each FFT `n = nthreads ÷ ntasks` threads uses them
# (`ntasks × n ≈ nthreads`), so the otherwise-idle cores accelerate the transforms
# instead of sitting out the bottleneck phase. Plans are built lazily inside the
# threaded passes, so this must wrap them to take effect.
function _with_fft_threads(f, n::Int)
    FFTW.set_num_threads(max(1, n))
    try
        return f()
    finally
        FFTW.set_num_threads(1)
    end
end

# Run `f` with the bulk reader decoding each leaf over `n` tasks, restoring 1
# after. Decode (byte-swap + complex repack + pol permute) is CPU-bound, so like
# the FFT it can use the cores the memory cap leaves idle. Nested under the group
# `tmap`, but Julia's scheduler caps live tasks at `nthreads`, so it composes.
function _with_decode_threads(f, n::Int)
    old = UVData._DECODE_NTASKS[]
    UVData._DECODE_NTASKS[] = max(1, n)
    try
        return f()
    finally
        UVData._DECODE_NTASKS[] = old
    end
end

# The shared fringe model. Phase, over the global frequency band:
#   - feed-COMMON per-scan constant + delay (`SharedFeeds`): the atmosphere/clock
#     terms that vary scan-to-scan and are the same for both polarization feeds;
#   - a GLOBAL R–L offset constant + delay (`GlobalTime × FeedComponent(2)`): the
#     instrumental feed-2−feed-1 offset, stable across the whole observation
#     (EHT-HOPS / rPICARD assumption). Solved once from all scans' cross hands so
#     the bright polarized scans pin it and weak scans inherit it (feeds that
#     would otherwise split into `ncomp = 2` are tied);
#   - per-scan rate (`SharedFeeds`): the fringe rate is common to both feeds, so it
#     is tied across them — exactly like the per-scan constant/delay. Solving it
#     `PerFeed` instead lets a spurious R–L rate (`rate₂ − rate₁`) float on noise;
#     since the Rate phase is `2π·rate·(t − t0_global)` with `t0` the WHOLE-TRACK
#     reference, that per-feed noise is multiplied by a per-scan lever arm of hours,
#     injecting a large, arbitrary, scan-to-scan R–L (RL/RR) phase jump. R–L rate is
#     negligible (EHT-HOPS), so tie it; a genuine offset would be a `GlobalTime ×
#     FeedComponent(2)` rate term (the analog of the R–L constant/delay), not PerFeed;
#   - a per-AP adhoc-phase constant (`SharedFeeds`): residual atmospheric phase is
#     non-birefringent (common to both feeds), so it is solved feed-common — which
#     denoises it and, crucially, contributes ZERO R–L phase. A `PerFeed` adhoc lets
#     per-AP solve noise differ between feeds and so injects spurious R–L (RL/RR)
#     scatter on top of the stable global instrumental R–L offset.
# Log-amplitude empty. `solve_station_systems!` reads the off1 columns this model
# declares — so the global-vs-per-scan split is a model choice, not solver code.
function _fringe_model(; dispersion::Bool = false, sbd_bands = nothing)
    # Ionospheric dispersion: per-scan, feed-common differential TEC (TECU),
    # phase = K·θ·(1/f0 − 1/f). NOT solved by the FFT search — measured by the
    # per-scan (Δτ, dTEC) band-phasor refinement (`_refine_scan_dispersion!`),
    # which jointly updates the per-scan delay (the two covary over any finite
    # band). `Dispersion` is its routing signature, like `PerChannel` for the
    # bandpass. Only included when the band layout can constrain it.
    disp = dispersion ?
        (TiedComponent(GainComponent(Dispersion(), PerScan(), GlobalFrequency()), SharedFeeds()),) : ()
    # Per-scan per-band-group single-band delay (fourfit's SBD): a station's
    # per-band signal path can move relative to its phase-cal tones between
    # scans (~30 ns on VR2505's YJ), which neither the wideband delay (one slope
    # across all groups) nor the time-invariant per-channel bandpass can track.
    # Measured from WITHIN-band chunk slopes by `_refine_scan_sbd!` — nearly
    # orthogonal to the cross-band observables that set the MBD delay and dTEC.
    # The Delay coordinate is (f − f0) with the GLOBAL f0, so correcting a group
    # slope about the group's own centre νg needs the companion per-group
    # constant −2πτ(νg − f0): net phase 2πτ(f − νg), zero at the group centre —
    # the cross-band solution is untouched. `FrequencyBands` is the routing
    # signature (excluded from stage-B).
    sbd = sbd_bands === nothing ? () : (
            TiedComponent(GainComponent(Delay(), PerScan(), FrequencyBands(sbd_bands)), SharedFeeds()),
            TiedComponent(GainComponent(ConstantTerm(), PerScan(), FrequencyBands(sbd_bands)), SharedFeeds()),
        )
    return StationGainModel(
        phase = (
            TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), SharedFeeds()),
            TiedComponent(GainComponent(ConstantTerm(), GlobalTime(), GlobalFrequency()), FeedComponent(2)),
            TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), SharedFeeds()),
            TiedComponent(GainComponent(Delay(), GlobalTime(), GlobalFrequency()), FeedComponent(2)),
            TiedComponent(GainComponent(Rate(), PerScan(), GlobalFrequency()), SharedFeeds()),
            disp...,
            sbd...,
            # Phase bandpass: per-channel, stable across the observation (HOPS-style),
            # per feed. Captures the residual nonlinear-in-frequency instrumental phase
            # that the per-scan (linear) delay cannot represent. Solved by a dedicated
            # frequency-stationization stage (`_solve_phase_bandpass!`), not by the
            # delay/rate search — `PerChannel` is its signature, used to route it.
            TiedComponent(GainComponent(PerChannel(), GlobalTime(), GlobalFrequency()), PerFeed()),
            TiedComponent(GainComponent(ConstantTerm(), PerIntegration(), GlobalFrequency()), SharedFeeds()),
        ),
        # Amplitude bandpass: per-channel, time-stable, per-feed log-amplitude — the
        # per-station instrumental amplitude shape (filterbank passband), FLATTENED
        # from the calibrator. Solved by the bandpass stage via a pluggable
        # `AbstractBandpassSmoother` (`_solve_amp_bandpass!`); the SNR gate drops
        # no-signal channels, which the smoother then estimates (or, for `FreeBandpass`,
        # leaves at gain 1). The absolute level stays the a-priori amplitude cal's job.
        logamp = (
            TiedComponent(GainComponent(PerChannel(), GlobalTime(), GlobalFrequency()), PerFeed()),
        ),
    )
end

# Stage-B engine components `(plan, kind)` — the delay/rate/const terms solved by
# the fringe search + stationization. EXCLUDES the per-integration adhoc term
# (`PerIntegration`, solved per-AP) and the phase bandpass (`PerChannel`, solved
# per-channel). `kind` is derived from the term type, so the routing is structural
# (by term/segmentation), not by hardcoded indices.
function _stageB_components(model, layout)
    comps = Tuple{ComponentPlan, Symbol}[]
    for (i, tc) in enumerate(model.phase)
        tc.component.time isa PerIntegration && continue   # adhoc (per-AP)
        tc.component.term isa PerChannel && continue        # bandpass (per-channel)
        tc.component.term isa Dispersion && continue        # dTEC (band-phasor refinement)
        tc.component.freq isa FrequencyBands && continue    # SBD (chunk-slope refinement)
        term = tc.component.term
        kind = term isa Delay ? :delay : term isa Rate ? :rate : :phase
        push!(comps, (layout.plans[i], kind))
    end
    return comps
end

# The dispersion component's plan (per-scan dTEC), or `nothing` when the model
# carries none; plus the per-scan feed-common Delay plan the refinement jointly
# updates (the delay↔dTEC covariance makes separate fits biased).
function _dispersion_plan(model, layout)
    i = findfirst(tc -> tc.component.term isa Dispersion, model.phase)
    return i === nothing ? nothing : layout.plans[i]
end

# Resolve the `dispersion` option (`:auto | true | false`). `:auto` enables the
# term only when the band layout can separate 1/ν from a linear delay: several
# sub-bands over a wide fractional bandwidth (VGOS 3–10.7 GHz qualifies; a
# single contiguous band cannot constrain the curvature and the term would just
# soak up delay).
function _dispersion_enabled(dispersion, geom::DataGeometry)
    dispersion === true && return true
    dispersion === false && return false
    dispersion === :auto || error("dispersion must be :auto, true or false (got $dispersion)")
    nb = length(unique(geom.spw_of_chan))
    fmin, fmax = extrema(geom.channel_freqs)
    return nb >= 4 && fmax / fmin > 1.3
end

function _perscan_delay_plan(model, layout)
    i = findfirst(
        tc -> tc.component.term isa Delay && !(tc.component.time isa GlobalTime) &&
            tc.component.freq isa GlobalFrequency,   # NOT the per-band-group SBD delay
        model.phase,
    )
    return i === nothing ? nothing : layout.plans[i]
end

# The SBD components' plans `(dplan, cplan, bands)` (per-scan per-band-group
# delay + companion constant), or `nothing` when the model carries none.
function _sbd_plans(model, layout)
    i = findfirst(
        tc -> tc.component.term isa Delay && tc.component.freq isa FrequencyBands,
        model.phase,
    )
    i === nothing && return nothing
    j = findfirst(
        tc -> tc.component.term isa ConstantTerm && tc.component.freq isa FrequencyBands,
        model.phase,
    )
    j === nothing && error("SBD delay component present without its companion constant")
    return (dplan = layout.plans[i], cplan = layout.plans[j], bands = model.phase[i].component.freq.ranges)
end

# Resolve the `sbd` option (`:auto | true | false`) into the band-group channel
# ranges, or `nothing` when disabled/unconstrainable. A single band group is
# fully degenerate with the stage-B wideband delay, so the term needs ≥ 2.
function _sbd_bands(sbd, geom::DataGeometry)
    sbd === false && return nothing
    sbd === true || sbd === :auto || error("sbd must be :auto, true or false (got $sbd)")
    bands = fringe_band_groups(geom.channel_freqs)
    return length(bands) >= 2 ? bands : nothing
end

# ant → representative-station map for co-located groups (separation below
# `max_sep` meters, e.g. the Onsala twins at ~75 m). Used to TIE the per-scan
# dTEC solve: co-located stations see the same ionosphere, so a differential
# TEC between them is pure solve error (VR2505 gave OE−OW = −2.4 TECU untied).
function _colocated_ties(antennas; max_sep::Real = 1000.0)
    n = length(antennas)
    xyz = antennas.station_xyz
    ties = collect(1:n)
    # Guard: missing/degenerate positions (synthetic tables often carry zeros)
    # would tie the whole array into one node; require a real VLBI-scale array
    # before trusting the positions at all.
    maxd2 = 0.0
    for j in 2:n, i in 1:(j - 1)
        maxd2 = max(maxd2, sum(abs2, Float64.(xyz[j]) .- Float64.(xyz[i])))
    end
    maxd2 > (10.0e3)^2 || return ties
    # NOTE the explicit nesting: in a comma-nested `for j, i` a `break` exits
    # BOTH levels — the old form stopped after the FIRST co-located pair in
    # the array (on VR2505 it tied Onsala OE-OW and silently never examined
    # the Wettzell twins, so WN-WS kept polluting the adhoc/bandpass solves
    # and the dTEC tie).
    for j in 2:n
        for i in 1:(j - 1)
            d2 = sum(abs2, Float64.(xyz[j]) .- Float64.(xyz[i]))
            if d2 <= float(max_sep)^2
                ties[j] = ties[i]
                break
            end
        end
    end
    return ties
end

# Baseline pairs joining co-located stations (same `_colocated_ties` group, e.g.
# the Onsala OE-OW and Wettzell WN-WS twins), both orders. These intra-site
# baselines carry enormous SNR but non-closing (crosstalk) frequency/time
# structure, so any solve that pools ALL baselines with ~snr² weights — the
# per-AP adhoc phasing and the phase/amp bandpass — is pulled to fit the
# crosstalk instead of the sky, splitting it into the two twins' gains and
# decohering their SKY baselines (on VR2505: OE-OW band-avg coherence 0.95 →
# 0.34 through the bandpass stages; WN's long baselines 1.00 → 0.6-0.7 through
# adhoc). Stage B is protected by the stationize closure screen; these stages
# pool raw residuals and need the exclusion up front. Empty when the array has
# no co-located pair (or positions are untrustworthy — see `_colocated_ties`).
function _colocated_pair_set(antennas; max_sep::Real = 1000.0)
    t = _colocated_ties(antennas; max_sep = max_sep)
    excl = Set{Tuple{Int, Int}}()
    for j in 2:length(t), i in 1:(j - 1)
        t[i] == t[j] || continue
        push!(excl, (i, j))
        push!(excl, (j, i))
    end
    return excl
end

# Index of the adhoc component (the per-integration phase term) within `model.phase`.
_adhoc_idx(model) = findfirst(tc -> tc.component.time isa PerIntegration, model.phase)

# The adhoc component's plan (the per-integration phase term).
_adhoc_plan(model, layout) = layout.plans[_adhoc_idx(model)]

# Whether the adhoc (per-integration) phase component is feed-common (`SharedFeeds`),
# so `solve_adhoc_phasing` solves one feed-common track and contributes zero R–L.
_adhoc_shared(model) = model.phase[_adhoc_idx(model)].tying isa SharedFeeds

# The phase-bandpass component's plan (the per-channel phase term), or `nothing`
# if the model carries no bandpass component.
function _bandpass_plan(model, layout)
    i = findfirst(tc -> tc.component.term isa PerChannel, model.phase)
    return i === nothing ? nothing : layout.plans[i]
end

# The amplitude-bandpass component's plan (the per-channel log-amp term), or
# `nothing`. Log-amp plans follow the phase plans in `layout.plans` (offset
# `nphase`), so index the `PerChannel` position within `model.logamp`.
function _amp_bandpass_plan(model, layout)
    j = findfirst(tc -> tc.component.term isa PerChannel, model.logamp)
    return j === nothing ? nothing : layout.plans[layout.nphase + j]
end

# One (source, scan) group of band leaves, already materialized, with the data
# concatenated along frequency.
struct _ScanGroup
    Vg::Array{ComplexF32, 4}             # (chan, ti, bl, pol) — native precision (halves RAM)
    Wg::Array{Float32, 4}
    fg::Vector{Float64}                  # channel freqs (Hz), all bands stacked
    tg::Vector{Float64}                  # times (hours)
    bl_pairs::Vector{Tuple{Int, Int}}
    pol_products::Vector{String}
    g_ci::Vector{Int}                    # global channel indices of fg
    g_ti::Vector{Int}                    # global time indices of tg
end

# Group leaf references by (source_name, scan_name) WITHOUT materializing — the
# leaves stay lazy so the caller can materialize one scan group at a time. A
# whole-dataset eager materialization here is what fills RAM on a real (24 GB)
# file: each band leaf is ~80 MB once promoted to ComplexF64, ×144 leaves ≈ 18 GB.
# Each group entry is a `(partition_key, lazy_leaf)` pair; the key is kept so the
# fused solve+reduce path can reassemble the output tree in the original layout.
function _scan_group_leaves(uvset::UVSet)
    groups = Dict{Tuple{String, String}, Vector{Any}}()
    order = Tuple{String, String}[]
    for (k, leaf) in UVData.branches(uvset)
        info = UVData.metadata(leaf)
        key = (info.source_name, info.scan_name)
        if !haskey(groups, key)
            groups[key] = Any[]
            push!(order, key)
        end
        push!(groups[key], (k, leaf))       # (partition key, lazy leaf reference)
    end
    return [groups[k] for k in order]
end

# Materialize ONE (source, scan) group and build its concatenated `_ScanGroup`.
# The solve reads only vis + weights (flag is redundant with w <= 0); skip the flag
# layer. `materialize_group` reads the scan's whole contiguous row span in ONE
# sequential read (FITS-IDI bulk path), extracting all bands — vs per-leaf, per-row
# seeks — then falls back to per-leaf for eager/non-IDI leaves.
_materialize_group_leaves(keyed_leaves_lazy) =
    UVData.materialize_group([l for (_, l) in keyed_leaves_lazy]; layers = (:vis, :weights, :uvw))

# Build the concatenated `_ScanGroup` (frequency stacked) for the fringe search /
# bandpass stage. Fast path: decode each band's vis+weights DIRECTLY into a
# contiguous channel-block of the stacked cube (`_try_build_scan_group_direct`),
# avoiding the per-band intermediate dense arrays + the explicit concat copy (one
# full in-RAM data copy and one read+write pass eliminated) and the unused uvw
# read. Falls back to materialize-then-copy when the group is not a single
# sibling-band IDI span or its channels do not stack contiguously.
function _materialize_concat_group(keyed_leaves_lazy, geom::DataGeometry)
    direct = _try_build_scan_group_direct(keyed_leaves_lazy, geom)
    direct === nothing || return direct
    leaves = _materialize_group_leaves(keyed_leaves_lazy)
    return _build_scan_group(leaves, geom)
end

# Try to build the stacked `_ScanGroup` by decoding straight into the cube. Returns
# `nothing` (caller falls back) unless every band leaf maps to ONE full contiguous
# ascending channel block of the stacked frequency axis — the normal case (distinct,
# non-interleaved sub-bands). Metadata (baselines/pols/times/freqs) comes from the
# LAZY leaves (dims are eager), so nothing is materialized until the decode.
function _try_build_scan_group_direct(keyed_leaves_lazy, geom::DataGeometry)
    lazy = [l for (_, l) in keyed_leaves_lazy]
    l0 = first(lazy)
    bl_pairs = collect(UVData.baselines(l0).pairs)
    pols = String.(pol_products(l0))
    tg = Float64.(lookup(l0[:vis], Ti))
    nti = length(tg)
    nbl = length(bl_pairs)
    npol = length(pols)

    # (global channel index, freq, leafidx, local channel) for every channel, sorted
    # by global index — the stacked-cube channel order (same as `_build_scan_group`).
    chan_entries = Tuple{Int, Float64, Int, Int}[]
    nchan_leaf = Vector{Int}(undef, length(lazy))
    for (li, leaf) in enumerate(lazy)
        ci, _ = leaf_window(geom, leaf)
        fs = Float64.(lookup(leaf[:vis], Frequency))
        nchan_leaf[li] = length(ci)
        for (lc, gc) in enumerate(ci)
            push!(chan_entries, (gc, fs[lc], li, lc))
        end
    end
    sort!(chan_entries; by = e -> e[1])
    nchan = length(chan_entries)

    # Require each leaf to be exactly one full contiguous ascending run (local
    # channels 1..no_chan land on a consecutive cube block). Anything else (a band
    # split or interleaved with another) → bail to the general copy path.
    blocks = Vector{UnitRange{Int}}(undef, length(lazy))
    fill!(blocks, 1:0)
    i = 1
    while i <= nchan
        li = chan_entries[i][3]
        chan_entries[i][4] == 1 || return nothing          # run must start at local channel 1
        j = i
        while j < nchan && chan_entries[j + 1][3] == li &&
                chan_entries[j + 1][4] == chan_entries[j][4] + 1
            j += 1
        end
        (j - i + 1) == nchan_leaf[li] || return nothing     # run must cover the whole band
        isempty(blocks[li]) || return nothing               # each leaf exactly once
        blocks[li] = i:j
        i = j + 1
    end
    any(isempty, blocks) && return nothing

    Vg = Array{ComplexF32}(undef, nchan, nti, nbl, npol)
    Wg = Array{Float32}(undef, nchan, nti, nbl, npol)
    fg = Vector{Float64}(undef, nchan)
    g_ci = Vector{Int}(undef, nchan)
    for (row, e) in enumerate(chan_entries)
        fg[row] = e[2]
        g_ci[row] = e[1]
    end

    dests = [
        (view(Vg, blocks[li], :, :, :), view(Wg, blocks[li], :, :, :))
            for li in eachindex(lazy)
    ]
    UVData.materialize_group_into!(dests, lazy; layers = (:vis, :weights)) || return nothing

    _, g_ti = leaf_window(geom, l0)
    return _ScanGroup(Vg, Wg, fg, tg, bl_pairs, pols, g_ci, g_ti)
end

# Materialize a group as its per-band `(key, leaf)` pairs WITHOUT building the
# concatenated copy — used by pass 2 (adhoc runs directly on the leaves, then they
# are corrected/reduced/reassembled in place). Holds a single data copy.
function _materialize_leaf_group(keyed_leaves_lazy)
    leaves = _materialize_group_leaves(keyed_leaves_lazy)
    return [(k, m) for ((k, _), m) in zip(keyed_leaves_lazy, leaves)]
end

# ── Pre-calibration (e.g. phase-cal tones) applied at materialization ──────────
#
# A `precal` CalibrationSolution (e.g. from `phasecal_solution`) is divided out
# of every scan group the moment it is materialized — the streaming equivalent
# of `apply_calibration(uvset, precal)` on a set too large to hold. `mask` marks
# GLOBAL channels to zero-weight (e.g. `tone_channel_mask`). Both passes and the
# bandpass stage go through the same two choke points below, so the whole solve
# consistently sees pre-calibrated data; the returned fringe solution is then
# the correction ON TOP of `precal` (total gain = precal ∘ solution).
# The carrier is a NamedTuple `(; ev, θ, mask)` (`ev = nothing` for flag-only),
# or `nothing` when neither a precal nor a mask is given.
function _make_precal(precal, flag_channels, geom::DataGeometry)
    (precal === nothing && flag_channels === nothing) && return nothing
    ev = nothing
    θ = Float64[]
    if precal !== nothing
        precal.geom.channel_freqs == geom.channel_freqs && precal.geom.times == geom.times ||
            error("precal: solution geometry does not match this uvset (build it from the same set)")
        ev = GainEvaluator(precal.model, precal.layout)
        θ = precal.θ
    end
    mask = flag_channels === nothing ? nothing : BitVector(flag_channels)
    mask === nothing || length(mask) == length(geom.channel_freqs) ||
        error("flag_channels: mask length $(length(mask)) ≠ nchan $(length(geom.channel_freqs))")
    return (; ev, θ, mask)
end

# A precal's gains are usually CONSTANT IN TIME across one scan's window (the
# phase-cal model is PerScan × PerSpectralWindow with no time-coordinate term),
# so evaluating the full (nchan × nti) gain cube wastes nti× the `cis` work —
# the dominant cost of applying the correction. True when no component reads a
# time coordinate and every component sees a single time segment in the window.
function _precal_time_constant(ev::GainEvaluator, g_ti)
    length(g_ti) <= 1 && return true
    for plan in ev.layout.plans
        plan.coord === Calibration.COORD_TIME && return false
        ts = plan.tseg_id[g_ti[1]]
        for gti in g_ti
            plan.tseg_id[gti] == ts || return false
        end
    end
    return true
end

# Divide the precal gains out of a concatenated scan group in place (and apply
# the channel mask). Weights transform as w → w·|g_a g_b|² (Var(V/g) = σ²/|g|²);
# for the phase-only phase-cal solution |g| = 1 and only phases move. The
# per-baseline slices are independent, so the division fans out over `inner`
# tasks (the cube is ~4 GiB on big VGOS scans — one thread is a real cost).
function _apply_precal!(grp::_ScanGroup, pc, inner::Integer = 1)
    pc === nothing && return grp
    if pc.ev !== nothing
        tconst = _precal_time_constant(pc.ev, grp.g_ti)
        g = tconst ? evaluate_gains(pc.ev, pc.θ, grp.g_ci, grp.g_ti[1]:grp.g_ti[1]) :
            evaluate_gains(pc.ev, pc.θ, grp.g_ci, grp.g_ti)      # (nchan, nti|1, nant, 2)
        nchan, nti, nbl, npol = size(grp.Vg)
        pairs = [(bi, p) for p in 1:npol for bi in 1:nbl]
        tasks = map(Iterators.partition(pairs, cld(length(pairs), clamp(Int(inner), 1, length(pairs))))) do chunk
            Threads.@spawn @inbounds for (bi, p) in chunk
                fa, fb = correlation_feed_pair(grp.pol_products[p])
                a, b = grp.bl_pairs[bi]
                for ti in 1:nti
                    gt = tconst ? 1 : ti
                    for c in 1:nchan
                        den = g[c, gt, a, fa] * conj(g[c, gt, b, fb])
                        (isfinite(den) && abs2(den) > 0) || continue
                        grp.Vg[c, ti, bi, p] /= den
                        grp.Wg[c, ti, bi, p] *= abs2(den)
                    end
                end
            end
        end
        foreach(wait, tasks)
    end
    if pc.mask !== nothing
        @inbounds for (c, gc) in enumerate(grp.g_ci)
            pc.mask[gc] && (grp.Wg[c, :, :, :] .= 0)
        end
    end
    return grp
end

# Divide the precal gains out of one band leaf's `V`/`W` arrays IN PLACE (and apply
# the channel mask). Shared core of the two leaf wrappers below; `V`/`W` are
# (nchan, nti, nbl, npol). `ci`/`ti` are the leaf's global channel/time windows.
function _divide_precal!(V, W, pc, ci, ti, bl_pairs, pols)
    nchan, nti, nbl, npol = size(V)
    if pc.ev !== nothing
        tconst = _precal_time_constant(pc.ev, ti)
        g = tconst ? evaluate_gains(pc.ev, pc.θ, ci, ti[1]:ti[1]) : evaluate_gains(pc.ev, pc.θ, ci, ti)
        @inbounds for p in 1:npol
            fa, fb = correlation_feed_pair(pols[p])
            for bi in 1:nbl
                a, b = bl_pairs[bi]
                for t in 1:nti
                    gt = tconst ? 1 : t
                    for c in 1:nchan
                        den = g[c, gt, a, fa] * conj(g[c, gt, b, fb])
                        (isfinite(den) && abs2(den) > 0) || continue
                        V[c, t, bi, p] /= den
                        W[c, t, bi, p] *= abs2(den)
                    end
                end
            end
        end
    end
    if pc.mask !== nothing
        @inbounds for (c, gc) in enumerate(ci)
            pc.mask[gc] && (W[c, :, :, :] .= 0)
        end
    end
    return nothing
end

# Same, for one materialized band leaf (pass 2 works leaf-wise). NON-mutating:
# an eager input's "materialized" leaf is the caller's own leaf, so dividing in
# place would corrupt the user's uvset — copy + rebuild instead (the same
# pattern as `apply_calibration`; one transient leaf copy). Use `_precal_leaf!`
# when the leaf's arrays are known-private (freshly materialized from a lazy
# source) — it skips this ~scan-sized copy.
function _precal_leaf(leaf, pc, geom::DataGeometry)
    pc === nothing && return leaf
    ci, ti = leaf_window(geom, leaf)
    V = copy(parent(leaf[:vis]))
    W = copy(parent(leaf[:weights]))
    _divide_precal!(V, W, pc, ci, ti, collect(UVData.baselines(leaf).pairs), String.(pol_products(leaf)))
    return with_visibilities(leaf, V, W)
end

# In-place variant for leaves whose backing arrays are PRIVATE (the streaming
# solve materializes each lazy leaf into fresh arrays before this runs, so there
# is no user-owned data to protect). Divides the precal out of the leaf's own
# `vis`/`weights` and returns the same leaf — no defensive copy, no rebuild. On a
# big VGOS scan the copy `_precal_leaf` makes is ~4 GiB of alloc + memcpy, so this
# is the pass-2 fast path (see `_materialize_leaf_group`).
function _precal_leaf!(leaf, pc, geom::DataGeometry)
    pc === nothing && return leaf
    ci, ti = leaf_window(geom, leaf)
    _divide_precal!(
        parent(leaf[:vis]), parent(leaf[:weights]), pc, ci, ti,
        collect(UVData.baselines(leaf).pairs), String.(pol_products(leaf)),
    )
    return leaf
end

# Precal-aware materialization wrappers — the solve paths call these; the plain
# 2-arg versions stay precal-free for external/diagnostic use. (The concat cube
# is always a private copy, so the group version mutates it freely.) The leaf
# version fans the per-leaf correction out over `inner` tasks (band leaves are
# independent).
function _materialize_concat_group(keyed_leaves_lazy, geom::DataGeometry, pc, inner::Integer = 1)
    grp = _materialize_concat_group(keyed_leaves_lazy, geom)
    return _apply_precal!(grp, pc, inner)
end

function _materialize_leaf_group(keyed_leaves_lazy, geom::DataGeometry, pc, inner::Integer = 1)
    keyed = _materialize_leaf_group(keyed_leaves_lazy)
    pc === nothing && return keyed
    # When every source leaf was lazy, `keyed`'s arrays are freshly materialized
    # (private), so precal can divide in place — no ~scan-sized defensive copy.
    # An eager source hands back the caller's own leaf, so fall back to the copy.
    private = all(((_, l),) -> UVData.is_lazy(l), keyed_leaves_lazy)
    apply = private ? _precal_leaf! : _precal_leaf
    out = Vector{Any}(undef, length(keyed))
    tasks = map(Iterators.partition(eachindex(keyed), cld(length(keyed), clamp(Int(inner), 1, length(keyed))))) do chunk
        Threads.@spawn for i in chunk
            k, m = keyed[i]
            out[i] = (k, apply(m, pc, geom))
        end
    end
    foreach(wait, tasks)
    return [out[i]::Tuple for i in eachindex(out)]
end

# Copy a contiguous channel-block from one band leaf's vis/weights (`V`/`W`) into
# the concatenated cubes at destination channel offset `dst0`. A FUNCTION BARRIER:
# `V`/`W` come from `parent(leaf[:vis])`, which is type-unstable at the call site,
# so doing the scalar copy inline made every element a dynamic dispatch (~1 MB/s).
# Passing them as arguments forces Julia to specialize this loop on their concrete
# runtime types; the `@simd` inner loop over the leading (stride-1) channel axis
# then runs at native speed.
function _concat_block!(
        Vg::Array{ComplexF32, 4}, Wg::Array{Float32, 4}, V, W,
        dst0::Int, lc0::Int, nbc::Int,
    )
    _, nti, nbl, npol = size(Vg)
    @inbounds for p in 1:npol, bl in 1:nbl, ti in 1:nti
        @simd for c in 0:(nbc - 1)
            Vg[dst0 + c, ti, bl, p] = V[lc0 + c, ti, bl, p]
            Wg[dst0 + c, ti, bl, p] = W[lc0 + c, ti, bl, p]
        end
    end
    return nothing
end

# Build the concatenated `_ScanGroup` from a group's already-materialized sibling
# band leaves: stack the frequency axis (sorted by channel frequency) and resolve
# global geom indices.
function _build_scan_group(leaves, geom::DataGeometry)
    let
        # Reference baselines/pols/times from the first band leaf.
        l0 = first(leaves)
        bl_pairs = collect(UVData.baselines(l0).pairs)
        pols = String.(pol_products(l0))
        tg = Float64.(lookup(l0[:vis], Ti))
        nti = length(tg)
        nbl = length(bl_pairs)
        npol = length(pols)

        # Collect (global channel index, freq, source leaf, local channel) for
        # every channel of every band, then sort by global channel index so the
        # concatenated frequency axis matches geom order.
        chan_entries = Tuple{Int, Float64, Int, Int}[]   # (g_ci, freq, leafidx, local_c)
        for (li, leaf) in enumerate(leaves)
            ci, _ = leaf_window(geom, leaf)
            fs = Float64.(lookup(leaf[:vis], Frequency))
            for (lc, gc) in enumerate(ci)
                push!(chan_entries, (gc, fs[lc], li, lc))
            end
        end
        sort!(chan_entries; by = e -> e[1])
        nchan = length(chan_entries)

        Vg = Array{ComplexF32}(undef, nchan, nti, nbl, npol)
        Wg = Array{Float32}(undef, nchan, nti, nbl, npol)
        fg = Vector{Float64}(undef, nchan)
        g_ci = Vector{Int}(undef, nchan)
        for (row, e) in enumerate(chan_entries)
            fg[row] = e[2]
            g_ci[row] = e[1]
        end

        # Cache-friendly concat: walk maximal runs of consecutive channels coming
        # from the same band leaf and copy each as a contiguous channel-block
        # (channel is the leading, stride-1 axis of both V and Vg). The old
        # `Vg[row, :, :, :] .= V[lc, :, :, :]` fixed the *leading* axis, scattering
        # single elements at stride `nchan` across the whole cube one channel-row at
        # a time (nchan times) — cache-hostile and ~5× slower than this block copy.
        i = 1
        @inbounds while i <= nchan
            li = chan_entries[i][3]
            lc0 = chan_entries[i][4]
            j = i
            while j < nchan && chan_entries[j + 1][3] == li &&
                    chan_entries[j + 1][4] == chan_entries[j][4] + 1
                j += 1
            end
            nbc = j - i + 1
            V = parent(leaves[li][:vis])
            W = parent(leaves[li][:weights])
            _concat_block!(Vg, Wg, V, W, i, lc0, nbc)
            i = j + 1
        end

        _, g_ti = leaf_window(geom, l0)
        return _ScanGroup(Vg, Wg, fg, tg, bl_pairs, pols, g_ci, g_ti)
    end
end

# One recorded fringe detection: baseline antennas, correlation product, SNR, and
# the PER-BASELINE false-alarm probability (single-search null — the right number
# for judging that one detection on its own).
const _DetRow = @NamedTuple{a::Int, b::Int, pol::String, snr::Float64, pfa::Float64}

# ── Solve progress reporting ───────────────────────────────────────────────────
#
# `progress` is `nothing` or a callback `(stage::Symbol, done::Int, total::Int)`.
# Each pass announces itself with `(stage, 0, total)` and then reports every
# completed scan group; stages are `:search` (× rounds), `:bandpass` (calibrator
# scans), and `:adhoc` (pass 2, which in `solve_and_reduce_fringes` includes the
# correct+reduce work). Completions arrive from worker tasks, so the callback is
# invoked under a lock — it may do I/O but should be quick. Callback errors are
# warned once and otherwise ignored (progress must never kill a solve).
const _PROGRESS_LOCK = ReentrantLock()

_progress_notify(::Nothing, stage, done, total) = nothing
function _progress_notify(cb, stage, done, total)
    lock(_PROGRESS_LOCK) do
        try
            cb(stage, Int(done), Int(total))
        catch err
            @warn "progress callback failed" stage err maxlog = 1
        end
    end
    return nothing
end

# Reusable pool of search workspaces, shared across the whole solve so FFTW
# plans survive scan-to-scan (a task-local value would rebuild a fresh workspace
# — and re-plan — for every group's short-lived inner tasks). Capacity bounds
# the number of live workspaces regardless of how many tasks contend.
_ws_pool(n::Integer) = begin
    ch = Channel{FringeWorkspace}(max(Int(n), 1))
    for _ in 1:max(Int(n), 1)
        put!(ch, FringeWorkspace())
    end
    ch
end

# Fringe-search every (baseline, product) of a materialized group on `Vsearch`
# (the raw cube, or the residual on rounds > 1). The searches are independent, so
# they run on `inner` concurrent tasks (each borrowing a workspace from `pool` —
# the MBD workspaces are a few MB, so per-task scratch is cheap; results are
# BIT-IDENTICAL to the serial loop since every search only reads shared state).
# Returns the detection matrix, the group's max valid SNR, the scan's effective
# number of independent search cells (per-search cells × number of
# cross-baseline×product searches — the null for the scan-level max SNR, consumed
# by `fringe_pfa`), and the `_DetRow` list of the VALID detections — the ones the
# stage-B solve consumes — so false-fringe screening (`suspect_fringes`) needs no
# second read of the data.
function _search_group(grp::_ScanGroup, Vsearch, f0, t0_sec, search, pool::Channel{FringeWorkspace}, inner::Integer = 1)
    nbl = length(grp.bl_pairs)
    npol = length(grp.pol_products)
    det = Matrix{FringeDetection}(undef, nbl, npol)
    # Build the search-grid geometry ONCE for the group (freqs/times are shared by
    # every baseline×product), and pass `view`s of the cube into the core so each
    # call neither re-sorts the axes nor copies its (chan, ti) slice.
    times = grp.tg .* 3600.0
    ax = _search_axes(grp.fg, times, search)
    ncross = count(pr -> pr[1] != pr[2], grp.bl_pairs)
    cells1 = _search_cells(ax, search)
    ncells = cells1 * max(ncross * npol, 1)

    pairs = [(bi, p) for p in 1:npol for bi in 1:nbl]
    nchunk = clamp(Int(inner), 1, length(pairs))
    tasks = map(Iterators.partition(pairs, cld(length(pairs), nchunk))) do chunk
        Threads.@spawn begin
            ws = take!(pool)
            try
                for (bi, p) in chunk
                    a, b = grp.bl_pairs[bi]
                    if a == b
                        det[bi, p] = FringeDetection(0.0, 0.0, 0.0, 0.0, 0.0, false)
                        continue
                    end
                    det[bi, p] = _baseline_fringe_search(
                        view(Vsearch, :, :, bi, p), view(grp.Wg, :, :, bi, p),
                        grp.fg, times, f0, t0_sec, ax, ws, search,
                    )
                end
            finally
                put!(pool, ws)
            end
        end
    end
    foreach(wait, tasks)

    # Assemble the scalar outputs SEQUENTIALLY in the original (product-major)
    # order so the recorded detection table is identical to the serial loop's.
    maxsnr = 0.0
    rows = _DetRow[]
    for p in 1:npol, bi in 1:nbl
        d = det[bi, p]
        d.valid || continue
        maxsnr = max(maxsnr, d.snr)
        push!(rows, (; a = grp.bl_pairs[bi][1], b = grp.bl_pairs[bi][2], pol = grp.pol_products[p], snr = d.snr, pfa = fringe_pfa(d.snr, cells1)))
    end
    return det, maxsnr, ncells, rows
end

# Adhoc for one group from its per-band LEAVES (no concatenated `_ScanGroup` — pass
# 2's memory-lean path). The per-AP residual is summed over each band leaf's
# channels (a function barrier per leaf, gains evaluated on the leaf window); the
# sum over all bands' channels is identical to accumulating over the concatenated
# cube, so the result matches the concatenated-cube accumulation exactly. `keyed` is the group's
# `(key, leaf)` pairs.
function _adhoc_group_leaves!(θ, keyed, geom::DataGeometry, ev, adhoc_plan, adhoc, ref_ant, nant; shared_feeds::Bool = false, inner::Integer = 1, excl = nothing, psI = nothing)
    leaves = [m for (_, m) in keyed]
    l0 = first(leaves)
    bl_pairs = collect(UVData.baselines(l0).pairs)
    pols = String.(pol_products(l0))
    tg = Float64.(lookup(l0[:vis], Ti))
    _, g_ti = leaf_window(geom, l0)
    nbl = length(bl_pairs); npol = length(pols); nap = length(tg)
    # Per-band-leaf accumulation on `inner` tasks with chunk-local partials
    # (~1 MB each), summed in chunk order — the leaves are independent and this
    # loop (gain evaluation + residual sum over every visibility) dominates the
    # adhoc pass on many-band data.
    chunks = collect(Iterators.partition(leaves, cld(length(leaves), clamp(Int(inner), 1, length(leaves)))))
    parts = map(chunks) do chunk
        Threads.@spawn begin
            rl = zeros(ComplexF64, nbl, npol, nap)
            wl = zeros(Float64, nbl, npol, nap)
            for leaf in chunk
                ci, ti = leaf_window(geom, leaf)
                g = evaluate_gains(ev, θ, ci, ti)       # (nchan_leaf, nti, nant, 2)
                _accumulate_leaf_rbar!(rl, wl, parent(leaf[:vis]), parent(leaf[:weights]), g, bl_pairs, pols)
            end
            (rl, wl)
        end
    end
    rbar = zeros(ComplexF64, nbl, npol, nap)
    wbar = zeros(Float64, nbl, npol, nap)
    for t in parts
        rl, wl = fetch(t)
        rbar .+= rl
        wbar .+= wl
    end
    # Drop co-located (intra-site) baselines from the per-AP solve — see
    # `_colocated_pair_set`. Zero weight ⇒ `_adhoc_ap_rows` skips the rows.
    if excl !== nothing
        for bi in eachindex(bl_pairs)
            bl_pairs[bi] in excl || continue
            fill!(view(rbar, bi, :, :), zero(ComplexF64))
            fill!(view(wbar, bi, :, :), 0.0)
        end
    end
    # Linear-feed data: collapse the four products to one pseudo-Stokes-I row
    # per (baseline, AP) using the field-rotation coefficients — see
    # `_pseudo_stokes_collapse!` (rotation-robust; single products can null).
    if psI !== nothing
        m0 = UVData.metadata(l0)
        jds = [psI.base_jd + Float64(t) / 24.0 for t in tg]
        ψ = _field_rotation_angles(m0.antennas, m0.ra, m0.dec, jds)
        _pseudo_stokes_collapse!(rbar, wbar, bl_pairs, pols, ψ)
    end
    # `tg` is in hours; pass SECONDS so the adhoc's `:auto` window (T_AP / T_coh) is
    # in physical units. Detrend uses only the mean, so the scaling is otherwise inert.
    as = solve_adhoc_phasing(rbar, wbar, bl_pairs, pols, nant, tg .* 3600.0; ref_ant = ref_ant, smoother = adhoc, shared_feeds = shared_feeds)
    for (ap, gti) in enumerate(g_ti)
        tseg = adhoc_plan.tseg_id[gti]
        for ant in 1:nant, feed in 1:2
            v = as.phase[ant, feed, ap]
            isfinite(v) || continue
            off = adhoc_plan.off1[ant, feed, tseg, 1]
            off == 0 && continue
            θ[off] = v
        end
    end
    return θ
end

# Accumulate one band leaf's per-AP residual into `rbar`/`wbar` as the
# inverse-variance mean of the CORRECTED data: `V/den` has variance 1/(w·|den|²)
# (Var(V) = 1/w), so its weight is `w·|den|²` — the same reweighting
# `apply_calibration` applies. Accumulating with the RAW `w` instead is only
# correct for |den| = 1 (phase-only gains); once the log-amp bandpass is in θ
# it UP-weights exactly the channels the amp solution marked low-|g| —
# amplitude-inflated noise dominating the per-AP phasor (on VR2505 this tripled
# K2's adhoc track noise, since its low band's |g| dip encodes its own phase
# scramble). Function barrier: `V`/`W` from `parent(leaf[...])` are
# type-unstable at the call site.
function _accumulate_leaf_rbar!(rbar, wbar, V, W, g, bl_pairs, pols)
    nchan, nti, nbl, npol = size(V)
    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(pols[p])
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            for tt in 1:nti, c in 1:nchan
                w = W[c, tt, bi, p]
                (w > 0 && isfinite(w)) || continue
                ga = g[c, tt, a, fa]
                gb = g[c, tt, b, fb]
                den = ga * conj(gb)
                (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(den)) || continue
                v = V[c, tt, bi, p] / den
                isfinite(v) || continue
                wd = w * abs2(den)
                rbar[bi, p, tt] += wd * v
                wbar[bi, p, tt] += wd
            end
        end
    end
    return rbar, wbar
end

# ── Per-scan (Δτ, dTEC) band-phasor refinement ────────────────────────────────
#
# The FFT search + stationization solve a per-scan LINEAR delay; the ionosphere
# adds a dispersive phase K·dTEC·(1/f0 − 1/f) whose linear-in-ν part the delay
# absorbs (biasing it by hundreds of ps on VGOS) and whose curvature survives as
# cross-band structure that no GLOBAL per-channel bandpass can track scan-to-scan.
# This stage measures both self-consistently from each scan's own residual —
# fourfit's ionospheric search, done as a per-baseline (Δτ, dTEC) grid fit over
# the scan's band phasors, then stationized through `solve_station_systems!`
# (closure screen + robust rejection included) into the per-scan delay and
# dispersion θ columns. Runs inside pass 2 on the already-materialized leaves, so
# it costs no extra read; the adhoc solve then sees dispersion-corrected
# residuals. Feed-common (ionosphere is non-birefringent to first order); cross
# hands are skipped like the rate solve. It ALSO runs inside the bandpass stage
# on each accumulated calibrator scan (see `_solve_bandpass_stage!`), so the
# frozen phase bandpass is solved from dispersion-corrected residuals; each
# scan's pass-2 refinement then measures the residual dTEC against that CLEAN
# curve (θ increments compose — for the accumulated scans pass 2 is a small
# polish on top of the bandpass-stage fit).

# Collapse one band leaf to one residual phasor per (baseline, product):
# `z[bi, p] = Σ w·|den|²·(V/den)`, `w[bi, p] = Σ w·|den|²` over the leaf's
# channels × APs — the inverse-variance mean of the corrected data (see
# `_accumulate_leaf_rbar!` for why the |den|² reweighting is required once
# amp gains live in θ). Function barrier (V/W type-unstable at the call site).
function _accumulate_leaf_band_phasor!(z, w, V, W, g, bl_pairs, pols)
    nchan, nti, nbl, npol = size(V)
    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(pols[p])
        fa == fb || continue                    # parallel hands only
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            a == b && continue
            acc = zero(ComplexF64)
            wsum = 0.0
            for tt in 1:nti, c in 1:nchan
                ww = W[c, tt, bi, p]
                (ww > 0 && isfinite(ww)) || continue
                ga = g[c, tt, a, fa]
                gb = g[c, tt, b, fb]
                den = ga * conj(gb)
                (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(den)) || continue
                v = V[c, tt, bi, p] / den
                isfinite(v) || continue
                wd = ww * abs2(den)
                acc += wd * v
                wsum += wd
            end
            z[bi, p] = acc
            w[bi, p] = wsum
        end
    end
    return nothing
end

# Joint (Δτ, dTEC) fit of one baseline's band phasors: coarse-to-fine grid
# maximization of |Σ_b z_b·cis(−2πτ(f_b−f0) − K·dt·(1/f0−1/f_b))| — the exact
# matched filter over the two smooth terms (phases are wrapped, so no linear
# fit applies). SNR is the debiased coherent amplitude over √Var, Var(Σz) = Σw
# for inverse-variance weights.
function _fit_band_dispersion(
        fbs::Vector{Float64}, zs::Vector{ComplexF64}, ws::Vector{Float64}, f0::Float64;
        tau_max::Float64 = 2.0e-8, dtec_max::Float64 = 45.0,
    )
    nb = length(fbs)
    xdisp = [Calibration.DISPERSION_K * (1.0 / f0 - 1.0 / fbs[b]) for b in 1:nb]
    xtau = [2π * (fbs[b] - f0) for b in 1:nb]
    best_a = -1.0
    best_d = 0.0
    best_t = 0.0
    y = Vector{ComplexF64}(undef, nb)
    function sweep(dts, τs)
        for dt in dts
            @inbounds for b in 1:nb
                y[b] = zs[b] * cis(-dt * xdisp[b])
            end
            for τ in τs
                acc = zero(ComplexF64)
                @inbounds for b in 1:nb
                    acc += y[b] * cis(-τ * xtau[b])
                end
                a = abs(acc)
                if a > best_a
                    best_a = a
                    best_d = dt
                    best_t = τ
                end
            end
        end
        return nothing
    end
    sweep(-dtec_max:0.25:dtec_max, -tau_max:2.0e-11:tau_max)
    sweep((best_d - 0.3):0.01:(best_d + 0.3), (best_t - 3.0e-11):1.0e-12:(best_t + 3.0e-11))
    # Parabolic sub-grid polish (one axis at a time): at high SNR the CRB is far
    # below the fine-grid step, and leftover quantization would read as a
    # significant residual to the downstream station solve.
    value(dt, τ) = abs(sum(zs[b] * cis(-dt * xdisp[b] - τ * xtau[b]) for b in 1:nb))
    for _ in 1:2
        for (step, isdt) in ((0.01, true), (1.0e-12, false))
            d0 = best_d
            t0 = best_t
            am = isdt ? value(d0 - step, t0) : value(d0, t0 - step)
            ap = isdt ? value(d0 + step, t0) : value(d0, t0 + step)
            den = am - 2 * best_a + ap
            den < 0 || continue
            δ = 0.5 * step * (am - ap) / den
            abs(δ) <= step || continue
            if isdt
                best_d = d0 + δ
            else
                best_t = t0 + δ
            end
            best_a = value(best_d, best_t)
        end
    end
    var = sum(ws)
    snr = var > 0 ? sqrt(max(best_a^2 - var, 0.0) / var) : 0.0
    return (tau = best_t, dtec = best_d, amp = best_a, snr = snr)
end

# Narrow delay window for the pass-2 POLISH of a scan the bandpass stage already
# fit (`reuse_bandpass_refine`): the residual on top of the stage's fit is small,
# so a tight window keeps the fine-grid resolution/accuracy while shrinking the
# dominant coarse sweep (its cost scales with window extent). The dTEC half-width
# is the user-tunable `bandpass_polish_dtec` (a too-tight window CLIPS noisy weak-
# scan residuals → the coherence regression a full skip caused; default ±20 TECU
# covers the worst observed). The delay window stays fixed here — well within the
# band-comb ambiguity clamp (`_band_delay_halfwindow`).
const _DTEC_POLISH_TAU = 8.0e-9      # s

# Refine one materialized scan group: band phasors → per-baseline (Δτ, dTEC) →
# two station solves accumulating into the per-scan delay and dispersion columns
# (disjoint per scan, so pass-2 groups can refine concurrently). Returns the
# number of detections the robust station solves excised.
function _refine_scan_dispersion!(
        θ, keyed, geom::DataGeometry, ev, delay_plan, disp_plan, ref_ant, nant;
        # Closure screen ON, post-fit MAD cut OFF (`reject_iters = 0`): the screen
        # is what kills junk baselines (co-located crosstalk), while the MAD cut is
        # calibrated for noisy FFT detections — the exact-filter fits here are
        # quantization-limited at high SNR and homogeneous cuts over-reject.
        # The τ window spans the FULL unique comb branch (clamped to half the
        # band-comb ambiguity by `_band_delay_halfwindow`, ±15.6 ns on VGOS): a
        # station whose stage-B delay was excised by the closure screen (strong
        # dispersion tie-breaks the comb degeneracy differently per baseline —
        # K2 at +22 TECU) carries its FULL delay into this fit, and the old
        # ±1.5 ns window could never recover it. `dtec_max` likewise covers the
        # worst observed station pairings (±22 vs −6 TECU).
        opts::Stationization = Stationization(reject_iters = 0), snr_min::Real = 8.0,
        tau_max::Real = 2.0e-8, dtec_max::Real = 45.0, inner::Integer = 1,
        ties = nothing,
    )
    disp_plan === nothing && return 0
    leaves = [m for (_, m) in keyed]
    length(leaves) >= 4 && return _refine_scan_dispersion_impl!(
        θ, leaves, geom, ev, delay_plan, disp_plan, ref_ant, nant,
        opts, Float64(snr_min), Float64(tau_max), Float64(dtec_max), Int(inner), ties,
    )
    return 0                                    # < 4 bands can't constrain 1/ν
end

function _refine_scan_dispersion_impl!(
        θ, leaves, geom, ev, delay_plan, disp_plan, ref_ant, nant,
        opts, snr_min, tau_max, dtec_max, inner, ties = nothing,
    )
    l0 = first(leaves)
    bl_pairs = collect(UVData.baselines(l0).pairs)
    pols = String.(pol_products(l0))
    feeds = [correlation_feed_pair(p) for p in pols]
    _, g_ti = leaf_window(geom, l0)
    nbl = length(bl_pairs)
    npol = length(pols)
    nlf = length(leaves)
    z = zeros(ComplexF64, nbl, npol, nlf)
    w = zeros(Float64, nbl, npol, nlf)
    fb = zeros(Float64, nlf)
    chunks = collect(Iterators.partition(1:nlf, cld(nlf, clamp(inner, 1, nlf))))
    tasks = map(chunks) do idxs
        Threads.@spawn for li in idxs
            leaf = leaves[li]
            ci, ti = leaf_window(geom, leaf)
            g = evaluate_gains(ev, θ, ci, ti)
            fs = lookup(leaf[:vis], Frequency)
            fb[li] = sum(Float64, fs) / length(fs)
            _accumulate_leaf_band_phasor!(
                view(z, :, :, li), view(w, :, :, li),
                parent(leaf[:vis]), parent(leaf[:weights]), g, bl_pairs, pols,
            )
        end
    end
    foreach(wait, tasks)
    return _dispersion_fit_stationize!(
        θ, z, w, fb, bl_pairs, pols, feeds, first(g_ti), geom,
        delay_plan, disp_plan, ref_ant, opts, snr_min, tau_max, dtec_max, ties,
    )
end

# Concat-cube variant: the same refinement on a materialized `_ScanGroup`, its
# per-spw channel blocks standing in for the band leaves. Lets the bandpass
# stage keep the fast decode-into-cube path (no per-leaf materialization).
function _refine_scan_dispersion!(
        θ, grp::_ScanGroup, geom::DataGeometry, ev, delay_plan, disp_plan, ref_ant, nant;
        opts::Stationization = Stationization(reject_iters = 0), snr_min::Real = 8.0,
        tau_max::Real = 2.0e-8, dtec_max::Real = 45.0, inner::Integer = 1,
        ties = nothing,
    )
    disp_plan === nothing && return 0
    soc = geom.spw_of_chan
    blocks = UnitRange{Int}[]                    # grp-local channel range per spw
    lo = 1
    for c in 2:length(grp.g_ci)
        if soc[grp.g_ci[c]] != soc[grp.g_ci[c - 1]]
            push!(blocks, lo:(c - 1))
            lo = c
        end
    end
    push!(blocks, lo:length(grp.g_ci))
    length(blocks) >= 4 || return 0              # < 4 bands can't constrain 1/ν
    feeds = [correlation_feed_pair(p) for p in grp.pol_products]
    nbl = length(grp.bl_pairs)
    npol = length(grp.pol_products)
    nlf = length(blocks)
    z = zeros(ComplexF64, nbl, npol, nlf)
    w = zeros(Float64, nbl, npol, nlf)
    fb = [sum(@view grp.fg[r]) / length(r) for r in blocks]
    chunks = collect(Iterators.partition(1:nlf, cld(nlf, clamp(Int(inner), 1, nlf))))
    tasks = map(chunks) do idxs
        Threads.@spawn for li in idxs
            r = blocks[li]
            g = evaluate_gains(ev, θ, grp.g_ci[r], grp.g_ti)
            _accumulate_leaf_band_phasor!(
                view(z, :, :, li), view(w, :, :, li),
                view(grp.Vg, r, :, :, :), view(grp.Wg, r, :, :, :), g,
                grp.bl_pairs, grp.pol_products,
            )
        end
    end
    foreach(wait, tasks)
    return _dispersion_fit_stationize!(
        θ, z, w, fb, grp.bl_pairs, grp.pol_products, feeds, first(grp.g_ti), geom,
        delay_plan, disp_plan, ref_ant, opts, Float64(snr_min), Float64(tau_max), Float64(dtec_max), ties,
    )
end

# Shared back half of the refinement: per-baseline joint (Δτ, dTEC) fits over
# the accumulated band phasors, then the two station solves. `ties` (ant →
# representative, from `_colocated_ties`) makes co-located stations share ONE
# dTEC: they see the same ionosphere, so the dispersion solve runs on the
# representative nodes (the co-located pair's own baseline drops out — its
# endpoints coincide) and members copy the representative's value. Delays stay
# per-station (instrumental paths differ).
function _dispersion_fit_stationize!(
        θ, z, w, fb, bl_pairs, pols, feeds, ti0, geom,
        delay_plan, disp_plan, ref_ant, opts, snr_min, tau_max, dtec_max, ties = nothing,
    )
    nbl, npol, nlf = size(z)
    Dτ = fill(_INVALID_DETECTION, nbl, npol)
    Dd = fill(_INVALID_DETECTION, nbl, npol)
    for p in 1:npol
        feeds[p][1] == feeds[p][2] || continue
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            a == b && continue
            rows_f = Float64[]
            rows_z = ComplexF64[]
            rows_w = Float64[]
            for li in 1:nlf
                w[bi, p, li] > 0 || continue
                push!(rows_f, fb[li])
                push!(rows_z, z[bi, p, li])
                push!(rows_w, w[bi, p, li])
            end
            length(rows_f) >= 4 || continue
            fit = _fit_band_dispersion(
                rows_f, rows_z, rows_w, geom.f0;
                tau_max = _band_delay_halfwindow(rows_f, tau_max), dtec_max = dtec_max,
            )
            fit.snr >= snr_min || continue
            Dτ[bi, p] = FringeDetection(fit.tau, 0.0, 0.0, fit.amp, fit.snr, true)
            Dd[bi, p] = FringeDetection(fit.dtec, 0.0, 0.0, fit.amp, fit.snr, true)
        end
    end

    if haskey(ENV, "GUSTAVO_DTEC_DEBUG")
        for p in 1:npol, bi in 1:nbl
            Dτ[bi, p].valid || continue
            println(
                "  dtec-fit bl=", bl_pairs[bi], " p=", p,
                " Δτ=", round(Dτ[bi, p].delay * 1.0e12; digits = 1), "ps",
                " dtec=", round(Dd[bi, p].delay; digits = 3),
                " snr=", round(Dτ[bi, p].snr; digits = 1),
            )
        end
    end
    nrej = 0
    tied = ties !== nothing && any(ties[i] != i for i in eachindex(ties))
    for (D, plan, tie) in ((Dτ, delay_plan, false), (Dd, disp_plan, tied))
        plan === nothing && continue
        any(d -> d.valid, D) || continue
        pairs_s = bl_pairs
        Ds = D
        if tie
            # Solve on representative nodes; a baseline whose endpoints collapse
            # to the same representative (the co-located pair itself) carries no
            # differential-TEC information and is dropped.
            pairs_s = [(ties[a], ties[b]) for (a, b) in bl_pairs]
            Ds = copy(D)
            for bi in eachindex(pairs_s)
                if pairs_s[bi][1] == pairs_s[bi][2]
                    for p in axes(Ds, 2)
                        Ds[bi, p] = _INVALID_DETECTION
                    end
                end
            end
            any(d -> d.valid, Ds) || continue
        end
        _, _, nr, _ = solve_station_systems!(
            θ, (StationScanDetections(Ds, pairs_s, feeds, ti0),), ((plan, :delay),);
            ref_ant = tie ? ties[ref_ant] : ref_ant, opts = opts,
        )
        nrej += nr
        if tie
            # Members inherit the representative's solved value (assignment, not
            # increment — both start equal, so totals stay equal).
            ts = plan.tseg_id[ti0]
            for a in eachindex(ties)
                ties[a] == a && continue
                for f in 1:2
                    offm = plan.off1[a, f, ts, 1]
                    offr = plan.off1[ties[a], f, ts, 1]
                    (offm == 0 || offr == 0) && continue
                    θ[offm] = θ[offr]
                end
            end
        end
    end
    haskey(ENV, "GUSTAVO_DTEC_DEBUG") && println("  dtec-refine rejected: ", nrej)
    return nrej
end

# Half the band-comb delay ambiguity: band phasors sampled on band centers with
# an (approximate) common spacing grid `g` cannot distinguish τ from τ + k/g, so
# the (Δτ, dTEC) fit must search a window with a UNIQUE branch — otherwise
# baselines tie-break the exact degeneracy to different branches, the
# measurements break closure, and the robust station solve excises them instead
# of fixing the delay. Uses the same folded-Euclid grid the MBD search uses.
function _band_delay_halfwindow(fbs::Vector{Float64}, tau_max::Float64)
    length(fbs) < 3 && return tau_max
    fs = sort(fbs)
    steps = diff(fs)
    g = _approx_gcd(steps, 0.05 * minimum(steps))
    g > 0 || return tau_max
    return min(tau_max, 0.499 / g)
end

# ── Per-scan band-group SBD refinement (fourfit's single-band delay) ──────────
#
# The wideband (MBD) delay and dTEC are constrained by CROSS-band structure;
# the WITHIN-band phase slope is nearly orthogonal to both and instrumentally
# real: a station's per-band signal path can move relative to its phase-cal
# tones between scans (VR2505's YJ drifts by ~30 ns in the 3 GHz group), which
# no time-invariant per-channel bandpass can represent. This stage measures the
# residual within-band slope per (baseline, band group) from sub-band CHUNK
# phasors (exact matched filter over one delay about the group's centre),
# stationizes each group's slopes, and accumulates into the per-scan
# `Delay × FrequencyBands` column plus its companion constant (net correction
# 2πτ(ν − νg): zero at the group centre, so the cross-band solution the MBD +
# dTEC terms own is untouched). SNR-gated — quiet stations contribute nothing.

# Accumulate one channel-block's inverse-variance chunk phasors:
# `z[bi, p, chunk_of_chan[c]] += w·|den|²·(V/den)` (parallel hands only).
# Function barrier (V/W type-unstable at the call site).
function _accumulate_leaf_chunks!(z, w, V, W, g, bl_pairs, pols, chunk_of_chan)
    nchan, nti, nbl, npol = size(V)
    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(pols[p])
        fa == fb || continue
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            a == b && continue
            for tt in 1:nti, c in 1:nchan
                ww = W[c, tt, bi, p]
                (ww > 0 && isfinite(ww)) || continue
                ga = g[c, tt, a, fa]
                gb = g[c, tt, b, fb]
                den = ga * conj(gb)
                (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(den)) || continue
                v = V[c, tt, bi, p] / den
                isfinite(v) || continue
                wd = ww * abs2(den)
                k = chunk_of_chan[c]
                z[bi, p, k] += wd * v
                w[bi, p, k] += wd
            end
        end
    end
    return nothing
end

# Exact matched filter for ONE delay over chunk phasors about centre `fc`:
# argmax_τ |Σ_k z_k·cis(−2πτ(f_k − fc))|, coarse→fine sweep + parabolic polish.
function _fit_chunk_delay(
        fs::Vector{Float64}, zs::Vector{ComplexF64}, ws::Vector{Float64}, fc::Float64;
        tau_max::Float64,
    )
    value(τ) = abs(sum(zs[k] * cis(-2π * τ * (fs[k] - fc)) for k in eachindex(fs)))
    best_a = -1.0
    best_t = 0.0
    coarse = tau_max / 400
    for τ in (-tau_max):coarse:tau_max
        a = value(τ)
        a > best_a && (best_a = a; best_t = τ)
    end
    fine = coarse / 50
    for τ in (best_t - coarse):fine:(best_t + coarse)
        a = value(τ)
        a > best_a && (best_a = a; best_t = τ)
    end
    am = value(best_t - fine)
    ap = value(best_t + fine)
    den = am - 2 * best_a + ap
    if den < 0
        δ = 0.5 * fine * (am - ap) / den
        if abs(δ) <= fine
            best_t += δ
            best_a = value(best_t)
        end
    end
    var = sum(ws)
    snr = var > 0 ? sqrt(max(best_a^2 - var, 0.0) / var) : 0.0
    return (tau = best_t, amp = best_a, snr = snr)
end

# Shared back half: per-(baseline, group) fits over the accumulated chunk
# phasors — the within-group SLOPE (exact matched filter) plus the slope-
# corrected group phasor's PHASE — then one feed-common station solve of each
# per group. Both are needed: the wideband delay's decomposition against the
# per-group slopes is ambiguous (a common-mode slope shift leaves per-group
# constants of 2πΔτ(f0 − νg) behind that no single per-scan constant can
# absorb), and real instruments carry genuine per-band phase offsets. θ gets
# the per-scan per-group delay and its constant: net correction
# 2πτ(f − νg) + φg, referenced to the group centre. `chunkf`/`chunkgrp` label
# each accumulated chunk with its centre frequency and band-group id.
function _sbd_fit_stationize!(
        θ, z, w, chunkf, chunkgrp, bl_pairs, pols, feeds, ti0, geom, sbd, ref_ant, nant;
        opts::Stationization = Stationization(reject_iters = 0),
        snr_min::Float64 = 8.0, tau_max::Float64 = 6.0e-8,
    )
    nbl, npol, _ = size(z)
    ngrp = length(sbd.bands)
    nrej = 0
    # ── Tier 1: per-group fits + within-group SLOPE guard ───────────────────
    # The per-baseline SNR gate cannot catch a BIASED fit: on a group whose
    # chunk phasors are internally decoherent (real per-channel bandpass
    # structure, not a delay — VGOS band 2 is the worst), the matched filter
    # happily returns a confident wrong slope, and applying it bends every
    # baseline of a previously-better group (the VR2505 full run: +SBD
    # degraded g1/g2 η_f on 1803+784 and 4C39.25 and band-2 on 3C454.3 while
    # genuinely fixing 0607-157 and YJ). So: keep a group's stationized slope
    # only when it RAISES the group's within-band coherence on the very chunk
    # phasors it was fit from. The group CONSTANT is judged separately — it
    # cancels inside the within-group coherence (|Σ z·e^{-iφ}| = |Σ z|), so
    # this tier is structurally blind to it. A failing slope is ZEROED and the
    # group phase refit at τ = 0 (NOT dropped with the group: φg carries the
    # cross-band alignment, and dropping it collapsed 1803+784's cross-band η
    # in the first, group-dropping version of this guard).
    gsols = NamedTuple[]
    for gidx in 1:ngrp
        ks = findall(==(gidx), chunkgrp)
        length(ks) >= 3 || continue
        fc = sum(chunkf[ks]) / length(ks)       # FIXED group centre (matches the θ write)
        rows_τ = _ObsRow[]
        # per accepted baseline row: (a, b, φ at fitted slope, φ at τ = 0, weight)
        φrows = Tuple{Int, Int, Float64, Float64, Float64}[]
        for p in 1:npol
            feeds[p][1] == feeds[p][2] || continue
            for bi in 1:nbl
                a, b = bl_pairs[bi]
                a == b && continue
                fs = Float64[]
                zs = ComplexF64[]
                ws = Float64[]
                for k in ks
                    w[bi, p, k] > 0 || continue
                    push!(fs, chunkf[k])
                    push!(zs, z[bi, p, k])
                    push!(ws, w[bi, p, k])
                end
                length(fs) >= 3 || continue
                fit = _fit_chunk_delay(fs, zs, ws, fc; tau_max = _band_delay_halfwindow(fs, tau_max))
                fit.snr >= snr_min || continue
                φs = angle(sum(zs[k] * cis(-2π * fit.tau * (fs[k] - fc)) for k in eachindex(fs)))
                φ0 = angle(sum(zs))
                # feed-common node (SharedFeeds model): both hands constrain feed 1.
                push!(rows_τ, _ObsRow(a, b, 1, 1, fit.tau, fit.snr^2, 0))
                push!(φrows, (a, b, φs, φ0, fit.snr^2))
            end
        end
        isempty(rows_τ) && continue
        τv, _, covτ, _ = _solve_observable_robust(rows_τ, nant, ref_ant, opts; use_chi = false, rewrap = 0)
        num0 = 0.0
        num1 = 0.0
        for p in 1:npol
            feeds[p][1] == feeds[p][2] || continue
            for bi in 1:nbl
                a, b = bl_pairs[bi]
                a == b && continue
                τab = (covτ[a, 1] ? τv[a, 1] : 0.0) - (covτ[b, 1] ? τv[b, 1] : 0.0)
                acc0 = zero(ComplexF64)
                acc1 = zero(ComplexF64)
                for k in ks
                    w[bi, p, k] > 0 || continue
                    zk = z[bi, p, k]
                    acc0 += zk
                    acc1 += zk * cis(-2π * τab * (chunkf[k] - fc))
                end
                num0 += abs(acc0)
                num1 += abs(acc1)
            end
        end
        slope_ok = num1 > num0
        if !slope_ok
            nrej += 1
            fill!(τv, 0.0)
        end
        rows_φ = [_ObsRow(r[1], r[2], 1, 1, slope_ok ? r[3] : r[4], r[5], 0) for r in φrows]
        φv, _, covφ, _ = _solve_observable_robust(rows_φ, nant, ref_ant, opts; use_chi = false, rewrap = 2)
        push!(gsols, (; gidx, ks, fc, τv, covτ, φv, covφ))
    end
    isempty(gsols) && return nrej
    # ── Tier 2: scan-level JOINT guard on the CROSS-band statistic ──────────
    # The group constants' whole job is aligning the groups' band phasors, so
    # judge them (together with the surviving slopes) on the coherent sum over
    # ALL groups per baseline — the ηx-flavoured statistic — and apply the
    # scan's SBD solution all-or-nothing.
    num0 = 0.0
    num1 = 0.0
    for p in 1:npol
        feeds[p][1] == feeds[p][2] || continue
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            a == b && continue
            acc0 = zero(ComplexF64)
            acc1 = zero(ComplexF64)
            for gs in gsols
                τab = (gs.covτ[a, 1] ? gs.τv[a, 1] : 0.0) - (gs.covτ[b, 1] ? gs.τv[b, 1] : 0.0)
                φab = (gs.covφ[a, 1] ? gs.φv[a, 1] : 0.0) - (gs.covφ[b, 1] ? gs.φv[b, 1] : 0.0)
                for k in gs.ks
                    w[bi, p, k] > 0 || continue
                    zk = z[bi, p, k]
                    acc0 += zk
                    acc1 += zk * cis(-(2π * τab * (chunkf[k] - gs.fc) + φab))
                end
            end
            num0 += abs(acc0)
            num1 += abs(acc1)
        end
    end
    num1 > num0 || return nrej + length(gsols)
    ts = sbd.dplan.tseg_id[ti0]
    for gs in gsols
        for a in 1:nant
            τa = (gs.covτ[a, 1] && isfinite(gs.τv[a, 1])) ? gs.τv[a, 1] : 0.0
            φa = (gs.covφ[a, 1] && isfinite(gs.φv[a, 1])) ? gs.φv[a, 1] : 0.0
            (τa == 0.0 && φa == 0.0) && continue
            offd = sbd.dplan.off1[a, 1, ts, gs.gidx]
            offd == 0 && continue
            θ[offd] += τa
            offc = sbd.cplan.off1[a, 1, ts, gs.gidx]
            offc == 0 && continue
            θ[offc] += φa - 2π * τa * (gs.fc - geom.f0)
        end
    end
    return nrej
end

# Leaf-group variant (pass 2): each band leaf is split into `nchunk` contiguous
# channel chunks; their phasors feed `_sbd_fit_stationize!`.
function _refine_scan_sbd!(
        θ, keyed::AbstractVector, geom::DataGeometry, ev, sbd, ref_ant, nant;
        nchunk::Integer = 4, snr_min::Real = 8.0, tau_max::Real = 6.0e-8, inner::Integer = 1,
    )
    sbd === nothing && return 0
    leaves = [m for (_, m) in keyed]
    l0 = first(leaves)
    bl_pairs = collect(UVData.baselines(l0).pairs)
    pols = String.(pol_products(l0))
    feeds = [correlation_feed_pair(p) for p in pols]
    _, g_ti = leaf_window(geom, l0)
    nbl = length(bl_pairs)
    npol = length(pols)
    # chunk table: nchunk per leaf, labelled with centre freq + band-group id
    nlf = length(leaves)
    ntot = nlf * Int(nchunk)
    z = zeros(ComplexF64, nbl, npol, ntot)
    w = zeros(Float64, nbl, npol, ntot)
    chunkf = zeros(Float64, ntot)
    chunkgrp = zeros(Int, ntot)
    chunks = collect(Iterators.partition(1:nlf, cld(nlf, clamp(Int(inner), 1, nlf))))
    tasks = map(chunks) do idxs
        Threads.@spawn for li in idxs
            leaf = leaves[li]
            ci, ti = leaf_window(geom, leaf)
            g = evaluate_gains(ev, θ, ci, ti)
            fs = Float64.(lookup(leaf[:vis], Frequency))
            nc = length(fs)
            grp = findfirst(r -> first(ci) in r, sbd.bands)
            grp === nothing && continue
            edges = round.(Int, range(0, nc; length = Int(nchunk) + 1))
            coc = Vector{Int}(undef, nc)
            for k in 1:Int(nchunk)
                lo, hi = edges[k] + 1, edges[k + 1]
                hi >= lo || continue
                kk = (li - 1) * Int(nchunk) + k
                coc[lo:hi] .= kk
                chunkf[kk] = sum(@view fs[lo:hi]) / (hi - lo + 1)
                chunkgrp[kk] = grp
            end
            _accumulate_leaf_chunks!(
                view(z, :, :, ((li - 1) * Int(nchunk) + 1):(li * Int(nchunk))),
                view(w, :, :, ((li - 1) * Int(nchunk) + 1):(li * Int(nchunk))),
                parent(leaf[:vis]), parent(leaf[:weights]), g, bl_pairs, pols,
                coc .- (li - 1) * Int(nchunk),
            )
        end
    end
    foreach(wait, tasks)
    return _sbd_fit_stationize!(
        θ, z, w, chunkf, chunkgrp, bl_pairs, pols, feeds, first(g_ti), geom, sbd, ref_ant, nant;
        snr_min = Float64(snr_min), tau_max = Float64(tau_max),
    )
end

# Concat-cube variant (bandpass stage): per-spw channel blocks of the stacked
# cube stand in for the leaves.
function _refine_scan_sbd!(
        θ, grp::_ScanGroup, geom::DataGeometry, ev, sbd, ref_ant, nant;
        nchunk::Integer = 4, snr_min::Real = 8.0, tau_max::Real = 6.0e-8, inner::Integer = 1,
    )
    sbd === nothing && return 0
    soc = geom.spw_of_chan
    blocks = UnitRange{Int}[]                    # grp-local channel range per spw
    lo = 1
    for c in 2:length(grp.g_ci)
        if soc[grp.g_ci[c]] != soc[grp.g_ci[c - 1]]
            push!(blocks, lo:(c - 1))
            lo = c
        end
    end
    push!(blocks, lo:length(grp.g_ci))
    feeds = [correlation_feed_pair(p) for p in grp.pol_products]
    nbl = length(grp.bl_pairs)
    npol = length(grp.pol_products)
    nlf = length(blocks)
    ntot = nlf * Int(nchunk)
    z = zeros(ComplexF64, nbl, npol, ntot)
    w = zeros(Float64, nbl, npol, ntot)
    chunkf = zeros(Float64, ntot)
    chunkgrp = zeros(Int, ntot)
    chs = collect(Iterators.partition(1:nlf, cld(nlf, clamp(Int(inner), 1, nlf))))
    tasks = map(chs) do idxs
        Threads.@spawn for li in idxs
            r = blocks[li]
            g = evaluate_gains(ev, θ, grp.g_ci[r], grp.g_ti)
            fs = grp.fg[r]
            nc = length(fs)
            bgrp = findfirst(rr -> grp.g_ci[first(r)] in rr, sbd.bands)
            bgrp === nothing && continue
            edges = round.(Int, range(0, nc; length = Int(nchunk) + 1))
            coc = Vector{Int}(undef, nc)
            for k in 1:Int(nchunk)
                klo, khi = edges[k] + 1, edges[k + 1]
                khi >= klo || continue
                kk = (li - 1) * Int(nchunk) + k
                coc[klo:khi] .= kk
                chunkf[kk] = sum(@view fs[klo:khi]) / (khi - klo + 1)
                chunkgrp[kk] = bgrp
            end
            _accumulate_leaf_chunks!(
                view(z, :, :, ((li - 1) * Int(nchunk) + 1):(li * Int(nchunk))),
                view(w, :, :, ((li - 1) * Int(nchunk) + 1):(li * Int(nchunk))),
                view(grp.Vg, r, :, :, :), view(grp.Wg, r, :, :, :), g,
                grp.bl_pairs, grp.pol_products,
                coc .- (li - 1) * Int(nchunk),
            )
        end
    end
    foreach(wait, tasks)
    return _sbd_fit_stationize!(
        θ, z, w, chunkf, chunkgrp, grp.bl_pairs, grp.pol_products, feeds,
        first(grp.g_ti), geom, sbd, ref_ant, nant;
        snr_min = Float64(snr_min), tau_max = Float64(tau_max),
    )
end

# Estimate the PEAK resident bytes for processing one scan group through the bulk
# reader. The bulk path holds, at once: the transient row-span buffer (~one copy
# of the raw bytes), the per-band materialized leaves, AND the concatenated
# `_ScanGroup` copy — plus a windowed gain cube. That is ~3–4× the raw vis size;
# we charge 4× (vis ComplexF32 = 8 B + weights Float32 = 4 B ⇒ 12 B/cell, ×4) as a
# safe peak. `group_leaves[i]` is a vector of `(key, lazy_leaf)`; summing
# `prod(size(leaf[:vis]))` over its bands gives the group's cell count without
# materializing anything.
# Estimate the PEAK resident bytes for one scan group (vis ComplexF32 8 B + weights
# Float32 4 B = 12 B/cell). During decode the search pass holds the stacked cube
# (one copy) PLUS the transient bulk-read span (≈ one copy — on-disk FLUX is the
# same 8 B/cell, measured span/cube ≈ 0.8–1.1 here), so peak ≈ 2× cube; pass 2
# similarly holds the per-band leaves plus the corrected copy `apply_calibration`
# allocates. We charge 2.5× for GC/heap headroom over that ~2.25× peak. The
# decode-into-cube change removed the OLD second copy (leaves AND cube held at once),
# but the now-larger in-RAM span (the 3 GiB cap lets big scans use the fast read)
# brings the peak back to ~2× — so the charge stays 2.5×.
function _group_peak_bytes(group)
    cells = 0
    for (_, leaf) in group
        cells += prod(size(leaf[:vis]))
    end
    return round(Int, 2.5 * 12 * cells)
end

# The DETERMINISTIC memory budget for the solve (bytes): an explicit `mem_budget`
# if given, else `mem_fraction` of TOTAL physical RAM. Total RAM is a machine
# constant, so the budget — and therefore `ntasks`, and therefore the whole solve's
# wall time — is REPRODUCIBLE across runs on a box. The earlier version budgeted
# from `/proc/meminfo` `MemAvailable`, which swings as other processes (IDE/language
# servers) come and go and so flipped the cap between e.g. 2 and 3 tasks on
# identical inputs — making the solve impossible to benchmark. The trade-off of
# using total: it can over-commit a heavily LOADED box, so `mem_fraction` must leave
# headroom for everything else (lower it on a shared box, or pass an explicit
# `mem_budget` for machine-independent control). The conservative per-group peak
# charge (`_group_peak_bytes`, 2.5× the real ~2× footprint) provides further slack.
function _memory_budget(mem_fraction, mem_budget)
    mem_budget === nothing || return Float64(mem_budget)
    return mem_fraction * Float64(Sys.total_memory())
end

# Bound `ntasks` so peak RAM (`ntasks × per-group peak`) stays under the
# deterministic memory budget. Defaulting `ntasks` to `Threads.nthreads()` OOMs on a
# real (24 GB) file: each concurrent group peaks at several GB. We size the cap from
# the LARGEST group and the budget. Returns ≥ 1. NOTE: the group loops themselves
# now admit per group via `_budget_scheduled_map`; this worst-case cap remains as
# the DETERMINISTIC sizing for `inner` and the bandpass-stage chunk count (whose
# accumulation order must not depend on runtime scheduling).
function _bounded_ntasks(group_leaves, ntasks, ngroups; mem_fraction = 0.6, mem_budget = nothing)
    requested = max(1, min(Int(ntasks), ngroups))
    peak = maximum(_group_peak_bytes, group_leaves; init = 0)
    peak <= 0 && return requested
    budget = _memory_budget(mem_fraction, mem_budget)
    cap = max(1, Int(floor(budget / peak)))
    used = min(requested, cap)
    if used < requested
        @info "Fringe solve: capping ntasks for memory" requested cap used peak_GB = round(peak / 2^30; digits = 2) total_GB = round(Sys.total_memory() / 2^30; digits = 2) budget_GB = round(budget / 2^30; digits = 2)
    end
    return used
end

# ── Budget-scheduled group execution ─────────────────────────────────────────
# The fixed `ntasks = ⌊budget / LARGEST-group charge⌋` cap (`_bounded_ntasks`)
# collapses to 1 when a track has a few big scans — serializing EVERY scan even
# though the median scan is several times smaller (VGOS VR2505: 4 scans charge
# 10.7 GB but the median is 2.7 GB, so a 18 GB budget ran 33 scans strictly one
# at a time, with each scan's decode I/O dead time on top). Instead, admit each
# group by ITS OWN charge against a shared memory budget: big groups run
# (nearly) alone, small groups pack into the leftover budget, and a group's
# decode overlaps other groups' compute. Among the groups that currently fit,
# the LARGEST is picked first, so the long poles start immediately and the tail
# packs around them; a group charged more than the whole budget is clamped so
# it still runs (alone). `max_tasks` bounds worker concurrency. Results are
# returned in `idxs` order, and `work` must be independent across groups (all
# fringe passes qualify: they write disjoint per-scan θ slots and per-scan
# output rows). Returns `(results, peak_concurrency)`.
function _budget_scheduled_map(work::F, idxs, charges, budget; max_tasks::Integer) where {F}
    n = length(idxs)
    n == 0 && return Any[], 0
    out = Vector{Any}(undef, n)
    # Positions into `idxs`/`charges` not yet started, biggest charge first.
    remaining = sort(collect(1:n); by = k -> -Float64(charges[k]))
    cond = Threads.Condition()
    avail = Ref(Float64(budget))
    inflight = Ref(0)
    peak = Ref(0)
    workers = map(1:max(1, min(Int(max_tasks), n))) do _
        Threads.@spawn while true
            k = 0
            amt = 0.0
            lock(cond)
            try
                while true
                    isempty(remaining) && break
                    # First (= largest) not-yet-started group that fits the
                    # remaining budget; when none fits, wait for a release. An
                    # over-budget group is clamped, so with nothing in flight
                    # (avail == budget) SOMETHING always fits — no deadlock.
                    j = findfirst(kk -> min(Float64(charges[kk]), Float64(budget)) <= avail[], remaining)
                    if j === nothing
                        wait(cond)
                    else
                        k = remaining[j]
                        deleteat!(remaining, j)
                        amt = min(Float64(charges[k]), Float64(budget))
                        avail[] -= amt
                        inflight[] += 1
                        peak[] = max(peak[], inflight[])
                        break
                    end
                end
            finally
                unlock(cond)
            end
            k == 0 && break
            try
                out[k] = work(idxs[k])
            finally
                lock(cond)
                try
                    avail[] += amt
                    inflight[] -= 1
                    notify(cond)
                finally
                    unlock(cond)
                end
            end
        end
    end
    # A failed worker rethrows here (before `out` is touched), after the other
    # workers have drained the queue.
    foreach(wait, workers)
    return map(identity, out), peak[]
end

# Pass 1: search every group (threaded; on the residual for rounds > 1), then a
# SINGLE global stage-B solve over all groups' detections — `solve_station_systems!`
# couples scans through the global R–L offset column, so it cannot be per-group.
# Mutates `θ` (stage-B slots), `scan_snr`, `scan_ncells` (the per-scan effective
# independent search cells, for `fringe_pfa`), `scan_dets` (the per-scan valid
# `_DetRow`s; on `rounds > 1` the LAST round's residual search, matching
# `scan_snr`), and the profiling vectors `scan_t_decode`/`scan_t_search`
# (seconds, ACCUMULATED across rounds); returns (chi, ncomp).
function _search_and_stationize!(
        θ, scan_snr, scan_ncells, scan_dets, scan_t_decode, scan_t_search,
        group_leaves, geom, ev, stageB, f0, t0_sec,
        search, rounds, ref_ant, charges, budget, max_tasks, ws_pool, inner, pc, progress,
    )
    ngroups = length(group_leaves)
    nrounds = max(rounds, 1)
    chi = NaN
    ncomp = 0
    nrej = 0
    peak_par = 0
    flags = Vector{Tuple{Int, Int}}()
    ndone = Threads.Atomic{Int}(0)
    _progress_notify(progress, :search, 0, ngroups * nrounds)
    for round in 1:nrounds
        dets, pk = _budget_scheduled_map(1:ngroups, charges, budget; max_tasks = max_tasks) do gi
            t0w = time_ns()
            grp = _materialize_concat_group(group_leaves[gi], geom, pc, inner)
            Vsearch = round > 1 ? _residual_vis(ev, θ, grp) : grp.Vg
            t1w = time_ns()
            det, snr, ncells, rows = _search_group(grp, Vsearch, f0, t0_sec, search, ws_pool, inner)
            t2w = time_ns()
            scan_snr[gi] = snr
            scan_ncells[gi] = ncells
            scan_dets[gi] = rows
            scan_t_decode[gi] += (t1w - t0w) / 1.0e9
            scan_t_search[gi] += (t2w - t1w) / 1.0e9
            _progress_notify(progress, :search, Threads.atomic_add!(ndone, 1) + 1, ngroups * nrounds)
            feeds = [correlation_feed_pair(p) for p in grp.pol_products]
            StationScanDetections(det, grp.bl_pairs, feeds, first(grp.g_ti))
        end
        peak_par = max(peak_par, pk)
        chi, ncomp, nrej, covered = solve_station_systems!(θ, dets, stageB; ref_ant = ref_ant)
        # EHT-HOPS-style station flags: a station that PARTICIPATES in a scan
        # (has baselines there) but is left UNCONSTRAINED by the surviving
        # stage-B rows keeps identity gains — record it as (station, geometry
        # scan id) so `apply_calibration` zero-weights its baselines instead
        # of passing raw phases through at full weight. Rebuilt each round
        # (matching θ, which the last round's solve owns).
        empty!(flags)
        for gi in eachindex(dets)
            scanid = geom.scan_of_time[dets[gi].ti]
            stations = Set{Int}()
            for (a, b) in dets[gi].bl_pairs
                a == b && continue
                push!(stations, a); push!(stations, b)
            end
            for st in stations
                (st, gi) in covered || push!(flags, (st, scanid))
            end
        end
    end
    return chi, ncomp, nrej, peak_par, flags
end

# EHT-HOPS-style flag block for the solution `info` (plain parallel vectors,
# HDF5-representable): (station, scan) pairs stage B left unconstrained and the
# intra-site baselines excluded for crosstalk. Shared by the final `info` and
# the per-group `sol_local` of the streaming reduce path, so the streamed
# output honors the flags identically to `apply_calibration` on the full set.
function _flag_info(station_flags, excl)
    pairs_ab = excl === nothing ? Tuple{Int, Int}[] : sort!([p for p in excl if p[1] < p[2]])
    return (;
        flagged_ant = Int[f[1] for f in station_flags],
        flagged_scan = Int[f[2] for f in station_flags],
        excluded_ant_a = Int[p[1] for p in pairs_ab],
        excluded_ant_b = Int[p[2] for p in pairs_ab],
    )
end

# Flatten the per-scan `_DetRow`s into parallel plain vectors for the solution
# `info` — HDF5-representable (so the detection table lands in the caltable file)
# and cheap to filter (`suspect_fringes`).
function _flatten_detections(scan_dets)
    n = sum(length, scan_dets; init = 0)
    det_scan = Vector{Int}(undef, n); det_ant_a = Vector{Int}(undef, n)
    det_ant_b = Vector{Int}(undef, n); det_pol = Vector{String}(undef, n)
    det_snr = Vector{Float64}(undef, n); det_pfa = Vector{Float64}(undef, n)
    i = 0
    for (gi, rows) in enumerate(scan_dets), r in rows
        i += 1
        det_scan[i] = gi; det_ant_a[i] = r.a; det_ant_b[i] = r.b
        det_pol[i] = r.pol; det_snr[i] = r.snr; det_pfa[i] = r.pfa
    end
    return (; det_scan, det_ant_a, det_ant_b, det_pol, det_snr, det_pfa)
end

# ── Amplitude-bandpass smoothers (pluggable estimators) ───────────────────────────
#
# The per-(station, feed) log-amp bandpass is solved from the SUM closure
# `log|V̄_ab(ν)| = la_a(ν) + lb_b(ν)` over each spw (a +1/+1, signless-Laplacian
# incidence — FULL RANK, so no reference state). HOW the per-channel shape is
# estimated — and how low-/no-signal channels are filled — is a pluggable strategy:
# add an `AbstractBandpassSmoother` subtype and a `_fit_bandpass_segment` method
# (defined below `solve_fringes`) to extend.
abstract type AbstractBandpassSmoother end

"""
    FreeBandpass()

Free per-channel closure: one independent WLS per channel, no regularisation.
Follows the data but does NOT estimate low-/no-signal channels (left at |g| = 1).
The `λ → 0` / `degree → ∞` limit of the others.
"""
struct FreeBandpass <: AbstractBandpassSmoother end

"""
    PolynomialBandpass(degree = 4)

Smooth per-spw polynomial of `degree` in a centred/scaled frequency coordinate, fit
by a single closure WLS (the `PolynomialFreq` design convention). Estimates gaps by
the fit. Assumes the in-spw bandpass is ~a low-order polynomial (smooth passband +
gentle roll-off); a high degree can ring (Runge) at the edges.
"""
struct PolynomialBandpass <: AbstractBandpassSmoother
    degree::Int
    function PolynomialBandpass(degree::Integer = 4)
        degree >= 1 || error("PolynomialBandpass: degree must be ≥ 1")
        return new(Int(degree))
    end
end

"""
    PenalizedBandpass(lambda = 1.0)

Roughness-penalised per-channel bandpass (a Whittaker smoother): a free value per
channel plus a 2nd-difference smoothness penalty of strength `lambda` (relative to
the per-channel data weight). Makes NO shape assumption — follows real structure
where the SNR supports it and smoothly interpolates gaps where it does not.
`lambda → 0` ⇒ [`FreeBandpass`](@ref); large `lambda` ⇒ flat.
"""
struct PenalizedBandpass <: AbstractBandpassSmoother
    lambda::Float64
    function PenalizedBandpass(lambda::Real = 1.0)
        lambda >= 0 || error("PenalizedBandpass: lambda must be ≥ 0")
        return new(Float64(lambda))
    end
end

"""
    solve_fringes(uvset; search, adhoc, rounds, ref_ant) -> CalibrationSolution

Fringe-fit `uvset`. Builds the shared fringe model (feed-common per-scan phase +
delay, a global stable R–L offset, per-scan rate, and a per-integration adhoc
term) and solves it in two passes over the (source, scan) groups: (1) a multi-band
fringe search on every group followed by ONE global stationization that ties the
feeds with a track-wide R–L offset; (2) the globally-closing adhoc phase per
group on the stage-B residual. Returns the packed `CalibrationSolution`.

Because the R–L offset couples scans, the stage-B solve is global — so the data
is read once for the search pass and once for the adhoc pass (`rounds` re-runs
the search pass on the residual).

`exclude_colocated` (default `true`) drops intra-site baselines (co-located
station pairs, e.g. the Onsala/Wettzell twins) from the adhoc and phase/amp
bandpass accumulations: their huge-SNR non-closing crosstalk otherwise pulls
both twins' gains and decoheres their sky baselines (see
[`_colocated_pair_set`](@ref)). Stage B keeps them (closure-screened).
"""
function solve_fringes(
        uvset::UVSet;
        search::FringeSearch = FringeSearch(),
        adhoc::AbstractAdhocSmoother = SavitzkyGolaySmoother(),
        rounds::Int = 1,
        ref_ant::Integer = 1,
        ntasks::Integer = Threads.nthreads(),
        mem_fraction::Real = 0.6,
        mem_budget = nothing,
        phase_bandpass::Bool = true,
        amp_bandpass::Bool = true,
        amp_smoother::AbstractBandpassSmoother = PenalizedBandpass(1.0),
        bandpass_source = nothing,
        precal::Union{Nothing, CalibrationSolution} = nothing,
        flag_channels = nothing,
        progress = nothing,
        bandpass_max_scans::Integer = 0,
        dispersion = :auto,
        sbd = :auto,
        dtec_tie_colocated::Bool = true,
        exclude_colocated::Bool = true,
        adhoc_pseudo_stokes = :auto,
        reuse_bandpass_refine::Bool = true,
        bandpass_polish_dtec::Real = 20.0,
    )
    geom = build_geometry(uvset)
    model = _fringe_model(
        dispersion = _dispersion_enabled(dispersion, geom),
        sbd_bands = _sbd_bands(sbd, geom),
    )
    first_leaf = first(values(UVData.branches(uvset)))
    nant = length(UVData.metadata(first_leaf).antennas)
    layout = plan_parameters(model, nant, geom)
    ev = GainEvaluator(model, layout)
    stageB = _stageB_components(model, layout)
    adhoc_plan = _adhoc_plan(model, layout)
    adhoc_shared = _adhoc_shared(model)
    bp_plan = _bandpass_plan(model, layout)
    amp_bp_plan = _amp_bandpass_plan(model, layout)
    disp_plan = _dispersion_plan(model, layout)
    ps_delay_plan = _perscan_delay_plan(model, layout)
    sbd_plans = _sbd_plans(model, layout)
    ties = dtec_tie_colocated ? _colocated_ties(UVData.metadata(first_leaf).antennas) : nothing
    excl = if exclude_colocated
        s = _colocated_pair_set(UVData.metadata(first_leaf).antennas)
        isempty(s) ? nothing : s
    else
        nothing
    end
    psI = _pseudo_stokes_config(
        adhoc_pseudo_stokes, first_leaf, UVData.DimensionalData.metadata(uvset),
    )
    dtec_rejected = Threads.Atomic{Int}(0)
    f0 = geom.f0
    t0_sec = geom.t0 * 3600.0
    pc = _make_precal(precal, flag_channels, geom)

    group_leaves = _scan_group_leaves(uvset)
    group_sources = _group_sources(group_leaves)
    ngroups = length(group_leaves)
    scan_snr = zeros(ngroups)
    scan_ncells = zeros(ngroups)
    scan_dets = [_DetRow[] for _ in 1:ngroups]
    scan_t_decode = zeros(ngroups)
    scan_t_search = zeros(ngroups)
    scan_t_decode2 = zeros(ngroups)
    scan_t_adhoc = zeros(ngroups)
    θ = zeros(layout.nθ)
    charges = [_group_peak_bytes(g) for g in group_leaves]
    budget = _memory_budget(mem_fraction, mem_budget)
    ntasks_use = _bounded_ntasks(group_leaves, ntasks, ngroups; mem_fraction = mem_fraction, mem_budget = mem_budget)
    # Threads not consumed by group tasks fan out over each group's independent
    # per-baseline searches (`inner`); FFT plans stay single-threaded — the inner
    # tasks supersede FFTW threading (and the MBD transforms are too small for it).
    inner = max(1, Threads.nthreads() ÷ ntasks_use)
    bp_prefetch = ntasks_use == 1 && 1.6 * maximum(charges; init = 0) <= budget
    ws_pool = _ws_pool(Threads.nthreads())
    t_search_pass = 0.0
    t_bandpass_stage = 0.0
    t_adhoc_pass = 0.0
    peak_par = 0
    station_flags = Tuple{Int, Int}[]

    chi, ncomp, nrej = _with_decode_threads(inner) do
        _with_fft_threads(1) do
            _with_single_blas_thread() do
                t0w = time_ns()
                ch, nc, nrj, pk1, station_flags = _search_and_stationize!(
                    θ, scan_snr, scan_ncells, scan_dets, scan_t_decode, scan_t_search,
                    group_leaves, geom, ev, stageB, f0, t0_sec,
                    search, rounds, ref_ant, charges, budget, ntasks, ws_pool, inner, pc, progress,
                )
                t1w = time_ns()
                # Phase + amplitude bandpass (between Stage-B and adhoc; orthogonal
                # frequency structure). Solved once from the brightest calibrator,
                # time-stable, from one shared per-channel residual accumulation.
                refined_bp = Set{Int}()
                if (phase_bandpass && bp_plan !== nothing) || (amp_bandpass && amp_bp_plan !== nothing)
                    cal = _bandpass_calibrator(bandpass_source, group_sources, scan_snr)
                    refined_bp = Set(
                        _solve_bandpass_stage!(
                            θ, group_leaves, group_sources, cal, geom, ev,
                            phase_bandpass ? bp_plan : nothing, nant;
                            amp_plan = amp_bandpass ? amp_bp_plan : nothing, ref_ant = ref_ant,
                            amp_smoother = amp_smoother, pc = pc, progress = progress,
                            max_scans = bandpass_max_scans, scan_snr = scan_snr,
                            ntasks = ntasks_use, inner = inner, prefetch = bp_prefetch,
                            disp_plan = disp_plan, ps_delay_plan = ps_delay_plan,
                            dtec_rejected = dtec_rejected, sbd_plans = sbd_plans, ties = ties,
                            excl = excl,
                        ),
                    )
                end
                t2w = time_ns()
                # Pass 2: adhoc per group on the residual after the global stage-B (each
                # group writes its own disjoint per-integration slots of the shared θ).
                nadh = Threads.Atomic{Int}(0)
                _progress_notify(progress, :adhoc, 0, ngroups)
                _, pk2 = _budget_scheduled_map(1:ngroups, charges, budget; max_tasks = ntasks) do gi
                    ta = time_ns()
                    keyed = _materialize_leaf_group(group_leaves[gi], geom, pc, inner)
                    tb = time_ns()
                    # Reuse the bandpass stage's per-scan dTEC/SBD θ where it already
                    # fit this scan (see the reduce path / `reuse_bandpass_refine`).
                    reuse_scan = reuse_bandpass_refine && gi in refined_bp
                    # Per-scan (Δτ, dTEC) refinement BEFORE the adhoc solve, so the
                    # adhoc phases fit dispersion-corrected residuals. Writes only
                    # this scan's θ columns — safe under the group tmap. Bandpass-
                    # stage-fit scans are polished in a narrow window (see the reduce
                    # path / `reuse_bandpass_refine`).
                    if disp_plan !== nothing
                        # `local`: an unannotated `nrj` would capture-and-rebind the
                        # outer stage-B rejection count (a boxed capture OhMyThreads
                        # rejects outright).
                        local nrej_scan = reuse_scan ?
                            _refine_scan_dispersion!(
                                θ, keyed, geom, ev, ps_delay_plan, disp_plan, ref_ant, nant;
                                inner = inner, ties = ties, tau_max = _DTEC_POLISH_TAU, dtec_max = bandpass_polish_dtec,
                            ) :
                            _refine_scan_dispersion!(
                                θ, keyed, geom, ev, ps_delay_plan, disp_plan, ref_ant, nant;
                                inner = inner, ties = ties,
                            )
                        Threads.atomic_add!(dtec_rejected, nrej_scan)
                    end
                    # Per-scan band-group SBD AFTER the dispersion refinement (so
                    # the within-band slopes it fits are dispersion-corrected). SBD
                    # is cheap — always full-refine.
                    sbd_plans === nothing ||
                        _refine_scan_sbd!(θ, keyed, geom, ev, sbd_plans, ref_ant, nant; inner = inner)
                    _adhoc_group_leaves!(θ, keyed, geom, ev, adhoc_plan, adhoc, ref_ant, nant; shared_feeds = adhoc_shared, inner = inner, excl = excl, psI = psI)
                    tc = time_ns()
                    scan_t_decode2[gi] = (tb - ta) / 1.0e9
                    scan_t_adhoc[gi] = (tc - tb) / 1.0e9
                    _progress_notify(progress, :adhoc, Threads.atomic_add!(nadh, 1) + 1, ngroups)
                    nothing
                end
                t3w = time_ns()
                t_search_pass = (t1w - t0w) / 1.0e9
                t_bandpass_stage = (t2w - t1w) / 1.0e9
                t_adhoc_pass = (t3w - t2w) / 1.0e9
                peak_par = max(pk1, pk2)
                (ch, nc, nrj)
            end
        end
    end

    info = (;
        nant = nant,
        nscan = ngroups,
        scan_max_snr = scan_snr,
        scan_ncells = scan_ncells,
        scan_chi = fill(chi, ngroups),
        scan_ncomp = fill(ncomp, ngroups),
        stageB_rejected = nrej,
        # EHT-HOPS-style flags: (station, geometry-scan-id) pairs that
        # participate in a scan but were left unconstrained by the surviving
        # stage-B rows (identity gains — see `_search_and_stationize!`), plus
        # the intra-site baselines excluded for crosstalk. `apply_calibration`
        # zero-weights both (kwarg `apply_flags`).
        _flag_info(station_flags, excl)...,
        dispersion_applied = disp_plan !== nothing,
        sbd_applied = sbd_plans !== nothing,
        dtec_rejected = dtec_rejected[],
        ant_names = String.(collect(UVData.metadata(first_leaf).antennas.name)),
        search = search,
        precal_applied = precal !== nothing,
        # Peak OBSERVED group concurrency under the budget scheduler (the old
        # fixed `_bounded_ntasks` cap no longer bounds the group loops).
        ntasks_used = peak_par,
        inner_tasks = inner,
        t_search_pass = t_search_pass,
        t_bandpass_stage = t_bandpass_stage,
        t_adhoc_pass = t_adhoc_pass,
        scan_t_decode = scan_t_decode,
        scan_t_search = scan_t_search,
        scan_t_decode2 = scan_t_decode2,
        scan_t_adhoc = scan_t_adhoc,
        _flatten_detections(scan_dets)...,
    )
    return CalibrationSolution(model, layout, geom, θ, info)
end

"""
    solve_and_reduce_fringes(uvset; postprocess, search, adhoc, rounds, ref_ant, ntasks)
        -> (sol::CalibrationSolution, output::UVSet)

Fringe fit **and** correction/reduction. The stage-B solve ties the feeds with a
track-wide R–L offset (so it is global, not per-scan); the data is therefore read
in two threaded passes: (1) search every group → one global stationization; (2)
per group, adhoc-phase → gain-correct → run through `postprocess` (a `UVSet ->
UVSet` map, default `identity`) → free. `postprocess` runs while the corrected
group is resident (no extra I/O), so the natural use is post-fringe reduction,
e.g.

    postprocess = uv -> combine_spw(time_bin_average(frequency_average(uv; nout = 1), 2.0))

`output` is a full `UVSet` mirroring the input tree with every leaf corrected and
reduced; `sol` is the same solution `solve_fringes` would return.

`ntasks` caps how many groups are processed concurrently; actual admission is
per group, by ITS OWN estimated peak bytes against a DETERMINISTIC memory
budget — `mem_budget` (absolute bytes) if given, else `mem_fraction` of TOTAL
physical RAM (a machine constant, so scheduling is reproducible across runs) —
via `_budget_scheduled_map`: big scans run (nearly) alone while small scans
pack the leftover budget, instead of every scan being serialized by the largest
one. The bulk reader holds a whole scan's row span plus its vis cube at once
(several GB on a real file), so an unbudgeted thread-count default would OOM.
On a shared/loaded box leave headroom: lower `mem_fraction` (or pass an
explicit `mem_budget`); raise it on a dedicated box.
"""
function solve_and_reduce_fringes(
        uvset::UVSet;
        postprocess = identity,
        search::FringeSearch = FringeSearch(),
        adhoc::AbstractAdhocSmoother = SavitzkyGolaySmoother(),
        rounds::Int = 1,
        ref_ant::Integer = 1,
        ntasks::Integer = Threads.nthreads(),
        mem_fraction::Real = 0.6,
        mem_budget = nothing,
        phase_bandpass::Bool = true,
        amp_bandpass::Bool = true,
        amp_smoother::AbstractBandpassSmoother = PenalizedBandpass(1.0),
        bandpass_source = nothing,
        precal::Union{Nothing, CalibrationSolution} = nothing,
        flag_channels = nothing,
        progress = nothing,
        bandpass_max_scans::Integer = 0,
        dispersion = :auto,
        sbd = :auto,
        dtec_tie_colocated::Bool = true,
        exclude_colocated::Bool = true,
        adhoc_pseudo_stokes = :auto,
        reuse_bandpass_refine::Bool = true,
        bandpass_polish_dtec::Real = 20.0,
    )
    geom = build_geometry(uvset)
    model = _fringe_model(
        dispersion = _dispersion_enabled(dispersion, geom),
        sbd_bands = _sbd_bands(sbd, geom),
    )
    first_leaf = first(values(UVData.branches(uvset)))
    nant = length(UVData.metadata(first_leaf).antennas)
    layout = plan_parameters(model, nant, geom)
    ev = GainEvaluator(model, layout)
    stageB = _stageB_components(model, layout)
    adhoc_plan = _adhoc_plan(model, layout)
    adhoc_shared = _adhoc_shared(model)
    bp_plan = _bandpass_plan(model, layout)
    amp_bp_plan = _amp_bandpass_plan(model, layout)
    disp_plan = _dispersion_plan(model, layout)
    ps_delay_plan = _perscan_delay_plan(model, layout)
    sbd_plans = _sbd_plans(model, layout)
    ties = dtec_tie_colocated ? _colocated_ties(UVData.metadata(first_leaf).antennas) : nothing
    excl = if exclude_colocated
        s = _colocated_pair_set(UVData.metadata(first_leaf).antennas)
        isempty(s) ? nothing : s
    else
        nothing
    end
    psI = _pseudo_stokes_config(
        adhoc_pseudo_stokes, first_leaf, UVData.DimensionalData.metadata(uvset),
    )
    dtec_rejected = Threads.Atomic{Int}(0)
    f0 = geom.f0
    t0_sec = geom.t0 * 3600.0
    pc = _make_precal(precal, flag_channels, geom)

    group_leaves = _scan_group_leaves(uvset)
    group_sources = _group_sources(group_leaves)
    ngroups = length(group_leaves)
    scan_snr = zeros(ngroups)
    scan_ncells = zeros(ngroups)
    scan_dets = [_DetRow[] for _ in 1:ngroups]
    scan_t_decode = zeros(ngroups)
    scan_t_search = zeros(ngroups)
    scan_t_decode2 = zeros(ngroups)
    scan_t_adhoc = zeros(ngroups)
    scan_t_reduce = zeros(ngroups)
    θ = zeros(layout.nθ)
    charges = [_group_peak_bytes(g) for g in group_leaves]
    budget = _memory_budget(mem_fraction, mem_budget)
    ntasks_use = _bounded_ntasks(group_leaves, ntasks, ngroups; mem_fraction = mem_fraction, mem_budget = mem_budget)
    # See solve_fringes: leftover threads run the per-baseline searches; FFT plans
    # stay single-threaded.
    inner = max(1, Threads.nthreads() ÷ ntasks_use)
    bp_prefetch = ntasks_use == 1 && 1.6 * maximum(charges; init = 0) <= budget
    ws_pool = _ws_pool(Threads.nthreads())
    t_search_pass = 0.0
    t_bandpass_stage = 0.0
    t_adhoc_pass = 0.0
    peak_par = 0
    station_flags = Tuple{Int, Int}[]

    out_pairs, chi, ncomp, nrej = _with_decode_threads(inner) do
        _with_fft_threads(1) do
            _with_single_blas_thread() do
                t0w = time_ns()
                ch, nc, nrj, pk1, station_flags = _search_and_stationize!(
                    θ, scan_snr, scan_ncells, scan_dets, scan_t_decode, scan_t_search,
                    group_leaves, geom, ev, stageB, f0, t0_sec,
                    search, rounds, ref_ant, charges, budget, ntasks, ws_pool, inner, pc, progress,
                )
                t1w = time_ns()
                # Phase + amplitude bandpass from the brightest calibrator (time-
                # stable), applied to every group's correction below via the full θ.
                # `refined_bp` = scans whose per-scan dTEC/SBD the stage already fit;
                # pass 2 reuses them (see `reuse_bandpass_refine`).
                refined_bp = Set{Int}()
                if (phase_bandpass && bp_plan !== nothing) || (amp_bandpass && amp_bp_plan !== nothing)
                    cal = _bandpass_calibrator(bandpass_source, group_sources, scan_snr)
                    refined_bp = Set(
                        _solve_bandpass_stage!(
                            θ, group_leaves, group_sources, cal, geom, ev,
                            phase_bandpass ? bp_plan : nothing, nant;
                            amp_plan = amp_bandpass ? amp_bp_plan : nothing, ref_ant = ref_ant,
                            amp_smoother = amp_smoother, pc = pc, progress = progress,
                            max_scans = bandpass_max_scans, scan_snr = scan_snr,
                            ntasks = ntasks_use, inner = inner, prefetch = bp_prefetch,
                            disp_plan = disp_plan, ps_delay_plan = ps_delay_plan,
                            dtec_rejected = dtec_rejected, sbd_plans = sbd_plans, ties = ties,
                            excl = excl,
                        ),
                    )
                end
                t2w = time_ns()
                # Pass 2: per group, adhoc → correct → reduce. θ is fully populated for
                # this group (global stage-B + this group's just-written adhoc slots), so
                # the group-local solution corrects identically to the global one. The
                # precal (if any) was divided into the leaves at materialization, so the
                # reduced OUTPUT carries precal ∘ solution.
                nadh = Threads.Atomic{Int}(0)
                _progress_notify(progress, :adhoc, 0, ngroups)
                results, pk2 = _budget_scheduled_map(1:ngroups, charges, budget; max_tasks = ntasks) do gi
                    ta = time_ns()
                    keyed = _materialize_leaf_group(group_leaves[gi], geom, pc, inner)
                    tb = time_ns()
                    # The bandpass stage already fit this scan's per-scan (Δτ, dTEC)
                    # + SBD θ (and the bandpass is built dispersion-corrected, so a
                    # re-fit here would recover ~zero) — reuse it instead of paying
                    # the stage's dominant cost twice.
                    reuse_scan = reuse_bandpass_refine && gi in refined_bp
                    # Per-scan (Δτ, dTEC) refinement BEFORE the adhoc solve (see
                    # `_refine_scan_dispersion!`); the reduce below then applies
                    # the dispersion-corrected θ. Scans the bandpass stage already
                    # fit are POLISHED in a narrow window (cheap) rather than re-run
                    # with the full grid search — keeps the weak-source increment a
                    # full skip dropped.
                    if disp_plan !== nothing
                        # `local`: an unannotated `nrj` would capture-and-rebind the
                        # outer stage-B rejection count (a boxed capture OhMyThreads
                        # rejects outright).
                        local nrej_scan = reuse_scan ?
                            _refine_scan_dispersion!(
                                θ, keyed, geom, ev, ps_delay_plan, disp_plan, ref_ant, nant;
                                inner = inner, ties = ties, tau_max = _DTEC_POLISH_TAU, dtec_max = bandpass_polish_dtec,
                            ) :
                            _refine_scan_dispersion!(
                                θ, keyed, geom, ev, ps_delay_plan, disp_plan, ref_ant, nant;
                                inner = inner, ties = ties,
                            )
                        Threads.atomic_add!(dtec_rejected, nrej_scan)
                    end
                    # Per-scan band-group SBD AFTER the dispersion refinement (so
                    # the within-band slopes it fits are dispersion-corrected). SBD
                    # is cheap (~4% of the stage) — always full-refine.
                    sbd_plans === nothing ||
                        _refine_scan_sbd!(θ, keyed, geom, ev, sbd_plans, ref_ant, nant; inner = inner)
                    _adhoc_group_leaves!(θ, keyed, geom, ev, adhoc_plan, adhoc, ref_ant, nant; shared_feeds = adhoc_shared, inner = inner, excl = excl, psI = psI)
                    tc = time_ns()
                    sub_branches = DimensionalData.TreeDict()
                    for (k, leaf) in keyed
                        sub_branches[k] = leaf
                    end
                    sub = DimensionalData.rebuild(uvset; branches = sub_branches)
                    # Carry the flag block so the streamed correction zero-weights
                    # unconstrained stations / excluded baselines exactly like a
                    # full-set `apply_calibration` would.
                    sol_local = CalibrationSolution(model, layout, geom, θ, _flag_info(station_flags, excl))
                    # This runs INSIDE the per-scan-group parallel map; the gain
                    # kernel must stay within this group's nested `inner` budget
                    # (the public default threads over all cores → oversubscribe).
                    reduced = postprocess(UVData.apply_calibration(sub, sol_local; ntasks = inner))
                    out = collect(pairs(UVData.branches(reduced)))
                    td = time_ns()
                    scan_t_decode2[gi] = (tb - ta) / 1.0e9
                    scan_t_adhoc[gi] = (tc - tb) / 1.0e9
                    scan_t_reduce[gi] = (td - tc) / 1.0e9
                    _progress_notify(progress, :adhoc, Threads.atomic_add!(nadh, 1) + 1, ngroups)
                    out
                end
                t3w = time_ns()
                t_search_pass = (t1w - t0w) / 1.0e9
                t_bandpass_stage = (t2w - t1w) / 1.0e9
                t_adhoc_pass = (t3w - t2w) / 1.0e9
                peak_par = max(pk1, pk2)
                (results, ch, nc, nrj)
            end
        end
    end

    out_branches = DimensionalData.TreeDict()
    for r in out_pairs
        for (k, leaf) in r
            out_branches[k] = leaf
        end
    end
    output = DimensionalData.rebuild(uvset; branches = out_branches)

    info = (;
        nant = nant,
        nscan = ngroups,
        scan_max_snr = scan_snr,
        scan_ncells = scan_ncells,
        scan_chi = fill(chi, ngroups),
        scan_ncomp = fill(ncomp, ngroups),
        stageB_rejected = nrej,
        # EHT-HOPS-style flags: (station, geometry-scan-id) pairs that
        # participate in a scan but were left unconstrained by the surviving
        # stage-B rows (identity gains — see `_search_and_stationize!`), plus
        # the intra-site baselines excluded for crosstalk. `apply_calibration`
        # zero-weights both (kwarg `apply_flags`).
        _flag_info(station_flags, excl)...,
        dispersion_applied = disp_plan !== nothing,
        sbd_applied = sbd_plans !== nothing,
        dtec_rejected = dtec_rejected[],
        ant_names = String.(collect(UVData.metadata(first_leaf).antennas.name)),
        search = search,
        precal_applied = precal !== nothing,
        # Peak OBSERVED group concurrency under the budget scheduler (the old
        # fixed `_bounded_ntasks` cap no longer bounds the group loops).
        ntasks_used = peak_par,
        inner_tasks = inner,
        t_search_pass = t_search_pass,
        t_bandpass_stage = t_bandpass_stage,
        t_adhoc_pass = t_adhoc_pass,
        scan_t_decode = scan_t_decode,
        scan_t_search = scan_t_search,
        scan_t_decode2 = scan_t_decode2,
        scan_t_adhoc = scan_t_adhoc,
        scan_t_reduce = scan_t_reduce,
        _flatten_detections(scan_dets)...,
    )
    sol = CalibrationSolution(model, layout, geom, θ, info)
    return sol, output
end

# ── Phase bandpass (HOPS-style passband): per-channel station phase, time-stable ──
#
# Accumulate one group's contribution to the per-(global-baseline, product, GLOBAL
# channel) coherent residual `rbar_bp` (and weight `wbar_bp`), for the phase-
# bandpass solve. The residual is V / (stage-B gains); BEFORE summing over time we
# counter-rotate each AP by its OWN band-averaged residual phase, removing the
# per-AP time phase (residual rate/drift, and what the adhoc would later remove)
# so the time-average is coherent and isolates the per-channel SHAPE. `blidx` maps
# `(a, b) -> row` in the global baseline table. Mirrors the adhoc accumulation but
# collapses over time per channel instead of over frequency per AP.
function _accumulate_bandpass_rbar!(rbar_bp, wbar_bp, blidx, ev::GainEvaluator, θ::AbstractVector, grp::_ScanGroup)
    g = evaluate_gains(ev, θ, grp.g_ci, grp.g_ti)    # (nchan, nti, nant, 2)
    nchan, nti, nbl, npol = size(grp.Vg)
    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(grp.pol_products[p])
        for bi in 1:nbl
            a, b = grp.bl_pairs[bi]
            a == b && continue
            idx = get(blidx, (a, b), 0)
            idx == 0 && continue
            for tt in 1:nti
                # Band-averaged residual phase for this AP (the per-AP time phase).
                # Weights carry the |den|² inverse-variance factor of the corrected
                # data (see `_accumulate_leaf_rbar!`; a no-op while the gains here
                # are phase-only — the amp bandpass is solved after this stage).
                acc = zero(ComplexF64)
                for c in 1:nchan
                    w = grp.Wg[c, tt, bi, p]
                    (w > 0 && isfinite(w)) || continue
                    ga = g[c, tt, a, fa]; gb = g[c, tt, b, fb]
                    den = ga * conj(gb)
                    (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(den)) || continue
                    v = grp.Vg[c, tt, bi, p] / den
                    isfinite(v) && (acc += w * abs2(den) * v)
                end
                abs(acc) > 0 || continue
                rot = conj(acc) / abs(acc)            # cis(-angle(acc)): de-rotate this AP
                for c in 1:nchan
                    w = grp.Wg[c, tt, bi, p]
                    (w > 0 && isfinite(w)) || continue
                    ga = g[c, tt, a, fa]; gb = g[c, tt, b, fb]
                    den = ga * conj(gb)
                    (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(den)) || continue
                    v = grp.Vg[c, tt, bi, p] / den
                    isfinite(v) || continue
                    gc = grp.g_ci[c]
                    wd = w * abs2(den)
                    rbar_bp[idx, p, gc] += wd * v * rot
                    wbar_bp[idx, p, gc] += wd
                end
            end
        end
    end
    return rbar_bp, wbar_bp
end

# Solve the per-(station, feed) phase bandpass from the accumulated per-channel
# residual and write it into `θ`'s `PerChannel` slots. For each global channel,
# the globally-closing per-feed phase is solved exactly like the adhoc (reusing
# `_solve_observable` over the (station, feed) graph) with the same scale-invariant
# SNR gate (`_track_noise2`, robust to uncalibrated weights). Each (station, feed)
# track is then referenced to its CIRCULAR-mean phase over channels (zero net phase
# ⇒ does not alias the Stage-B constant phase). No cross-channel unwrap/smoothing —
# the per-channel phase is applied as `cis(φ)`, for which wrapping is irrelevant,
# and unwrapping across the sub-band gaps would be unsafe.
function _solve_phase_bandpass!(
        θ, rbar_bp, wbar_bp, bl_pairs, pol_products, nant, plan;
        ref_ant::Integer = 1, snr_floor::Real = 1.0,
    )
    nbl, npol, nchan = size(rbar_bp)
    feeds = [correlation_feed_pair(p) for p in pol_products]
    noise2 = [_track_noise2(rbar_bp, wbar_bp, bi, p, nchan) for bi in 1:nbl, p in 1:npol]
    phase = fill(NaN, nant, 2, nchan)
    for gc in 1:nchan
        rows = _ObsRow[]
        for bi in 1:nbl, p in 1:npol
            a, b = bl_pairs[bi]
            a == b && continue
            r = rbar_bp[bi, p, gc]; w = wbar_bp[bi, p, gc]
            (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
            n2 = noise2[bi, p]
            snr2 = isfinite(n2) && n2 > 0 ? abs2(r / w) / n2 : abs2(r) / w
            snr2 >= snr_floor^2 || continue
            fa, fb = feeds[p]
            push!(rows, _ObsRow(a, b, fa, fb, angle(r), snr2, _chi_sign(fa, fb)))
        end
        ph, _, _, _ = _solve_observable(rows, nant, ref_ant; use_chi = true, rewrap = 0)
        phase[:, :, gc] .= ph
    end

    # Circular-mean reference per (station, feed) → zero net applied phase (gauge).
    for a in 1:nant, f in 1:2
        acc = zero(ComplexF64)
        @inbounds for gc in 1:nchan
            v = phase[a, f, gc]
            isfinite(v) && (acc += cis(v))
        end
        abs(acc) > 0 || continue
        m = angle(acc)
        @inbounds for gc in 1:nchan
            v = phase[a, f, gc]
            isfinite(v) || continue
            off = plan.off1[a, f, 1, 1]
            off == 0 && continue
            θ[off + plan.clocal[gc] - 1] = rem2pi(v - m, RoundNearest)
        end
    end
    return θ
end

# `_fit_bandpass_segment(smoother, …)` implementations for the amplitude-bandpass
# smoother types (defined above `solve_fringes`). Each takes ONE spw's gated closure
# observations — `na`/`nb` node indices, `ci` local-channel index, `val = log|V̄|`,
# `w = SNR²`, `xseg` the centred/scaled in-spw frequency coordinate — and returns
# `la_seg::Matrix` (nnodes × nchan_seg, `NaN` where unestimable).

# Bucket observation indices by their local channel.
function _bandpass_obs_by_channel(ci, nseg)
    byc = [Int[] for _ in 1:nseg]
    for i in eachindex(ci)
        push!(byc[ci[i]], i)
    end
    return byc
end

# Free per-channel closure: independent signless-Laplacian WLS per channel.
function _fit_bandpass_segment(::FreeBandpass, na, nb, ci, val, w, nnodes, nseg, xseg, ridge)
    la = fill(NaN, nnodes, nseg)
    pen = fill(float(ridge), nnodes)
    for (c, idx) in enumerate(_bandpass_obs_by_channel(ci, nseg))
        isempty(idx) && continue
        A = zeros(length(idx), nnodes)
        touched = falses(nnodes)
        for (r, i) in enumerate(idx)
            A[r, na[i]] += 1.0; A[r, nb[i]] += 1.0
            touched[na[i]] = true; touched[nb[i]] = true
        end
        sol = weighted_regularized_least_squares(A, val[idx], w[idx], pen)
        for node in 1:nnodes
            touched[node] && (la[node, c] = sol[node])
        end
    end
    return la
end

# Per-spw polynomial: one closure WLS over `nb = degree+1` coefficients per node;
# θ-column for (node, k) is `(node-1)*nb + k`. Evaluated at every channel (incl. gaps).
function _fit_bandpass_segment(sm::PolynomialBandpass, na, nb, ci, val, w, nnodes, nseg, xseg, ridge)
    deg = clamp(sm.degree, 1, max(1, nseg - 1))
    nbf = deg + 1
    B = Float64[xseg[c]^k for c in 1:nseg, k in 0:deg]      # nseg × nbf basis
    ncol = nnodes * nbf
    A = zeros(length(val), ncol)
    touched = falses(nnodes)
    @inbounds for i in eachindex(val)
        oa = (na[i] - 1) * nbf; ob = (nb[i] - 1) * nbf
        for k in 1:nbf
            A[i, oa + k] += B[ci[i], k]; A[i, ob + k] += B[ci[i], k]
        end
        touched[na[i]] = true; touched[nb[i]] = true
    end
    coef = weighted_regularized_least_squares(A, val, w, fill(float(ridge), ncol))
    la = fill(NaN, nnodes, nseg)
    for node in 1:nnodes
        touched[node] || continue
        c0 = (node - 1) * nbf
        for c in 1:nseg
            v = 0.0
            @inbounds for k in 1:nbf
                v += coef[c0 + k] * B[c, k]
            end
            la[node, c] = v
        end
    end
    return la
end

# Roughness-penalised: free per-channel closure, then a per-node Whittaker
# (2nd-difference) penalised WLS across channels — interpolating gated channels via
# the penalty, following the data elsewhere.
function _fit_bandpass_segment(sm::PenalizedBandpass, na, nb, ci, val, w, nnodes, nseg, xseg, ridge)
    la0 = _fit_bandpass_segment(FreeBandpass(), na, nb, ci, val, w, nnodes, nseg, xseg, ridge)
    (sm.lambda <= 0 || nseg < 3) && return la0
    prec = zeros(nnodes, nseg)                              # per-(node, channel) precision Σ w
    for i in eachindex(val)
        prec[na[i], ci[i]] += w[i]; prec[nb[i], ci[i]] += w[i]
    end
    la = copy(la0)
    for node in 1:nnodes
        any(>(0), @view prec[node, :]) || continue
        la[node, :] .= _whittaker_smooth(view(la0, node, :), view(prec, node, :), sm.lambda, ridge)
    end
    return la
end

# 1-D Whittaker smoother: minimise  Σ w_i (x_i − y_i)² + (λ·w̄) Σ (x_{i−1} − 2x_i + x_{i+1})².
# `w_i = 0` (and `y_i` non-finite) where a channel had no data → the penalty alone
# sets it (interpolation). `λ` is scaled by the median positive weight so it is
# data-relative. Dense pentadiagonal normal-matrix solve (no SparseArrays).
function _whittaker_smooth(y, w, lambda::Real, ridge::Real)
    n = length(y)
    pos = [w[i] for i in 1:n if w[i] > 0]
    λ = lambda * (isempty(pos) ? 1.0 : median(pos))
    M = zeros(n, n); rhs = zeros(n)
    @inbounds for i in 1:n
        wi = (w[i] > 0 && isfinite(y[i])) ? float(w[i]) : 0.0
        M[i, i] += wi + ridge
        rhs[i] += wi * (wi > 0 ? y[i] : 0.0)
    end
    @inbounds for i in 1:(n - 2)                            # 2nd-difference rows [1, −2, 1]
        c = (i, i + 1, i + 2); s = (1.0, -2.0, 1.0)
        for a in 1:3, b in 1:3
            M[c[a], c[b]] += λ * s[a] * s[b]
        end
    end
    return M \ rhs
end

# Solve the per-(station, feed) AMPLITUDE bandpass (log-amp) from the accumulated
# residual and write it into `θ`'s log-amp `PerChannel` slots — flattening the
# per-station instrumental frequency response (the filterbank passband). Gathers the
# gated closure observations PER SPW (so an estimator never crosses a sub-band gap)
# and hands them to `smoother` (an `AbstractBandpassSmoother` — see above). The
# scale-invariant SNR gate (`_track_noise2`) drops no-signal channels from the fit
# (then estimated, or not, per the smoother); a spw with no signal stays |g| = 1.
# A final zero-band-mean gauge per (station, feed) keeps the bandpass to SHAPE only —
# the absolute level is the a-priori amplitude cal's job.
function _solve_amp_bandpass!(
        θ, rbar_bp, wbar_bp, bl_pairs, pol_products, nant, plan, channel_freqs;
        snr_floor::Real = 1.0, ridge::Real = 1.0e-6,
        spw_of_chan::AbstractVector{<:Integer} = Int[],
        smoother::AbstractBandpassSmoother = PenalizedBandpass(1.0),
        max_logamp::Real = log(10.0),
        spike_sigma::Real = 5.0,
    )
    nbl, npol, nchan = size(rbar_bp)
    feeds = [correlation_feed_pair(p) for p in pol_products]
    noise2 = [_track_noise2(rbar_bp, wbar_bp, bi, p, nchan) for bi in 1:nbl, p in 1:npol]
    nnodes = 2 * nant
    soc = isempty(spw_of_chan) ? ones(Int, nchan) : collect(spw_of_chan)
    la = fill(NaN, nant, 2, nchan)

    for bnd in sort(unique(soc))
        chans = [gc for gc in 1:nchan if soc[gc] == bnd]
        nseg = length(chans); nseg == 0 && continue
        fs = Float64[channel_freqs[gc] for gc in chans]
        center = sum(fs) / nseg
        scale = maximum(abs.(fs .- center)); scale = scale > 0 ? scale : 1.0
        xseg = [(fs[ci] - center) / scale for ci in 1:nseg]

        # Gated closure observations for this spw (local channel index `ci`).
        na = Int[]; nbn = Int[]; cii = Int[]; vals = Float64[]; wts = Float64[]
        for (ci, gc) in enumerate(chans), bi in 1:nbl, p in 1:npol
            a, b = bl_pairs[bi]
            a == b && continue
            r = rbar_bp[bi, p, gc]; w = wbar_bp[bi, p, gc]
            (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
            nz = noise2[bi, p]
            snr2 = isfinite(nz) && nz > 0 ? abs2(r / w) / nz : abs2(r) / w
            snr2 >= snr_floor^2 || continue
            amp = abs(r / w); amp > 0 || continue
            fa, fb = feeds[p]
            push!(na, _node(a, fa, nant)); push!(nbn, _node(b, fb, nant)); push!(cii, ci)
            push!(vals, log(amp)); push!(wts, snr2)
        end
        isempty(vals) && continue
        la_seg = _fit_bandpass_segment(smoother, na, nbn, cii, vals, wts, nnodes, nseg, xseg, ridge)
        for node in 1:nnodes
            ant = (node - 1) % nant + 1; feed = (node - 1) ÷ nant + 1
            for ci in 1:nseg
                v = la_seg[node, ci]
                isfinite(v) && (la[ant, feed, chans[ci]] = v)
            end
        end
    end

    # Narrow-spike guard: additive contamination (pcal tones, RFI) violates the
    # multiplicative gain model — a contaminated channel shows EXCESS amplitude,
    # the fit hands it |g| > 1, and `apply_calibration` would then UP-weight it
    # (w → w·|g|²), amplifying exactly the channels that should be distrusted.
    # Genuine passband structure is smooth or negative (roll-off), so narrow
    # POSITIVE log-amp outliers vs the per-(station, feed, spw) robust scale are
    # excised (left unapplied, |g| = 1) instead of trusted. `spike_sigma = 0`
    # disables the guard.
    if spike_sigma > 0
        for a in 1:nant, f in 1:2, bnd in sort(unique(soc))
            chans = [gc for gc in 1:nchan if soc[gc] == bnd]
            v = [la[a, f, gc] for gc in chans if isfinite(la[a, f, gc])]
            length(v) >= 8 || continue
            med = median(v)
            s = 1.4826 * median(abs.(v .- med))
            cut = spike_sigma * max(s, 0.02)
            for gc in chans
                isfinite(la[a, f, gc]) || continue
                la[a, f, gc] - med > cut && (la[a, f, gc] = NaN)
            end
        end
    end

    # Zero band-mean log-amp gauge per (station, feed) — SHAPE only. Write the slots.
    for a in 1:nant, f in 1:2
        acc = 0.0; n = 0
        @inbounds for gc in 1:nchan
            v = la[a, f, gc]
            isfinite(v) && (acc += v; n += 1)
        end
        n == 0 && continue
        m = acc / n
        off = plan.off1[a, f, 1, 1]
        off == 0 && continue
        @inbounds for gc in 1:nchan
            v = la[a, f, gc]
            isfinite(v) || continue
            val = v - m
            # Leave implausibly-large corrections UNAPPLIED (|g| = 1). A smoother (the
            # default) interpolates gaps and self-regularizes, but `FreeBandpass` (or a
            # near-zero `lambda`) can hand a low-SNR band-edge channel that barely
            # clears the gate a huge log-amp; applying it would up-weight that channel's
            # noise, since `apply_calibration` scales weights by |g|². The bound is
            # generous (|g| ≤ 10) so real passband roll-off/structure passes unchanged —
            # only pathological noise blow-ups are gated.
            θ[off + plan.clocal[gc] - 1] = abs(val) > max_logamp ? 0.0 : val
        end
    end
    return θ
end

# Bandpass stage: solve the phase bandpass from the calibrator source and write
# it into `θ`. Accumulates the per-channel residual over that source's scans
# (after the global Stage-B) PLUS a coverage top-up — the best scan of any other
# source needed to reach stations the calibrator never observed — then one
# closing per-channel solve. Time-stable, so it applies to ALL scans via
# `GlobalTime`.
#
# When the model carries a dispersion term (`disp_plan`), each accumulated scan
# is (Δτ, dTEC)-refined FIRST (`_refine_scan_dispersion!` on its leaves, exactly
# as pass 2 does). Without this, scans with different ionospheres decohere the
# accumulation: stage-B removes only the linear-in-f part of each scan's
# dispersion, and the leftover 1/ν curvature differs scan-to-scan by
# K·ΔdTEC·(1/f0 − 1/ν) — ~1.6 rad per TECU of scan spread at 3 GHz (vs 6.8 GHz
# f0), so a ±5 TECU night WIPES OUT the low band's phase bandpass and the
# corrupted curve then DEGRADES every scan it is applied to. The per-scan
# refinement re-runs in pass 2 on the then-bandpass-corrected residual (θ
# increments compose), so scan dTEC stays measured against the CLEAN curve.
#
# `max_scans > 0` caps the accumulation to that many HIGHEST-SNR calibrator scans
# (needs `scan_snr`): the bandpass is time-stable, so a few strong scans carry
# essentially all the information, and on a single-calibrator-dominated file the
# uncapped stage is close to a full extra read of the data. The scans accumulate
# on `ntasks` concurrent tasks with chunk-local accumulators summed in chunk
# order (deterministic for a fixed task count; differs from the serial sum only
# by float rounding).
function _solve_bandpass_stage!(
        θ, group_leaves, group_sources, cal_source, geom, ev, plan, nant;
        amp_plan = nothing, ref_ant::Integer = 1, snr_floor::Real = 1.0,
        amp_smoother::AbstractBandpassSmoother = PenalizedBandpass(1.0),
        pc = nothing, progress = nothing,
        max_scans::Integer = 0, scan_snr = nothing, ntasks::Integer = 1, inner::Integer = 1,
        prefetch::Bool = false,
        disp_plan = nothing, ps_delay_plan = nothing, dtec_rejected = nothing,
        sbd_plans = nothing, ties = nothing, excl = nothing,
    )
    nchan = length(geom.channel_freqs)
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    # Co-located baselines never enter the accumulation (see `_colocated_pair_set`);
    # their rbar tracks stay zero and the per-channel SNR gates skip them.
    blidx = Dict(
        bl_pairs[i] => i for i in eachindex(bl_pairs)
            if excl === nothing || bl_pairs[i] ∉ excl
    )
    cal_gis = cal_source === :all ? collect(eachindex(group_leaves)) :
        [gi for gi in eachindex(group_leaves) if group_sources[gi] == cal_source]
    isempty(cal_gis) && return Int[]           # calibrator absent → leave bandpass at 0
    if max_scans > 0 && scan_snr !== nothing && length(cal_gis) > max_scans
        ord = sortperm([isfinite(scan_snr[gi]) ? scan_snr[gi] : -Inf for gi in cal_gis]; rev = true)
        cal_gis = sort(cal_gis[ord[1:Int(max_scans)]])
    end
    # Stations the calibrator never observed would be left with NO bandpass —
    # top up with the best scan of whatever source covers each of them.
    if scan_snr !== nothing
        cal_gis = sort(vcat(cal_gis, _bandpass_coverage_topup(cal_gis, group_leaves, scan_snr)))
    end
    pols = String.(pol_products(last(first(group_leaves[cal_gis[1]]))))
    _progress_notify(progress, :bandpass, 0, length(cal_gis))
    ndone = Threads.Atomic{Int}(0)
    chunks = collect(Iterators.partition(cal_gis, cld(length(cal_gis), clamp(Int(ntasks), 1, length(cal_gis)))))
    parts = map(chunks) do chunk
        Threads.@spawn begin
            rl = zeros(ComplexF64, length(bl_pairs), length(pols), nchan)
            wl = zeros(Float64, length(bl_pairs), length(pols), nchan)
            # Depth-1 prefetch: decode scan N+1 while scan N is refined and
            # accumulated. The ACCUMULATION ORDER is unchanged (still the chunk's
            # scan order), so the folded rbar/wbar are bit-identical to the
            # sequential loop — this only hides the decode I/O behind compute.
            # The caller enables it only when the budget can hold the one extra
            # resident group (serialized single-chunk case).
            nxt = prefetch ? Threads.@spawn(_materialize_concat_group(group_leaves[$(first(chunk))], geom, pc, inner)) : nothing
            td = 0.0; tp = 0.0; tsb = 0.0; tac = 0.0   # per-op timers (profiling)
            for (ci, gi) in enumerate(chunk)
                td += @elapsed grp = nxt === nothing ?
                    _materialize_concat_group(group_leaves[gi], geom, pc, inner) :
                    fetch(nxt)::_ScanGroup
                if prefetch && ci < length(chunk)
                    local ginext = chunk[ci + 1]
                    nxt = Threads.@spawn _materialize_concat_group(group_leaves[$ginext], geom, pc, inner)
                end
                if disp_plan !== nothing
                    # Dispersion-correct THIS scan before accumulating (see the
                    # stage comment). Writes only this scan's per-scan θ columns,
                    # so concurrent chunk tasks stay disjoint.
                    tp += @elapsed begin
                        local nrej_scan = _refine_scan_dispersion!(
                            θ, grp, geom, ev, ps_delay_plan, disp_plan, ref_ant, nant;
                            inner = inner, ties = ties,
                        )
                        dtec_rejected === nothing || Threads.atomic_add!(dtec_rejected, nrej_scan)
                    end
                end
                # SBD-correct THIS scan likewise (see `_refine_scan_sbd!`) — the
                # frozen bandpass must not average per-scan within-band slopes.
                tsb += @elapsed (
                    sbd_plans === nothing ||
                        _refine_scan_sbd!(θ, grp, geom, ev, sbd_plans, ref_ant, nant; inner = inner)
                )
                tac += @elapsed _accumulate_bandpass_rbar!(rl, wl, blidx, ev, θ, grp)
                _progress_notify(progress, :bandpass, Threads.atomic_add!(ndone, 1) + 1, length(cal_gis))
            end
            (rl, wl, td, tp, tsb, tac)
        end
    end
    rbar_bp = zeros(ComplexF64, length(bl_pairs), length(pols), nchan)
    wbar_bp = zeros(Float64, length(bl_pairs), length(pols), nchan)
    Td = 0.0; Tp = 0.0; Tsb = 0.0; Tac = 0.0
    for t in parts
        rl, wl, td, tp, tsb, tac = fetch(t)
        rbar_bp .+= rl
        wbar_bp .+= wl
        Td += td; Tp += tp; Tsb += tsb; Tac += tac
    end
    tsolve = @elapsed begin
        plan === nothing ||
            _solve_phase_bandpass!(θ, rbar_bp, wbar_bp, bl_pairs, pols, nant, plan; ref_ant = ref_ant, snr_floor = snr_floor)
        amp_plan === nothing ||
            _solve_amp_bandpass!(
            θ, rbar_bp, wbar_bp, bl_pairs, pols, nant, amp_plan, geom.channel_freqs;
            snr_floor = snr_floor, spw_of_chan = geom.spw_of_chan, smoother = amp_smoother,
        )
    end
    if get(ENV, "GUSTAVO_BP_PROFILE", "0") == "1"
        @info "[bandpass-profile] Σ decode $(round(Td, digits = 1)) / dTEC $(round(Tp, digits = 1)) / SBD $(round(Tsb, digits = 1)) / accum $(round(Tac, digits = 1)) / solve $(round(tsolve, digits = 1)) s over $(length(cal_gis)) scans, $(length(parts)) chunks"
    end
    # The scans whose per-scan (Δτ, dTEC) + SBD θ columns were refined here. Pass 2
    # can REUSE these instead of re-refining (`reuse_bandpass_refine`): the bandpass
    # is dispersion-corrected before it is built, so the residual dispersion pass 2
    # would re-fit is ~zero and the refine is the stage's dominant cost (~72%).
    return cal_gis
end

# Source name of each (source, scan) group, without materializing.
_group_sources(group_leaves) =
    [UVData.metadata(last(first(g))).source_name for g in group_leaves]

# The calibrator for the bandpass: the explicit `bandpass_source`, else the
# source with the largest TOTAL detection SNR over its scans — not the single
# brightest scan. A one-scan source can carry the brightest scan while covering
# a fraction of the array (on VR2505, 3C454.3's lone scan shares ZERO stations
# with 0607-157); total SNR weighs brightness AND scan count, so the track's
# workhorse source wins and its station coverage comes with it.
function _bandpass_calibrator(bandpass_source, group_sources, scan_snr)
    # `:all`: accumulate EVERY scan of EVERY source. Mixing sources is safe for
    # the per-channel SHAPE for the same reason the coverage top-up is (see
    # `_bandpass_coverage_topup`), and it removes the single-source restriction
    # that left stations outside the calibrator's scans with a ONE-scan curve
    # (VR2505: HB/HV/KE/YG never appear in a 1803+784 scan).
    bandpass_source === :all && return :all
    bandpass_source !== nothing && return String(bandpass_source)
    isempty(group_sources) && return ""
    tot = Dict{String, Float64}()
    for (gi, src) in enumerate(group_sources)
        isfinite(scan_snr[gi]) || continue
        tot[src] = get(tot, src, 0.0) + scan_snr[gi]
    end
    isempty(tot) && return group_sources[1]
    return argmax(src -> tot[src], collect(keys(tot)))
end

# Station set of one (source, scan) group, from lazy-leaf metadata (no reads).
function _group_stations(group)
    sts = Set{Int}()
    for (_, leaf) in group
        for (a, b) in UVData.baselines(leaf).pairs
            a == b && continue
            push!(sts, a)
            push!(sts, b)
        end
    end
    return sts
end

# Coverage top-up: stations absent from every calibrator scan would get NO
# bandpass (g = 1). For each such station, add the highest-SNR scan (any source)
# that contains it. Mixing sources is safe for the bandpass SHAPE: a source's
# structure phase is flat in frequency per baseline, so it biases every channel
# of the per-channel solve identically and cancels in the shape; per-scan
# ionosphere differences land in the frozen curve's mean, which each scan's
# dTEC is measured relative to.
function _bandpass_coverage_topup(cal_gis, group_leaves, scan_snr)
    covered = Set{Int}()
    for gi in cal_gis
        union!(covered, _group_stations(group_leaves[gi]))
    end
    extra = Int[]
    order = sortperm([isfinite(scan_snr[gi]) ? scan_snr[gi] : -Inf for gi in eachindex(group_leaves)]; rev = true)
    for gi in order
        gi in cal_gis && continue
        sts = _group_stations(group_leaves[gi])
        isempty(setdiff(sts, covered)) && continue
        push!(extra, gi)
        union!(covered, sts)
    end
    return extra
end

# Residual visibilities for one scan group: Vg divided by the current θ gains
# evaluated at the group's (global chan, global ti) window. Used for rounds > 1
# (the search needs a full residual cube); the adhoc stage uses the fused
# per-leaf accumulation (`_accumulate_leaf_rbar!`) instead.
function _residual_vis(ev::GainEvaluator, θ::AbstractVector, grp::_ScanGroup)
    g = evaluate_gains(ev, θ, grp.g_ci, grp.g_ti)    # (nchan, nti, nant, 2)
    nchan, nti, nbl, npol = size(grp.Vg)
    out = similar(grp.Vg)
    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(grp.pol_products[p])
        for bi in 1:nbl
            a, b = grp.bl_pairs[bi]
            for tt in 1:nti, c in 1:nchan
                ga = g[c, tt, a, fa]
                gb = g[c, tt, b, fb]
                denom = ga * conj(gb)
                if abs(ga) < 1.0e-12 || abs(gb) < 1.0e-12 || !isfinite(denom)
                    out[c, tt, bi, p] = ComplexF64(NaN, NaN)
                else
                    out[c, tt, bi, p] = grp.Vg[c, tt, bi, p] / denom
                end
            end
        end
    end
    return out
end
