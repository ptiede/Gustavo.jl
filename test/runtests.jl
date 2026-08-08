using Gustavo
using Test
using LinearAlgebra
using Statistics
using Random
using StructArrays
using Dates
using OrderedCollections
using FITSFiles: Card
using CairoMakie
using DimensionalData
using DimensionalData: DimArray, DimStack, dims, Ti
using Gustavo.UVData: Pol, Frequency, UVW, Baseline, UVSet, pol_products
using Gustavo.UVData: antennas, baselines, source_name, scan_name, frequencies, timestamps
using PolarizedTypes: RPol, LPol

# Test helper: reconstruct the legacy off1/off2 index tables from a ComponentPlan.
include("plan_offsets.jl")

include("test_calibration.jl")

# FITS-IDI writer round-trip tests (Phase 2 of the fringe-fitter refactor).
include("test_fitsidi.jl")

# Per-baseline FFT fringe search (Phase 3 of the fringe-fitter refactor).
include("test_fringe_search.jl")

# Per-feed stationization with closure (Phase 4 of the fringe-fitter refactor).
include("test_stationize.jl")

# OU / Matérn-1/2 state-space phase smoother primitives (underpins adhoc :gp).
include("test_statespace.jl")

# Per-observable frequency-shape specs and their per-track fit (the bandpass
# smoother's shape assumptions; ARShape rides the OU primitives above).
include("test_shapes.jl")

# Globally-closing adhoc phasing (Phase 5 of the fringe-fitter refactor).
include("test_adhoc.jl")

# End-to-end CalibrationSolution + fit pipeline (Phase 6).
include("test_pipeline.jl")

# Modular calibration pipeline (CalibrationPipeline / calibrate refactor).
# Reuses _build_fringe_uvset + CAL/FP/UVP aliases from test_pipeline.jl.
include("test_pipeline_config.jl")

# Composable-pipeline interface (step protocol, transforms, selections, stage
# snapshots, fit/calibrate/fitcalibrate). Reuses the same aliases.
include("test_interface.jl")

# New streaming engine vs the frozen monolith oracle (M2 gates): grouping /
# materialization / search parity, and the transform chain vs the precal path.
include("test_stream.jl")
include("test_transforms.jl")

# The FringeFit step on the new engine (M3 gates): θ ≡ frozen stage A,
# cross-feed rate opt-in, fit-on-subset masking, transforms on the new path.
include("test_fringe_step.jl")

# The Bandpass step on the new engine (M4 gates): fringe+bandpass θ
# vs the frozen monolith, refine-kernel bit parity, coverage top-up,
# step_solution extraction + portable ApplySolution.
include("test_bandpass_step.jl")

# Bandpass(smoother = JointSmoother()): the alternating complex-visibility +
# per-scan source-coherence solve, vs. the closure-based PerTrackSmoother
# default, and the shape specs acting as priors inside its gain update.
include("test_joint_bandpass.jl")

# The TemporalSmoother step + output sink: multi-scan full-pipeline solves
# (incl. the refine polish split), standalone calibrate ≡ fused output,
# AprioriAmplitude as a recorded output-chain step.
include("test_smoother_step.jl")

# Fringe diagnostics + Makie plot stubs (Phase 8).
include("test_fringe_diagnostics.jl")

# fringe_station_solutions θ-decode + the `rel_time` model option.
# Reuses _build_fringe_uvset + CAL/FP/UVP aliases from test_pipeline.jl.
include("test_fringe_station_solutions.jl")

# Phase-cal (injected tone) calibration: multitone fit + precal hook.
# Reuses _build_fringe_uvset + _coherence from test_pipeline.jl.
include("test_phasecal.jl")

# Per-station weight correction (station_weight_scale / the weight_scale option).
# Reuses _build_fringe_uvset + the FP/CAL/UVP aliases from test_pipeline.jl.
include("test_weight_scale.jl")

# The executor seam: the group scheduler under each outer scheduler — dispatch
# order, task cap, and bit-identical θ/outputs whichever one runs the pass.
include("test_executors.jl")

function synthetic_uvdata()
    vis = ComplexF64[
        1.0 + 0.0im 0.8 + 0.1im 0.9 - 0.1im 0.7 + 0.2im;
        0.9 + 0.1im 0.7 + 0.2im 0.8 + 0.0im 0.6 + 0.1im;
        1.1 - 0.1im 0.9 + 0.0im 1.0 + 0.1im 0.8 - 0.1im;
        1.0 + 0.2im 0.8 + 0.3im 0.9 + 0.1im 0.7 + 0.0im;

        0.9 + 0.0im 0.7 + 0.1im 0.8 - 0.1im 0.6 + 0.2im;
        0.8 + 0.1im 0.6 + 0.2im 0.7 + 0.0im 0.5 + 0.1im;
        1.0 - 0.1im 0.8 + 0.0im 0.9 + 0.1im 0.7 - 0.1im;
        0.9 + 0.2im 0.7 + 0.3im 0.8 + 0.1im 0.6 + 0.0im;
    ]
    vis = ComplexF32.(reshape(vis, 2, 4, 4))
    weights = fill(1.0f0, size(vis))
    obs_time_synth = [0.0, 1.0]
    # Internal MSv4-canonical correlation order. AIPS Stokes axis on disk is
    # [-1,-2,-3,-4] (RR/LL/RL/LR) which maps to ["PP","QQ","PQ","QP"]; the
    # FITS read path permutes to MSv4 ["PP","PQ","QP","QQ"]. Since the
    # synthetic vis values are placeholders, we just label the dim in MSv4
    # order — the round-trip test exercises the read/write permutation.
    pol_labels_synth = ["PP", "PQ", "QP", "QQ"]
    channel_freqs_synth = collect(1.0:4.0)
    vis = DimArray(vis, (Ti(obs_time_synth), Pol(pol_labels_synth), Frequency(channel_freqs_synth)))
    weights = DimArray(weights, (Ti(obs_time_synth), Pol(pol_labels_synth), Frequency(channel_freqs_synth)))
    uvw = DimArray(zeros(Float32, 2, 3), (Ti(obs_time_synth), UVW(["U", "V", "W"])))

    UV = Gustavo.UVData
    nominal_basis_v = [(RPol(), LPol()), (RPol(), LPol())]
    response_v = [Diagonal(ones(ComplexF32, 2)) for _ in 1:2]
    pol_angles_v = [(0.0f0, 0.0f0), (0.0f0, 0.0f0)]
    antennas_v = [
        UV.Antenna(;
            name = "AA",
            station_xyz = zeros(3),
            mount = UV.MountAltAz(),
            nominal_basis = nominal_basis_v[1],
            response = response_v[1],
            pol_angles = pol_angles_v[1],
        ),
        UV.Antenna(;
            name = "AX",
            station_xyz = zeros(3),
            mount = UV.MountAltAz(),
            nominal_basis = nominal_basis_v[2],
            response = response_v[2],
            pol_angles = pol_angles_v[2],
        ),
    ]
    antennas = UV.AntennaTable(
        StructArray(antennas_v), zeros(3), "TEST",
        (POLCALA = [Float32[], Float32[]], POLCALB = [Float32[], Float32[]]),
    )
    freq_setup = UV.FrequencySetup(;
        name = "FRQSEL_1",
        ref_freq = 1.0e9,
        channel_freqs = collect(1.0:4.0),
        ch_widths = fill(1.0f0, 4),
        total_bandwidths = fill(1.0f0, 4),
        sidebands = Int32.(fill(1, 4)),
    )
    array_obs = UV.ObsArrayMetadata(;
        telescope = "TEST", instrume = "TEST",
        date_obs = "2000-01-01", equinox = 2000.0f0, bunit = "JY",
        rdate = "2000-01-01", earth_rot_rate = 360.0f0, poltype = "APPROX",
    )

    # Memo-117-shaped primary HDU cards: NAXIS=7 random-groups layout plus
    # CTYPE cards for the regular axes (so STOKES/FREQ/RA/DEC round-trip
    # through `parse_stokes_axis` and friends), plus PTYPE cards for the
    # canonical random parameters.
    primary_cards = Card[
        Card("NAXIS", 7),
        Card("OBJECT", "TEST"),
        Card("TELESCOP", "TEST"),
        Card("INSTRUME", "TEST"),
        Card("DATE-OBS", "2000-01-01"),
        Card("CTYPE2", "COMPLEX"),
        Card("CRVAL2", 1.0), Card("CDELT2", 1.0), Card("CRPIX2", 1.0),
        Card("CTYPE3", "STOKES"),
        Card("CRVAL3", -1.0), Card("CDELT3", -1.0), Card("CRPIX3", 1.0),
        Card("CTYPE4", "FREQ"),
        Card("CRVAL4", 1.0e9), Card("CDELT4", 1.0), Card("CRPIX4", 1.0),
        Card("CTYPE5", "IF"),
        Card("CTYPE6", "RA"), Card("CRVAL6", 0.0),
        Card("CTYPE7", "DEC"), Card("CRVAL7", 0.0),
        Card("PTYPE1", "UU---SIN"),
        Card("PTYPE2", "VV---SIN"),
        Card("PTYPE3", "WW---SIN"),
        Card("PTYPE4", "BASELINE"),
        Card("PTYPE5", "DATE"),
    ]
    uvset = UVSet(
        (
            vis = vis,
            weights = weights,
            uvw = uvw,
            obs_time = obs_time_synth,
            record_scan_name = ["1", "2"],
            baselines = UV.BaselineIndex([(1, 2), (1, 2)], [(1, 2)]; antenna_names = ["AA", "AX"]),
            extra_columns = NamedTuple(),
            antennas = antennas,
            array_obs = array_obs,
            freq_setups = UV.FrequencySetup[freq_setup],
            record_spw_index = Int32[1, 1],
            source_name = "TEST",
            ra = 0.0, dec = 0.0,
            basename = "synthetic",
        )
    )
    UV.register_primary_cards!(uvset, primary_cards)
    return uvset
end

@testset "Gustavo.jl" begin
    for sub in (:UVData, :Calibration, :Fringe)
        @test isdefined(Gustavo, sub)
        @test getfield(Gustavo, sub) isa Module
    end
    @test isdefined(Gustavo, :fitcalibrate)
end

@testset "top-level export surface" begin
    top = names(Gustavo)

    # A bare `using Gustavo` spans the production path end to end: read a set,
    # fit/calibrate it, extract the per-stage step `sol[:fringe]` returns, write it.
    for n in (
            :UVSet, :load_uvfits, :load_fitsidi, :write_uvfits, :write_fitsidi,
            :fit, :calibrate, :fitcalibrate, :StepSolution,
        )
        @test n in top
    end

    # Streaming-engine internals stay behind `Fringe`: reachable for users who
    # drive the engine directly, absent from the pipeline-level namespace. They
    # are `Streaming`'s, re-exported — the same binding under both names.
    for n in (:ScanGroupSpec, :materialize_cube, :materialize_leaves)
        @test !(n in top)
        @test n in names(Gustavo.Fringe)
        @test n in names(Gustavo.Streaming)
        @test getproperty(Gustavo.Fringe, n) === getproperty(Gustavo.Streaming, n)
    end

    # The four submodules, named at the top level.
    for n in (:UVData, :Calibration, :Streaming, :Fringe)
        @test n in top
    end

    # Every axis name a stored or returned array can carry, so scripts index
    # leaves with a bare `using Gustavo`. `Ti` is DimensionalData's dim under
    # both names — the same binding, so no ambiguity when both are loaded.
    for n in (:Pol, :Frequency, :Ant, :Baseline, :Ti, :UVW, :Feed, :Scan)
        @test n in top
        @test getproperty(Gustavo, n) <: DimensionalData.Dimension
    end
    @test Gustavo.Ti === DimensionalData.Ti

    # Solver-internal parameter bookkeeping: still reachable, no longer exported.
    @test !(:ComponentPlan in names(Gustavo.Calibration))
    @test Gustavo.Calibration.ComponentPlan isa Type

    # The term-authoring interface: the hooks a new `AbstractGainTerm`
    # implements, exported as Gustavo's documented extension point.
    for n in (:term_axes, :param_shapes, :term_eval, :term_label, :freq_coordinate, :time_coordinate)
        @test n in names(Gustavo.Calibration)
    end
    @test !(:nparams_per_block in names(Gustavo.Calibration))
    @test Gustavo.Calibration.nparams_per_block isa Function
end

@testset "Baseline stability plots" begin
    UV = Gustavo.UVData
    data = synthetic_uvdata()
    corr = UV.apply((leaf, _info, _meta) -> UV.rebuild_visibilities(leaf, parent(leaf[:vis]) .* (1.0 + 0.0im), parent(leaf[:weights])), data)

    gains = reshape(
        ComplexF64[
            1.0 * cis(0.1), 2.0 * cis(0.2),
            3.0 * cis(0.3), 4.0 * cis(0.4),
            5.0 * cis(0.5), 6.0 * cis(0.6),
            7.0 * cis(0.7), 8.0 * cis(0.8),
            1.1 * cis(0.2), 2.1 * cis(0.3),
            3.1 * cis(0.4), 4.1 * cis(0.5),
            5.1 * cis(0.6), 6.1 * cis(0.7),
            7.1 * cis(0.8), 8.1 * cis(0.9),
        ], 2, 2, 2, 2
    )

    pol_idx, pol_labels = UV.resolve_plot_polarizations(data; pol = :parallel)
    @test pol_idx == [1, 4]
    @test pol_labels == ["PP", "QQ"]

    pol_idx, pol_labels = UV.resolve_plot_polarizations(data; pol = ["QQ", "PQ"])
    @test pol_idx == [4, 2]
    @test pol_labels == ["QQ", "PQ"]

    @test !isnothing(UV.plot_stability(data, corr, ("AA", "AX"); quantity = :phase, pol = "PP"))
    @test !isnothing(UV.plot_stability(data, corr, ("AA", "AX"); quantity = :amplitude, pol = :all, relative = true))
    @test !isnothing(UV.plot_gain_solutions(gains, data))
    @test !isnothing(UV.plot_gain_solutions(gains, data; quantity = :amplitude, pol = 1, sites = "AA", relative = false))
    @test !isnothing(UV.plot_gain_solutions(gains, data; quantity = :phase, pol = [2], sites = ["AX"]))
    fig_embed = Figure(size = (1400, 500))
    @test !isnothing(UV.plot_stability(fig_embed[1, 1], data, corr, ("AA", "AX"); quantity = :phase, pol = "PP"))
    @test !isnothing(UV.plot_gain_solutions(fig_embed[1, 2], gains, data; quantity = :phase, pol = [2], sites = ["AX"]))

    fig = UV.plot_stability(data, corr, ("AA", "AX"); quantity = :phase, pol = "PP")
    @test_nowarn show(IOBuffer(), MIME("image/png"), fig)
    @test_nowarn show(IOBuffer(), MIME("image/png"), fig_embed)
end

@testset "Amplitude stability summary" begin
    UV = Gustavo.UVData
    vis_block = ComplexF64[
        1.0 + 0.0im 2.0 + 0.0im;
        -1.0 + 0.0im -2.0 + 0.0im
    ]
    weight_block = ones(Float64, 2, 2)
    groups = [1, 2]

    summary = UV.scan_averaged_amplitude_series(vis_block, weight_block; relative = false, groups = groups)
    @test summary ≈ [1.0, 2.0]

    noise_vis = reshape(ComplexF64[1.0 + 0.0im, 2.0 + 0.0im], 1, 2)
    noise_weights = fill(2.0, 1, 2)

    _, amp_noise = UV.amplitude_series_with_noise(noise_vis, noise_weights; relative = false)
    @test amp_noise ≈ fill(1 / sqrt(2), 2)

    rel_amp, rel_amp_noise = UV.amplitude_series_with_noise(noise_vis, noise_weights; relative = true)
    @test rel_amp ≈ [1.0, 2.0]
    @test rel_amp_noise ≈ [0.0, 1 / sqrt(2)]

    phase, phase_noise = UV.phase_series_with_noise(noise_vis, noise_weights; relative = false)
    @test phase ≈ [0.0, 0.0]
    @test phase_noise ≈ [1 / sqrt(2), 1 / (2sqrt(2))]

    rel_phase, rel_phase_noise = UV.phase_series_with_noise(noise_vis, noise_weights; relative = true)
    @test rel_phase ≈ [0.0, 0.0]
    @test rel_phase_noise ≈ [0.0, 1 / (2sqrt(2))]
end

@testset "Diagnostics series y-limits" begin
    UV = Gustavo.UVData

    ylims = UV.finite_series_ylims(([1.0, 2.0, NaN], [4.0]); pad_fraction = 0.1, min_pad = 0.0)
    @test collect(ylims) ≈ [0.7, 4.3]

    ylims_noise = UV.finite_series_ylims(([1.0, 1.0],), ([10.0, 0.2],); pad_fraction = 0.1, min_pad = 0.0, noise_cap_fraction = 0.5)
    @test collect(ylims_noise) ≈ [0.4, 1.6]

    @test isnothing(UV.finite_series_ylims(([NaN], [Inf, -Inf])))
    @test isequal(UV.shared_track(([1.0, 2.0, NaN], [1.0, 2.0, NaN])), [1.0, 2.0, NaN])
    @test isnothing(UV.shared_track(([1.0, 2.0], [1.0, 3.0])))
end

@testset "Reference-relative series" begin
    UV = Gustavo.UVData

    # `Calibration` re-exports the phase referencing helper from `UVData`.
    @test Gustavo.Calibration.phase_relative_to_ref === UV.phase_relative_to_ref

    @test UV.phase_relative_to_ref([0.5, 1.5, 2.5]) ≈ [0.0, 1.0, 2.0]
    # Differences wrap into (-π, π].
    @test UV.phase_relative_to_ref([0.0, 3π / 2]) ≈ [0.0, -π / 2]
    # A non-finite reference falls through to the first finite entry.
    @test UV.phase_relative_to_ref([NaN, 1.0, 2.0]) ≈ [NaN, 0.0, 1.0] nans = true
    @test all(isnan, UV.phase_relative_to_ref([NaN, NaN]))
    # ref_idx outside the axes yields all-NaN rather than throwing.
    @test all(isnan, UV.phase_relative_to_ref([1.0, 2.0], 5))

    @test UV.amplitude_relative_to_ref([2.0, 4.0, 1.0]) ≈ [1.0, 2.0, 0.5]
    # Non-positive and non-finite amplitudes are not usable references.
    @test UV.amplitude_relative_to_ref([0.0, 4.0, 2.0]) ≈ [NaN, 1.0, 0.5] nans = true
    @test all(isnan, UV.amplitude_relative_to_ref([0.0, -1.0, NaN]))
end

@testset "write_uvfits HDU construction" begin
    UV = Gustavo.UVData
    data = synthetic_uvdata()

    # End-to-end: write_uvfits should run on the synthetic dataset.
    tmp = tempname() * ".uvfits"
    try
        @test UV.write_uvfits(tmp, data) == tmp
        @test isfile(tmp)
        @test filesize(tmp) > 0
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
end

@testset "UVData round-trip preserves canonical fields" begin
    UV = Gustavo.UVData
    data = synthetic_uvdata()

    fs_orig = UV.freq_setup(data)
    @test eltype(fs_orig.channel_freqs) == Float64
    @test eltype(fs_orig.ch_widths) == Float32
    @test eltype(fs_orig.total_bandwidths) == Float32
    @test eltype(fs_orig.sidebands) == Int32

    tmp = tempname() * ".uvfits"
    try
        UV.write_uvfits(tmp, data)
        round_set = UV.load_uvfits(tmp)
        @test round_set isa UV.UVSet

        # Compare leaves directly: load → write → reload should preserve
        # vis/weights/uvw cubes per partition and the baseline codes.
        for (k, leaf_orig) in pairs(UV.branches(data))
            leaf_round = UV.branches(round_set)[k]
            @test parent(leaf_round[:uvw]) == Float32.(parent(leaf_orig[:uvw]))
            @test parent(leaf_round[:weights]) == Float32.(parent(leaf_orig[:weights]))
            @test UV.baselines(leaf_round).pairs == UV.baselines(leaf_orig).pairs
        end

        fs_round = UV.freq_setup(round_set)
        @test fs_round.channel_freqs == fs_orig.channel_freqs
        @test fs_round.ch_widths == fs_orig.ch_widths
        @test fs_round.total_bandwidths == fs_orig.total_bandwidths
        @test fs_round.sidebands == fs_orig.sidebands
        @test pol_products(round_set) == pol_products(data)

        # obs_time round-trip precision: AIPS DATE PTYPE columns are
        # stored as Float32 (~3.6e-7 hour ULP at 24h magnitude, i.e.
        # ~1.3 ms). The round-tripped Ti axis must agree to within that
        # budget for every leaf.
        for (k, leaf_orig) in pairs(UV.branches(data))
            leaf_round = UV.branches(round_set)[k]
            t_orig = collect(UV.obs_time(leaf_orig))
            t_round = collect(UV.obs_time(leaf_round))
            @test length(t_round) == length(t_orig)
            @test maximum(abs.(t_round .- t_orig)) < 2.0e-3   # 2 ms slack
        end
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
end

@testset "UVDataset extra_columns preservation" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    base_root = UV.metadata(base)

    # Materialise the flat record stream we need to feed UVSet's NamedTuple
    # factory. Walk leaves in order to assemble per-record arrays with the
    # same shape the load path produces.
    leaves_collected = collect(values(UV.branches(base)))
    obs_times = Float64[]
    scan_indices = String[]
    bl_pairs_per_record = Tuple{Int, Int}[]
    vis_chunks = Any[]
    weights_chunks = Any[]
    uvw_chunks = Any[]
    for leaf in leaves_collected
        info = UV.metadata(leaf)
        ti_lookup = collect(UV.obs_time(leaf))
        bls = info.baselines
        vis_p = parent(leaf[:vis])     # (Frequency, Ti, Baseline, Pol)
        w_p = parent(leaf[:weights])
        uvw_p = parent(leaf[:uvw])     # (Ti, Baseline, UVW)
        for (ti, bi) in info.record_order
            push!(obs_times, ti_lookup[ti])
            push!(scan_indices, info.scan_name)
            push!(bl_pairs_per_record, bls.pairs[bi])
            # Slice (Frequency, Pol) for one (ti, bi); transpose to (Pol, Frequency)
            # to match the flat fixture's (Ti, Pol, Frequency) layout.
            push!(vis_chunks, copy(transpose(vis_p[:, ti, bi, :])))
            push!(weights_chunks, copy(transpose(w_p[:, ti, bi, :])))
            push!(uvw_chunks, uvw_p[ti, bi, :])
        end
    end
    nint = length(obs_times)
    npol = size(vis_chunks[1], 1)
    nchan = size(vis_chunks[1], 2)
    vis_flat = Array{ComplexF32, 3}(undef, nint, npol, nchan)
    weights_flat = Array{Float32, 3}(undef, nint, npol, nchan)
    uvw_flat = Array{Float32, 2}(undef, nint, 3)
    for i in 1:nint
        vis_flat[i, :, :] .= vis_chunks[i]
        weights_flat[i, :, :] .= weights_chunks[i]
        uvw_flat[i, :] .= uvw_chunks[i]
    end
    pol_labels = pol_products(base)
    chan_freqs = UV.channel_freqs(UV.freq_setup(base))
    vis_da = DimArray(vis_flat, (Ti(obs_times), UV.Pol(pol_labels), UV.Frequency(chan_freqs)))
    weights_da = DimArray(weights_flat, (Ti(obs_times), UV.Pol(pol_labels), UV.Frequency(chan_freqs)))
    uvw_da = DimArray(uvw_flat, (Ti(obs_times), UV.UVW(["U", "V", "W"])))

    unique_pairs = sort(unique(bl_pairs_per_record))
    bls = UV.BaselineIndex(bl_pairs_per_record, unique_pairs; antenna_names = UV.union_antennas(base).name)

    inttim_values = Float32.(0.5 .* (1:nint))
    primary_cards = Card[
        Card("PTYPE1", "UU---SIN"),
        Card("PTYPE2", "VV---SIN"),
        Card("PTYPE3", "WW---SIN"),
        Card("PTYPE4", "BASELINE"),
        Card("PTYPE5", "DATE"),
        Card("PTYPE6", "INTTIM"),
        Card("PCOUNT", 6),
    ]
    extras = (var"INTTIM" = inttim_values,)

    base_fs = UV.freq_setup(base)
    uvset = UV.UVSet(
        (
            vis = vis_da, weights = weights_da, uvw = uvw_da,
            obs_time = obs_times,
            record_scan_name = scan_indices,
            baselines = bls,
            extra_columns = extras,
            antennas = UV.union_antennas(base),
            array_obs = base_root.array_obs,
            freq_setups = UV.FrequencySetup[base_fs],
            record_spw_index = ones(Int32, length(obs_times)),
            source_name = "TEST", ra = 0.0, dec = 0.0,
            basename = "synthetic",
        )
    )
    UV.register_primary_cards!(uvset, primary_cards)

    # Verify INTTIM round-trip through the leaves. AIPS DATE PTYPE columns
    # are reconstructed at write time from `obs_time` + RDATE; no
    # `date_param` field on `PartitionInfo`.
    leaves_collected = collect(values(UV.branches(uvset)))
    cat_inttim = vcat([UV.extra_columns(l).INTTIM for l in leaves_collected]...)
    @test cat_inttim == inttim_values

    tmp = tempname() * ".uvfits"
    try
        @test UV.write_uvfits(tmp, uvset) == tmp
        @test filesize(tmp) > 0
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
end

@testset "UVSet partition tree shape" begin
    UV = Gustavo.UVData
    uvset = synthetic_uvdata()
    @test uvset isa UV.UVSet
    nleaves = length(UV.branches(uvset))
    @test nleaves == 2
    for s in 1:nleaves
        key = UV.partition_key(; source_key = :src_TEST, scan_name = string(s))
        @test haskey(UV.branches(uvset), key)
        leaf = UV.branches(uvset)[key]
        @test UV.primary_scan_name(leaf) == string(s)
        @test ndims(parent(leaf[:vis])) == 4
        @test ndims(parent(leaf[:uvw])) == 3
    end
end

@testset "UVSet apply" begin
    UV = Gustavo.UVData
    uvset = synthetic_uvdata()

    # Identity apply preserves shape and globals.
    same = UV.apply((p, _info, _root) -> p, uvset)
    @test same isa UV.UVSet
    @test length(UV.branches(same)) == length(UV.branches(uvset))
    @test UV.union_antennas(same) == UV.union_antennas(uvset)
    for (key, _) in UV.branches(uvset)
        @test haskey(UV.branches(same), key)
    end

    # Transformed leaves: scale weights by 2.
    scaled = UV.apply(uvset) do leaf, _info, _root
        new_w = parent(leaf[:weights]) .* 2
        UV.rebuild_visibilities(leaf, parent(leaf[:vis]), new_w)
    end
    for (key, leaf) in UV.branches(scaled)
        orig = UV.branches(uvset)[key]
        @test parent(leaf[:weights]) ≈ 2 .* parent(orig[:weights])
    end
end

@testset "UVSet selection ergonomics" begin
    UV = Gustavo.UVData
    using DimensionalData: At
    uvset = synthetic_uvdata()

    # Single-scan selection returns a leaf DimTree.
    p1 = UV.select_scan(uvset, "TEST", 1)
    @test p1 isa UV.DimTree
    @test p1 === UV.branches(uvset)[UV.partition_key(; source_key = :src_TEST, scan_name = "1")]

    # Multi-scan selection via select_partition returns a sub-UVSet.
    sub_s1 = UV.select_partition(uvset; scan = 1)
    @test sub_s1 isa UV.UVSet
    @test length(UV.branches(sub_s1)) == 1

    # Source filter — single source "TEST" still returns the full set.
    sub_test = UV.select_source(uvset, "TEST")
    @test length(UV.branches(sub_test)) == length(UV.branches(uvset))
    sub_other = UV.select_source(uvset, "NONEXISTENT")
    @test isempty(UV.branches(sub_other))

    # Station filter — synthetic fixture has antennas "AA","AX" so the only
    # baseline AA-AX touches both. Filtering by "AA" keeps it; by "QQ"
    # drops everything.
    sub_aa = UV.select_station(uvset, "AA")
    @test sub_aa isa UV.UVSet
    @test length(UV.branches(sub_aa)) == length(UV.branches(uvset))
    for (_, leaf) in UV.branches(sub_aa)
        for bi in eachindex(UV.baselines(leaf).pairs)
            @test "AA" in (UV.baselines(leaf).ant1_names[bi], UV.baselines(leaf).ant2_names[bi])
        end
    end
    sub_qq = UV.select_station(uvset, "QQ")
    @test isempty(UV.branches(sub_qq))

    # Single-baseline filter.
    sub_bl = UV.select_baseline(uvset, "AA-AX")
    @test sub_bl isa UV.UVSet
    @test length(UV.branches(sub_bl)) == length(UV.branches(uvset))
    for (_, leaf) in UV.branches(sub_bl)
        @test UV.baselines(leaf).labels == ["AA-AX"]
        @test size(parent(leaf[:vis]), 2) == 1   # Baseline axis collapsed to 1
    end

    # Time window — synthetic fixture has obs_time = [0.0, 1.0], one per scan.
    win = UV.time_window(uvset, 0.5, 1.5)
    @test win isa UV.UVSet
    @test length(UV.branches(win)) == 1
    @test haskey(UV.branches(win), UV.partition_key(; source_key = :src_TEST, scan_name = "2"))
end

@testset "scan_average(uvset) collapses Ti to length 1" begin
    UV = Gustavo.UVData
    uvset = synthetic_uvdata()
    avg_set = UV.scan_average(uvset)

    @test avg_set isa UV.UVSet
    @test length(UV.branches(avg_set)) == length(UV.branches(uvset))
    for (key, leaf) in UV.branches(avg_set)
        # Layout: (Frequency, Ti, Baseline, Pol). Ti axis is dim 2.
        @test size(parent(leaf[:vis]), 2) == 1
        @test length(UV.obs_time(leaf)) == 1
    end

    # The reducer-style entry point composes via apply.
    via_apply = UV.apply(UV.TimeAverage(), uvset)
    for (key, leaf) in UV.branches(via_apply)
        @test parent(leaf[:vis]) == parent(UV.branches(avg_set)[key][:vis])
    end
end

@testset "Weighted LSQ accepts mixed-precision RHS" begin
    # Regression: Memo-117 cleanup made `weights` Float32 while `A` is built in
    # Float64 by `design_matrices`. LinearSolve's QR `ldiv!` errored on the
    # Float64 factorization against a Float32 RHS — promote to a shared eltype.
    CALIB = Gustavo.Calibration
    A = Float64[1.0 0.0; 1.0 1.0; 1.0 2.0; 1.0 3.0]
    b32 = Float32[1.0, 2.0, 3.1, 3.9]
    iv32 = Float32[1.0, 1.0, 1.0, 1.0]
    x = CALIB.weighted_least_squares(A, b32, iv32)
    @test eltype(x) == Float64
    @test x ≈ A \ Float64.(b32) rtol = 1.0e-6

    # Same for the regularized and constrained variants.
    xr = CALIB.weighted_regularized_least_squares(A, b32, iv32, [0.0, 0.0])
    @test xr ≈ A \ Float64.(b32) rtol = 1.0e-6
    C = Float64[1.0 0.0]
    d = Float64[0.5]
    xc = CALIB.weighted_constrained_least_squares(A, b32, iv32, C, d)
    @test isfinite(xc[1]) && isfinite(xc[2])
end

@testset "Regularized WLS: penalty matrix generalizes the diagonal vector" begin
    CALIB = Gustavo.Calibration
    A = Float64[1.0 0.0; 1.0 1.0; 1.0 2.0; 1.0 3.0]
    b = Float64[1.0, 2.0, 3.1, 3.9]
    iv = ones(4)
    lambda = [0.2, 0.7]

    x_vec = CALIB.weighted_regularized_least_squares(A, b, iv, lambda)
    x_mat = CALIB.weighted_regularized_least_squares(A, b, iv, Diagonal(sqrt.(lambda)))
    @test x_vec ≈ x_mat rtol = 1.0e-10

    # A non-diagonal R (a first-difference roughness penalty) must reduce to
    # solving the row-stacked normal equations directly.
    R = Float64[1.0 -1.0]
    x_r = CALIB.weighted_regularized_least_squares(A, b, iv, R)
    sw = sqrt.(iv)
    Aw = A .* sw
    bw = b .* sw
    x_ref = vcat(Aw, R) \ vcat(bw, zeros(size(R, 1)))
    @test x_r ≈ x_ref rtol = 1.0e-10
end

@testset "DimArray slicing" begin
    using DimensionalData
    UV = Gustavo.UVData
    raw = synthetic_uvdata()

    # Per-leaf Pol slice on a UVSet: pull leaf, then DimTree's selector.
    leaf1 = UV.select_scan(raw, "TEST", 1)
    sliced = leaf1[Pol = At("PP")]
    @test sliced isa DimensionalData.AbstractDimTree
end

# Build a second single-source UVSet so we can exercise multi-source
# composition via merge_uvsets. The fixture mirrors `synthetic_uvdata`
# but with source name "3C273" and slightly shifted obs_times.
function synthetic_uvdata_3c273()
    base = synthetic_uvdata()
    UV = Gustavo.UVData
    leaves_orig = collect(values(UV.branches(base)))
    new_branches = Gustavo.UVData.DimensionalData.TreeDict()
    for leaf in leaves_orig
        info = UV.metadata(leaf)
        info_3c = UV.update(
            info;
            source_name = "3C273",
            source_key = UV.sanitize_source("3C273"),
            field_name = "3C273",
            ra = 187.27791666666667,
            dec = 12.391122222222222,
            partition_name = "synthetic_0_3C273_$(info.scan_name)",
        )
        new_leaf = UV._build_leaf(
            leaf[:vis], leaf[:weights], leaf[:uvw];
            partition_info = info_3c,
        )
        new_key = UV.partition_key(info_3c)
        new_branches[new_key] = new_leaf
    end
    return Gustavo.UVData.DimensionalData.rebuild(base; branches = new_branches)
end

# Same antennas/array_obs as `synthetic_uvdata_3c273` but with non-overlapping
# scan time intervals — exercises that merging two UVSets whose leaves carry
# distinct scan windows preserves each leaf's own window.
function synthetic_uvdata_3c273_shifted()
    UV = Gustavo.UVData
    src = synthetic_uvdata_3c273()
    # Pick a shift large enough that the new Ti axes don't overlap the original.
    dt = 100.0
    new_branches = Gustavo.UVData.DimensionalData.TreeDict()
    # Layout: (Frequency, Ti, Baseline, Pol) for vis/weights and
    # (UVW, Ti, Baseline) for uvw — Ti is dim 2 in both. Replace it.
    _replace_ti(da, new_t) = begin
        old_dims = dims(da)
        new_dims = (old_dims[1], Ti(new_t), old_dims[3:end]...)
        return DimArray(parent(da), new_dims)
    end
    for (k, leaf) in UV.branches(src)
        info = UV.metadata(leaf)
        old_vis = leaf[:vis]
        new_t = collect(dims(old_vis, Ti)) .+ dt
        new_vis = _replace_ti(old_vis, new_t)
        new_w = _replace_ti(leaf[:weights], new_t)
        new_uvw = _replace_ti(leaf[:uvw], new_t)
        new_leaf = UV._build_leaf(new_vis, new_w, new_uvw; partition_info = info)
        new_branches[k] = new_leaf
    end
    return Gustavo.UVData.DimensionalData.rebuild(src; branches = new_branches)
end

@testset "multi-source UVSet" begin
    UV = Gustavo.UVData
    set_test = synthetic_uvdata()
    set_3c = synthetic_uvdata_3c273()

    multi = UV.merge_uvsets(set_test, set_3c)
    @test multi isa UV.UVSet
    @test length(UV.branches(multi)) == 4
    @test UV.sources(multi) == ["TEST", "3C273"]
    @test sort(UV.scan_ids(multi, "TEST")) == ["1", "2"]
    @test sort(UV.scan_ids(multi, "3C273")) == ["1", "2"]
    # summary returns one row per leaf with correct source/scan info.
    rows = Base.summary(multi)
    @test length(rows) == 4
    @test count(r -> r.source_name == "TEST", rows) == 2
    @test count(r -> r.source_name == "3C273", rows) == 2

    # select_source narrows to one source's leaves.
    sub_test = UV.select_source(multi, "TEST")
    @test length(UV.branches(sub_test)) == 2
    @test all(UV.metadata(l).source_name == "TEST" for (_, l) in UV.branches(sub_test))

    # select_scan returns the leaf for (source, scan_name).
    leaf = UV.select_scan(multi, "3C273", "1")
    @test UV.metadata(leaf).source_name == "3C273"
    @test UV.metadata(leaf).scan_name == "1"

    # Tab-completable Partitions accessor.
    ps = UV.partitions(multi)
    @test :src_TEST_spw_0_scan_1 in propertynames(ps)
    @test :src_3C273_spw_0_scan_1 in propertynames(ps)
    @test ps.src_3C273_spw_0_scan_1 === UV.branches(multi)[:src_3C273_spw_0_scan_1]

    # apply preserves tree shape; per-leaf metadata flows through.
    averaged = UV.apply(UV.TimeAverage(), multi)
    @test length(UV.branches(averaged)) == 4
    @test all(size(parent(l[:vis]), 2) == 1 for (_, l) in UV.branches(averaged))

    # write_uvfits errors clearly on multi-source input but works after select_source.
    tmp = tempname() * ".uvfits"
    try
        @test_throws ErrorException UV.write_uvfits(tmp, multi)
        @test UV.write_uvfits(tmp, UV.select_source(multi, "TEST")) == tmp
        @test isfile(tmp)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end

    # With per-leaf scan windows (no root scans table), merging two UVSets
    # whose leaves have shifted Ti axes preserves each leaf's own scan_window.
    set_shifted = synthetic_uvdata_3c273_shifted()
    multi2 = UV.merge_uvsets(set_test, set_shifted)
    @test length(UV.branches(multi2)) ==
        length(UV.branches(set_test)) + length(UV.branches(set_shifted))
    # Each per-source leaf keeps its own scan_name; the shifted set still has
    # scan labels "1" and "2".
    @test sort(UV.scan_ids(multi2, "3C273")) == ["1", "2"]
    leaf_shifted = UV.select_scan(multi2, "3C273", "1")
    @test UV.scan_window(leaf_shifted) != UV.scan_window(UV.select_scan(multi2, "TEST", "1"))

    # Strict equality rejects mismatched array-wide metadata.
    set_other_telescope = let s = synthetic_uvdata()
        m = UV.metadata(s)
        new_arr = Gustavo.UVData.ObsArrayMetadata(;
            telescope = "OTHER",
            instrume = m.array_obs.instrume,
            date_obs = m.array_obs.date_obs,
            equinox = m.array_obs.equinox,
            bunit = m.array_obs.bunit,
            extras = m.array_obs.extras,
        )
        new_root = Gustavo.UVData.UVMetadata(new_arr)
        Gustavo.UVData.DimensionalData.rebuild(s; metadata = new_root)
    end
    @test_throws ErrorException UV.merge_uvsets(set_test, set_other_telescope)
end

@testset "Source name sanitization" begin
    UV = Gustavo.UVData
    @test UV.sanitize_source("TEST") == :src_TEST
    @test UV.sanitize_source("3C273") == :src_3C273    # digit-leading
    @test UV.sanitize_source("Sgr A*") == :src_Sgr_A_  # non-identifier chars
    @test UV.sanitize_source("NGC 4486") == :src_NGC_4486
    @test UV.sanitize_source("") == :src_unknown
    @test UV.sanitize_source("  ") == :src_unknown
    @test UV.partition_key(; source_key = :src_3C273, scan_name = "5") == :src_3C273_spw_0_scan_5
end

@testset "Structural ==/hash for metadata types" begin
    UV = Gustavo.UVData
    using LinearAlgebra: Diagonal

    fs1() = UV.FrequencySetup(;
        name = "FRQSEL_1", ref_freq = 1.0e9,
        channel_freqs = collect(1.0:4.0),
        ch_widths = fill(1.0f0, 4), total_bandwidths = fill(1.0f0, 4),
        sidebands = Int32.(fill(1, 4)),
    )
    @test fs1() == fs1()
    @test hash(fs1()) == hash(fs1())
    # Differing field flips equality.
    fs_alt = UV.FrequencySetup(;
        name = "FRQSEL_1", ref_freq = 1.0e9,
        channel_freqs = collect(1.0:5.0), ch_widths = fill(1.0f0, 5),
        total_bandwidths = fill(1.0f0, 5), sidebands = Int32.(fill(1, 5)),
    )
    @test fs1() != fs_alt
    # Dedup in a Dict relies on both `==` and `hash`.
    d = Dict(fs1() => :first)
    d[fs1()] = :second
    @test d[fs1()] == :second
    @test length(d) == 1

    obs1() = UV.ObsArrayMetadata(;
        telescope = "TEST", instrume = "TEST",
        date_obs = "2000-01-01", equinox = 2000.0f0, bunit = "JY",
    )
    @test obs1() == obs1()
    @test hash(obs1()) == hash(obs1())

    mnt() = UV.MountAltAz()
    @test mnt() == mnt()
    @test hash(mnt()) == hash(mnt())

    ant() = UV.Antenna(;
        name = "AA", station_xyz = zeros(3), mount = UV.MountAltAz(),
        nominal_basis = (RPol(), LPol()),
        response = Diagonal(ones(ComplexF32, 2)),
        pol_angles = (0.0f0, 0.0f0),
    )
    @test ant() == ant()
    @test hash(ant()) == hash(ant())
end

@testset "Per-leaf freq_setup populated from load" begin
    UV = Gustavo.UVData
    data = synthetic_uvdata()
    fs_root = UV.freq_setup(data)
    # Every leaf carries its own copy of the same setup.
    for (_, leaf) in UV.branches(data)
        @test UV.freq_setup(leaf) == fs_root
    end
    # Union axis collapses identical setups to length 1.
    axis = UV.union_frequency_axis(data)
    @test length(axis) == 1
    @test axis[1] == fs_root
    # nchannels / spw_center_frequency dispatch through freq_setup(uvset).
    @test UV.nchannels(data) == length(UV.channel_freqs(fs_root))
    @test UV.spw_center_frequency(data) == UV.spw_center_frequency(fs_root)

    # Round-trip through write/read keeps the per-leaf setup.
    tmp = tempname() * ".uvfits"
    try
        UV.write_uvfits(tmp, data)
        round_set = UV.load_uvfits(tmp)
        for (_, leaf) in UV.branches(round_set)
            @test UV.channel_freqs(UV.freq_setup(leaf)) == UV.channel_freqs(fs_root)
        end
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
end

@testset "freq_setup(uvset) errors on multi-SPW; union_frequency_axis returns both" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    # Build a sibling leaf with a distinct FrequencySetup, swap it into the
    # second scan slot. Hand-built — no FITS round-trip yet (Phase 1.5).
    fs_alt = UV.FrequencySetup(;
        name = "FRQSEL_2", ref_freq = 2.0e9,
        channel_freqs = collect(2.0e9:1.0e3:(2.0e9 + 3.0e3)),
        ch_widths = fill(1.0f3, 4), total_bandwidths = fill(1.0f3, 4),
        sidebands = Int32.(fill(1, 4)),
    )
    branches = Gustavo.UVData.DimensionalData.TreeDict()
    for (k, leaf) in UV.branches(base)
        info = UV.metadata(leaf)
        if info.scan_name == "2"
            new_info = UV.update(info; freq_setup = fs_alt)
            new_leaf = UV._build_leaf(
                leaf[:vis], leaf[:weights], leaf[:uvw];
                partition_info = new_info,
            )
            branches[k] = new_leaf
        else
            branches[k] = leaf
        end
    end
    multi = Gustavo.UVData.DimensionalData.rebuild(base; branches = branches)

    @test length(UV.union_frequency_axis(multi)) == 2
    @test_throws ArgumentError UV.freq_setup(multi)
    # nchannels(uvset) routes through freq_setup(uvset) and inherits the throw.
    @test_throws ArgumentError UV.nchannels(multi)
end

@testset "partition_axes is single-point-of-extension" begin
    UV = Gustavo.UVData
    leaf = first(values(UV.branches(synthetic_uvdata())))
    info = UV.metadata(leaf)
    @test UV.partition_key(info) == :src_TEST_spw_0_scan_1

    # Append a synthetic axis without touching `partition_key` or any other
    # call site — only the axis tuple changes.
    extended = (
        UV.DEFAULT_PARTITION_AXES...,
        UV.PartitionAxis(:obs, info -> isempty(info.intent) ? "" : "obs_$(info.intent)"),
    )
    @test UV.partition_key(info, extended) == :src_TEST_spw_0_scan_1
    info_with_intent = UV.update(info; intent = "TARGET")
    @test UV.partition_key(info_with_intent, extended) == :src_TEST_spw_0_scan_1_obs_TARGET
end

"""
    synthetic_two_spw_flat()

Build a flat NamedTuple with two `FrequencySetup`s and per-record
`record_spw_index = [1, 2]` so `UVSet(flat)` splits records into two
leaves sharing scan_name "1" but differing in SPW. Mirrors
`synthetic_uvdata` shape but with two FQ rows.
"""
function synthetic_two_spw_flat()
    UV = Gustavo.UVData
    obs_t = [0.0, 1.0]
    pol_lab = ["PP", "PQ", "QP", "QQ"]
    chan_freq = collect(1.0:4.0)
    vis = ComplexF32.(rand(ComplexF32, 2, 4, 4))
    weights = fill(1.0f0, 2, 4, 4)
    vis_da = DimArray(vis, (Ti(obs_t), Pol(pol_lab), Frequency(chan_freq)))
    weights_da = DimArray(weights, (Ti(obs_t), Pol(pol_lab), Frequency(chan_freq)))
    uvw_da = DimArray(zeros(Float32, 2, 3), (Ti(obs_t), UVW(["U", "V", "W"])))

    # Reuse antennas / array_config / array_obs / primary_cards from the
    # single-source fixture. Same structure, just two SPWs.
    base = synthetic_uvdata()
    base_meta = UV.metadata(base)

    fs_a = UV.FrequencySetup(;
        name = "spw_0", ref_freq = 1.0e9,
        channel_freqs = collect(1.0:4.0), ch_widths = fill(1.0f0, 4),
        total_bandwidths = fill(1.0f0, 4), sidebands = Int32.(fill(1, 4)),
        extras = (; frqsel = Int32(1)),
    )
    fs_b = UV.FrequencySetup(;
        name = "spw_1", ref_freq = 1.0e9,
        channel_freqs = collect(5.0:8.0), ch_widths = fill(1.0f0, 4),
        total_bandwidths = fill(1.0f0, 4), sidebands = Int32.(fill(1, 4)),
        extras = (; frqsel = Int32(2)),
    )

    flat = (
        vis = vis_da, weights = weights_da, uvw = uvw_da,
        obs_time = obs_t,
        record_scan_name = ["1", "1"],
        record_spw_index = Int32[1, 2],
        baselines = UV.BaselineIndex(
            [(1, 2), (1, 2)], [(1, 2)]; antenna_names = ["AA", "AX"]
        ),
        extra_columns = NamedTuple(),
        antennas = UV.union_antennas(base),
        array_obs = base_meta.array_obs,
        freq_setups = UV.FrequencySetup[fs_a, fs_b],
        source_name = "TEST", ra = 0.0, dec = 0.0,
        basename = "synthetic_2spw",
    )
    return flat, fs_a, fs_b
end

@testset "synthetic two-FREQID flat → UVSet" begin
    UV = Gustavo.UVData
    flat, fs_a, fs_b = synthetic_two_spw_flat()
    uvset = UV.UVSet(flat)

    @test length(UV.branches(uvset)) == 2
    @test haskey(UV.branches(uvset), :src_TEST_spw_0_scan_1)
    @test haskey(UV.branches(uvset), :src_TEST_spw_1_scan_1)
    @test UV.freq_setup(UV.branches(uvset)[:src_TEST_spw_0_scan_1]) == fs_a
    @test UV.freq_setup(UV.branches(uvset)[:src_TEST_spw_1_scan_1]) == fs_b
    @test UV.union_frequency_axis(uvset) == [fs_a, fs_b]
    @test_throws ArgumentError UV.freq_setup(uvset)
end

@testset "multi-FREQID write → load_uvfits round-trip" begin
    UV = Gustavo.UVData
    flat, fs_a, fs_b = synthetic_two_spw_flat()
    uvset = UV.UVSet(flat)
    # Inherit cards from a sibling fixture so the writer has PTYPE / STOKES context.
    UV.register_primary_cards!(uvset, UV.primary_cards(synthetic_uvdata()))

    tmp = tempname() * ".uvfits"
    try
        UV.write_uvfits(tmp, uvset)
        round_set = UV.load_uvfits(tmp)
        @test length(UV.branches(round_set)) == 2
        # Round-trip preserves the two distinct frequency setups; scan_name
        # may renumber via NX row index, but (spw, freq) identity survives.
        round_setups = UV.union_frequency_axis(round_set)
        @test length(round_setups) == 2
        @test UV.channel_freqs(round_setups[1]) == UV.channel_freqs(fs_a)
        @test UV.channel_freqs(round_setups[2]) == UV.channel_freqs(fs_b)
        @test get(round_setups[1].extras, :frqsel, Int32(0)) == Int32(1)
        @test get(round_setups[2].extras, :frqsel, Int32(0)) == Int32(2)
    finally
        isfile(tmp) && rm(tmp)
    end
end

@testset "load_uvfits prefers NX START/END VIS over NX TIME degeneracy" begin
    UV = Gustavo.UVData
    FITS = parentmodule(Card)
    data = synthetic_uvdata()
    tmp_in = tempname() * ".uvfits"
    tmp_out = tempname() * ".uvfits"
    try
        UV.write_uvfits(tmp_in, data)
        fid = FITS.fits(tmp_in)
        hdus = [fid[i] for i in 1:length(fid)]
        nx_hdu = last(hdus)
        nx = nx_hdu.data
        nrow = length(collect(nx.TIME))
        degenerate_nx = (
            TIME = fill(first(collect(nx.TIME)), nrow),
            var"TIME INTERVAL" = collect(nx.var"TIME INTERVAL"),
            var"SOURCE ID" = collect(nx.var"SOURCE ID"),
            SUBARRAY = collect(nx.SUBARRAY),
            var"FREQ ID" = collect(nx.var"FREQ ID"),
            var"START VIS" = collect(nx.var"START VIS"),
            var"END VIS" = collect(nx.var"END VIS"),
        )
        hdus[end] = FITS.HDU(FITS.Bintable, degenerate_nx, nx_hdu.cards)
        FITS.write(tmp_out, hdus)

        round_set = UV.load_uvfits(tmp_out)
        @test length(UV.branches(round_set)) == 2
        @test sort(UV.scan_ids(round_set, "TEST")) == ["1", "2"]
    finally
        isfile(tmp_in) && rm(tmp_in; force = true)
        isfile(tmp_out) && rm(tmp_out; force = true)
    end
end

@testset "AIPS BASELINE encoding: 255-antenna limit on write" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    leaf = first(values(UV.branches(base)))
    info = UV.metadata(leaf)
    bad_bls = UV.BaselineIndex(
        [(1, 256), (1, 256)], [(1, 256)]; antenna_names = ["A", "B"]
    )
    bad_info = UV.update(info; baselines = bad_bls)
    bad_leaf = UV._build_leaf(
        leaf[:vis], leaf[:weights], leaf[:uvw];
        partition_info = bad_info,
    )
    branches = Gustavo.UVData.DimensionalData.TreeDict(:src_TEST_spw_0_scan_1 => bad_leaf)
    bad_set = Gustavo.UVData.DimensionalData.rebuild(base; branches = branches)
    tmp = tempname() * ".uvfits"
    try
        @test_throws Exception UV.write_uvfits(tmp, bad_set)
    finally
        isfile(tmp) && rm(tmp)
    end
end

@testset "primary_cards lives in FITS-extension WeakKeyDict" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    # Fixture registered cards explicitly — the accessor should return them.
    cards = UV.primary_cards(base)
    @test !isempty(cards)
    # Cards follow rebuild-style operations (e.g. select_source).
    selected = UV.select_source(base, "TEST")
    @test UV.primary_cards(selected) == cards
end

@testset "Phase 1.6 invariants: BaselineIndex + UVMetadata" begin
    UV = Gustavo.UVData
    # decode_baseline must not exist in the UVData public namespace.
    @test !isdefined(UV, :decode_baseline)
    # BaselineIndex must not carry AIPS-only fields.
    @test !(:codes in fieldnames(UV.BaselineIndex))
    @test !(:unique_codes in fieldnames(UV.BaselineIndex))
    @test :pairs_per_record in fieldnames(UV.BaselineIndex)
    # primary_cards must not be a UVMetadata field.
    @test !(:primary_cards in fieldnames(UV.UVMetadata))
end

@testset "AIPS leak check: setup_name is MSv4, no record_freqid" begin
    UV = Gustavo.UVData
    fs = UV.freq_setup(first(values(UV.branches(synthetic_uvdata()))))
    @test UV.setup_name(fs) == "FRQSEL_1" || startswith(UV.setup_name(fs), "spw_")
    # `flat.record_freqid` no longer exists; constructing a UVSet from a flat
    # tuple with that legacy key should fail (renamed to record_spw_index).
    base = synthetic_uvdata()
    bl = first(values(UV.branches(base)))
    info = UV.metadata(bl)
    bad_flat = (
        vis = bl[:vis], weights = bl[:weights], uvw = bl[:uvw],
        obs_time = collect(UV.obs_time(bl)),
        record_scan_name = ["1", "1"],
        record_freqid = Int32[1, 1],   # legacy name — should not be picked up
        baselines = info.baselines,
        extra_columns = NamedTuple(),
        antennas = UV.union_antennas(base),
        array_obs = UV.metadata(base).array_obs,
        freq_setups = UV.FrequencySetup[UV.freq_setup(bl)],
        source_name = "TEST", ra = 0.0, dec = 0.0,
    )
    @test_throws Exception UV.UVSet(bad_flat)
end

@testset "scan_name accessors live on PartitionInfo" begin
    UV = Gustavo.UVData
    data = synthetic_uvdata()
    for (k, leaf) in UV.branches(data)
        # Each leaf maps to exactly one (source, scan, sub_scan); the scan
        # label is a scalar field on PartitionInfo.
        @test UV.scan_name(leaf) isa AbstractString
        @test UV.scan_name(leaf) == UV.primary_scan_name(leaf)
        @test UV.scan_intents(leaf) isa Vector{String}
        @test UV.sub_scan_name(leaf) == ""
    end
end

@testset "scan_window derived from Ti axis" begin
    UV = Gustavo.UVData
    data = synthetic_uvdata()
    for (_, leaf) in UV.branches(data)
        lo, hi = UV.scan_window(leaf)
        ti = collect(UV.obs_time(leaf))
        @test lo == minimum(ti)
        @test hi == maximum(ti)
    end
end

@testset "participating_antennas surfaces sub-array participation" begin
    UV = Gustavo.UVData
    data = synthetic_uvdata()
    for (_, leaf) in UV.branches(data)
        ants = UV.participating_antennas(leaf)
        @test ants isa Vector{String}
        # Participating set ⊆ root antenna union.
        @test issubset(Set(ants), Set(UV.union_antennas(data).name))
    end
end

@testset "sub_scan_name disambiguates colliding (source, scan)" begin
    # Mirrors xradio's SUB_SCAN_NUMBER partitioning: two sub-arrays
    # observing the same source at the same scan would collide on
    # `:<source>_scan_<n>`. Setting `sub_scan_name` distinctly on each leaf
    # disambiguates via `:<source>_scan_<n>_<sub>`.
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    @test UV.partition_key(; source_key = :TEST, scan_name = "1") === :TEST_spw_0_scan_1
    @test UV.partition_key(; source_key = :TEST, scan_name = "1", sub_scan_name = "A") ===
        :TEST_spw_0_scan_1_A
    @test UV.partition_key(; source_key = :TEST, scan_name = "1", sub_scan_name = "A") !==
        UV.partition_key(; source_key = :TEST, scan_name = "1", sub_scan_name = "B")

    # Build a sibling leaf with a distinct sub_scan_name on top of an
    # existing leaf and confirm both coexist after merge_uvsets-style assembly.
    branches = Gustavo.UVData.DimensionalData.TreeDict()
    for (k, leaf) in UV.branches(base)
        branches[k] = leaf
        info = UV.metadata(leaf)
        if info.scan_name == "1"
            sub_info = UV.update(info; sub_scan_name = "B")
            sub_leaf = UV._build_leaf(
                leaf[:vis], leaf[:weights], leaf[:uvw];
                partition_info = sub_info,
            )
            sub_key = UV.partition_key(sub_info)
            branches[sub_key] = sub_leaf
        end
    end
    multi_sa = Gustavo.UVData.DimensionalData.rebuild(base; branches = branches)
    @test haskey(UV.branches(multi_sa), :src_TEST_spw_0_scan_1)
    @test haskey(UV.branches(multi_sa), :src_TEST_spw_0_scan_1_B)
    # Multi-subarray write (Phase 1.7+2.5) succeeds when leaves share antennas;
    # multiple AN HDUs are emitted when antennas differ.
    tmp = tempname() * ".uvfits"
    try
        UV.register_primary_cards!(multi_sa, UV.primary_cards(base))
        @test UV.write_uvfits(tmp, multi_sa) == tmp
        @test isfile(tmp)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
end

@testset "Phase 1.7 invariants: per-leaf antennas, ArrayConfig deletion" begin
    UV = Gustavo.UVData
    @test !isdefined(UV, :ArrayConfig)
    @test fieldnames(UV.UVMetadata) == (:array_obs,)
    @test :antennas in fieldnames(UV.PartitionInfo)
    @test :subarray_name in fieldnames(UV.PartitionInfo)
end

@testset "antennas accessor + AntennaTable shared by reference" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    leaves_v = collect(values(UV.branches(base)))
    leaf1 = first(leaves_v)
    leaf2 = last(leaves_v)
    # Single-subarray fixture: every leaf points at the same AntennaTable
    # instance (shared reference). Mutation requires constructing a new
    # table — silent inconsistency requires explicit work.
    @test UV.antennas(leaf1) === UV.antennas(leaf2)
end

@testset "union_antennas happy path / conflict path" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    @test UV.union_antennas(base).name == UV.antennas(first(values(UV.branches(base)))).name

    # Synthesize a 2-leaf UVSet where the *same* antenna name has different
    # station_xyz across leaves → conflict.
    leaf = first(values(UV.branches(base)))
    info = UV.metadata(leaf)
    bad_ant = UV.Antenna(;
        name = "AA", station_xyz = [1.0, 0.0, 0.0],
        mount = UV.MountAltAz(),
        nominal_basis = (RPol(), LPol()),
        response = Diagonal(ones(ComplexF32, 2)),
        pol_angles = (0.0f0, 0.0f0),
    )
    other_ant = info.antennas[2]
    bad_table = UV.AntennaTable(
        StructArray([bad_ant, other_ant]),
        UV.array_xyz(info.antennas), UV.array_name(info.antennas),
        UV.extras(info.antennas),
    )
    bad_info = UV.update(info; antennas = bad_table)
    bad_leaf = UV._build_leaf(
        leaf[:vis], leaf[:weights], leaf[:uvw];
        partition_info = bad_info,
    )
    branches = Gustavo.UVData.DimensionalData.TreeDict()
    for (k, v) in UV.branches(base)
        branches[k] = v
    end
    branches[:src_TEST_spw_0_sub_1_scan_1] = bad_leaf
    multi = Gustavo.UVData.DimensionalData.rebuild(base; branches = branches)
    @test_throws ErrorException UV.union_antennas(multi)
end

@testset "union_pol_products happy path / conflict path" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    @test UV.union_pol_products(base) == UV.pol_products(first(values(UV.branches(base))))
end

@testset "multi-AN-extver round-trip (different antenna sets per leaf)" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    leaf = first(values(UV.branches(base)))
    info = UV.metadata(leaf)
    # Build a perturbed AntennaTable with one antenna's station_xyz shifted.
    orig = info.antennas
    perturbed_v = [
        UV.Antenna(;
                name = orig.name[i],
                station_xyz = i == 1 ? orig.station_xyz[i] .+ 1.0 : orig.station_xyz[i],
                mount = orig.mount[i], nominal_basis = orig.nominal_basis[i],
                response = orig.response[i], pol_angles = orig.pol_angles[i],
            ) for i in 1:length(orig)
    ]
    perturbed = UV.AntennaTable(
        StructArray(perturbed_v),
        UV.array_xyz(orig), UV.array_name(orig), UV.extras(orig),
    )
    # Hand the second leaf the perturbed table; tag with sub_name so the
    # partition key separates the two subarrays.
    branches = Gustavo.UVData.DimensionalData.TreeDict()
    for (k, l) in UV.branches(base)
        l_info = UV.metadata(l)
        if l_info.scan_name == "2"
            new_info = UV.update(l_info; antennas = perturbed, subarray_name = "sub_1")
            new_l = UV._build_leaf(
                l[:vis], l[:weights], l[:uvw];
                partition_info = new_info,
            )
            branches[UV.partition_key(new_info)] = new_l
        else
            branches[k] = l
        end
    end
    multi = Gustavo.UVData.DimensionalData.rebuild(base; branches = branches)

    tmp = tempname() * ".uvfits"
    try
        UV.write_uvfits(tmp, multi)
        round_set = UV.load_uvfits(tmp)
        # Two leaves come back; round-trip preserves distinct antenna tables.
        @test length(UV.branches(round_set)) == 2
        round_tables = unique([UV.antennas(l) for (_, l) in UV.branches(round_set)])
        @test length(round_tables) == 2
        # The shifted antenna's station_xyz survives round-trip.
        sub_leaf_round = first(
            filter(
                ((_, l),) -> UV.metadata(l).subarray_name != "",
                collect(UV.branches(round_set)),
            )
        )[2]
        @test sub_leaf_round !== nothing
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
end

@testset "UVFITS write rejects mixed pol products" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    leaf = first(values(UV.branches(base)))
    info = UV.metadata(leaf)
    # Reshape vis to a different pol set (drop the cross-hands so leaves
    # disagree across the 2 scans).
    reduced_vis = leaf[:vis][Pol = 1:2]
    reduced_w = leaf[:weights][Pol = 1:2]
    reduced_uvw = leaf[:uvw]
    reduced_leaf = UV._build_leaf(
        reduced_vis, reduced_w, reduced_uvw;
        partition_info = info,
    )
    branches = Gustavo.UVData.DimensionalData.TreeDict()
    for (k, l) in UV.branches(base)
        branches[k] = l
    end
    # Replace the second leaf with the reduced-pol leaf.
    keys_v = collect(keys(branches))
    branches[keys_v[2]] = reduced_leaf
    mixed = Gustavo.UVData.DimensionalData.rebuild(base; branches = branches)
    tmp = tempname() * ".uvfits"
    try
        @test_throws ErrorException UV.write_uvfits(tmp, mixed)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
end

@testset "ANTAB parser: GAIN + TSYS layouts" begin
    BP = Gustavo.UVData
    text = """
    GAIN AA ELEV DPFU = 0.031000 POLY = 1.0 /
    GAIN MG ELEV DPFU = 0.0179, 0.0168 POLY = 0.727119, 0.00947339, -0.00008222 /
    GAIN NN ELEV DPFU = 1.000000, 1.000000 POLY = 1.0 /

    TSYS AA  FT=1.0  TIMEOFF=0
    INDEX = 'L1|R1', 'L2|R2'
    /
    100 12:00:00     2.5    3.5
    100 12:30:00     2.7    3.7
    /
    TSYS MG timeoff= 0.0  FT = 1.0  INDEX = 'R1:32', 'L1:32' /
    100 12:00:00     200.0  220.0
    100 13:00:00     180.0  210.0
    /
    TSYS NN timeoff= 0.0  FT = 1.0  INDEX = 'R1:32', 'L1:32' /
    100 12:00:00     500.0  600.0
    100 13:00:00     520.0  580.0
    /
    """
    path, io = mktemp()
    try
        write(io, text); close(io)
        new_path = path * "_e22a26_b1_proc.AN"
        cp(path, new_path; force = true)
        antab = BP.load_antab(new_path)

        @test length(antab.stations) == 3
        @test haskey(antab, "AA") && haskey(antab, "MG") && haskey(antab, "NN")
        @test antab.year == 2022
        @test antab.track_label == "e22a26_b1"

        # Per-channel ALMA block, paired-pol columns.
        st_aa = antab["AA"]
        @test st_aa.nchannels == 2
        @test length(st_aa.tsys.times) == 2
        @test BP.tsys_at(st_aa, st_aa.tsys.times[1], 1, :R) ≈ 2.5
        @test BP.tsys_at(st_aa, st_aa.tsys.times[1], 1, :L) ≈ 2.5    # paired
        @test BP.tsys_at(st_aa, st_aa.tsys.times[1], 2, :L) ≈ 3.5

        # Aggregate per-pol columns broadcast across channels.
        st_mg = antab["MG"]
        @test st_mg.nchannels == 0
        @test BP.tsys_at(st_mg, st_mg.tsys.times[1], 1, :R) ≈ 200.0
        @test BP.tsys_at(st_mg, st_mg.tsys.times[1], 17, :L) ≈ 220.0

        # Time interpolation midpoint.
        midpoint = st_mg.tsys.times[1] + (st_mg.tsys.times[2] - st_mg.tsys.times[1]) ÷ 2
        @test BP.tsys_at(st_mg, midpoint, 1, :R) ≈ 190.0 atol = 0.5

        # Out-of-range time returns NaN.
        @test isnan(BP.tsys_at(st_mg, st_mg.tsys.times[end] + Dates.Hour(2), 1, :R))

        # Elevation-gain Horner.
        @test BP.elevation_gain(st_aa.gain, 30.0) ≈ 1.0
        @test BP.elevation_gain(st_mg.gain, 0.0) ≈ st_mg.gain.poly[1]
        @test BP.elevation_gain(st_mg.gain, 45.0) ≈
            (st_mg.gain.poly[1] + st_mg.gain.poly[2] * 45 + st_mg.gain.poly[3] * 45^2)

        # DPFU broadcasting for single-value GAIN rows.
        @test st_aa.gain.dpfu == (0.031, 0.031)
        @test st_mg.gain.dpfu == (0.0179, 0.0168)

        # Year override and unparseable filename should error.
        antab2 = BP.load_antab(new_path; year = 2017)
        @test antab2.year == 2017
    finally
        isfile(path) && rm(path)
    end
end

@testset "apply_calibration: synthetic UVSet" begin
    UV = Gustavo.UVData
    BP = Gustavo.UVData

    base = synthetic_uvdata()
    leaves_v = collect(values(UV.branches(base)))
    ants = UV.union_antennas(base)
    ant_names = ants.name

    # Build an ANTAB calibration directly so we can pin SEFD values
    # without relying on elevation evaluation. Both stations have flat
    # POLY=[1.0] and DPFU=1.0 so SEFD = Tsys exactly. Cover every leaf's
    # scan window so the per-scan Tsys lookup finds at least one antab
    # row in each.
    rdate = UV.DimensionalData.metadata(base).array_obs.rdate
    base_dt = DateTime(Date(rdate))
    ts_all = unique(sort!(reduce(vcat, [collect(UV.obs_time(l)) for l in leaves_v])))
    isempty(ts_all) && error("synthetic obs_time empty")
    times = [base_dt + Millisecond(round(Int, t * 3_600_000)) for t in ts_all]
    extended_times = [times[1] - Hour(2); times; times[end] + Hour(2)]

    sefd_aa = 100.0
    sefd_ax = 400.0
    function _make_station(name, sefd)
        gain = BP.AntabGainCurve((1.0, 1.0), [1.0])
        # Aggregate per-pol columns: 'R1:32' broadcasts SEFD to all channels.
        cols = [(0, :R), (0, :L)]
        vals = repeat([sefd sefd], length(extended_times), 1)
        ts_series = BP.AntabTsysSeries(extended_times, cols, vals)
        return BP.AntabStation(name, gain, ts_series, 0)
    end
    stations = Dict(
        "AA" => _make_station("AA", sefd_aa),
        "AX" => _make_station("AX", sefd_ax),
    )
    antab = BP.AntabCalibration(
        "synthetic", "synth", 2000, stations,
    )

    # min_elevation_deg = -Inf: this test fakes station_xyz = zeros(3) (elevation
    # ill-defined) and uses a flat gain curve, so we disable the below-horizon
    # cutoff to keep it purely a SEFD-scaling check.
    corr = BP.apply_calibration(base, antab; min_elevation_deg = -Inf)

    # The synthetic `synthetic_uvdata` fakes `station_xyz = zeros(3)` —
    # the elevation calculation will be ill-defined there, but the test
    # antab has POLY=[1.0] so g_E always evaluates to 1 regardless. We
    # only need SEFD = 100 on AA, 400 on AX, hence cal_factor =
    # sqrt(SEFD_a * SEFD_b) = sqrt(100*400) = 200 on the AA-AX baseline.
    expected_factor = sqrt(sefd_aa * sefd_ax)
    for (k, leaf_in) in UV.branches(base)
        leaf_out = UV.branches(corr)[k]
        # Take a finite, weight>0 sample.
        vis_in = parent(leaf_in[:vis])
        vis_out = parent(leaf_out[:vis])
        w_out = parent(leaf_out[:weights])
        idx = findfirst(i -> isfinite(vis_in[i]) && w_out[i] > 0, eachindex(vis_in))
        @test idx !== nothing
        @test abs(vis_out[idx]) ≈ abs(vis_in[idx]) * expected_factor rtol = 1e-6
    end
end

@testset "tsys_in_window rejects outliers outside the scan window" begin
    BP = Gustavo.UVData
    # Three rows: an "in-scan" row, a slew-time outlier outside the window,
    # and another "in-scan" row. The window-mean must average only the
    # in-window rows and never see the outlier.
    base_dt = DateTime(2022, 3, 27, 0, 0, 0)
    times = [base_dt, base_dt + Minute(2), base_dt + Minute(10), base_dt + Minute(20)]
    cols  = [(0, :R), (0, :L)]
    vals  = Float64[
        100.0   120.0;
        110.0   130.0;
        1.0e6   1.0e6;        # slew/outlier — outside the scan window below
        105.0   125.0;
    ]
    st = BP.AntabStation("XX",
        BP.AntabGainCurve((1.0, 1.0), [1.0]),
        BP.AntabTsysSeries(times, cols, vals),
        0,
    )

    # Window covers rows 1 + 2 (mean = 105 R, 125 L), excludes the outlier at +10min.
    t_lo, t_hi = base_dt - Second(1), base_dt + Minute(5)
    @test BP.tsys_in_window(st, t_lo, t_hi, 1, :R) ≈ 105.0
    @test BP.tsys_in_window(st, t_lo, t_hi, 1, :L) ≈ 125.0
    # No rows in window → NaN
    @test isnan(BP.tsys_in_window(st, base_dt + Hour(2), base_dt + Hour(3), 1, :R))
    # Window that covers only the outlier returns the outlier (caller's responsibility)
    @test BP.tsys_in_window(st, base_dt + Minute(8), base_dt + Minute(15), 1, :R) ≈ 1.0e6
end

@testset "apply_calibration: missing station warns" begin
    UV = Gustavo.UVData
    BP = Gustavo.UVData
    base = synthetic_uvdata()

    leaves_v = collect(values(UV.branches(base)))
    rdate = UV.DimensionalData.metadata(base).array_obs.rdate
    base_dt = DateTime(Date(rdate))
    ts_all = unique(sort!(reduce(vcat, [collect(UV.obs_time(l)) for l in leaves_v])))
    times = [base_dt + Millisecond(round(Int, t * 3_600_000)) for t in ts_all]
    extended_times = [times[1] - Hour(2); times; times[end] + Hour(2)]
    aa_only = Dict(
        "AA" => BP.AntabStation(
            "AA",
            BP.AntabGainCurve((1.0, 1.0), [1.0]),
            BP.AntabTsysSeries(
                extended_times, [(0, :R), (0, :L)],
                repeat([100.0 100.0], length(extended_times), 1),
            ),
            0,
        ),
    )
    antab = BP.AntabCalibration("synth", "synth", 2000, aa_only)

    # Suppress the warnings here — the warning behavior is exercised by the
    # missing-station check in `apriori_flux_gains`. We just need to confirm
    # the call succeeds (silently with :ignore) and errors with :error.
    out = BP.apply_calibration(base, antab; on_missing_station = :ignore)
    @test out isa Gustavo.UVData.UVSet
    flux = BP.apriori_flux_gains(base, antab; on_missing_station = :ignore)
    @test all(g -> "AX" in g.missing_stations, values(flux))
    @test_throws ErrorException BP.apply_calibration(
        base, antab; on_missing_station = :error,
    )
end

@testset "apply / mapleaves / flatmap arity overloads" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    leaves_v = collect(values(UV.branches(base)))
    @test length(leaves_v) == 2

    # 1-arg `f(leaf)`
    nleaves_1 = UV.mapleaves(leaf -> length(UV.obs_time(leaf)), base)
    @test nleaves_1 isa OrderedCollections.OrderedDict
    @test all(v == length(UV.obs_time(l)) for ((_, v), l) in zip(nleaves_1, leaves_v))

    # 2-arg `f(leaf, info)`
    nleaves_2 = UV.mapleaves(base) do leaf, info
        (; scan = info.scan_name, n = length(UV.obs_time(leaf)))
    end
    @test all(getproperty(v, :scan) == UV.scan_name(l) for ((_, v), l) in zip(nleaves_2, leaves_v))

    # 3-arg `f(leaf, info, root)` matches the legacy signature.
    nleaves_3 = UV.mapleaves(base) do leaf, info, root
        (; scan = info.scan_name, telescope = root.array_obs.telescope)
    end
    @test all(getproperty(v, :telescope) == UV.metadata(base).array_obs.telescope
              for v in values(nleaves_3))

    # `flatmap` produces a vcat'd Vector
    rows = UV.flatmap(base) do leaf
        info = UV.metadata(leaf)
        [(; scan = info.scan_name, label = lbl) for lbl in UV.baselines(leaf).labels]
    end
    @test rows isa AbstractVector
    @test length(rows) == sum(length(UV.baselines(l).labels) for l in leaves_v)

    # `apply` still returns a UVSet when `f` returns a DimTree.
    out = UV.apply(base) do leaf
        UV.rebuild_visibilities(leaf, leaf[:vis], leaf[:weights])
    end
    @test out isa UV.UVSet
    @test length(UV.branches(out)) == length(UV.branches(base))
end

@testset "BaselineIndex lookup sugar" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    leaf = first(values(UV.branches(base)))
    bls = UV.baselines(leaf)
    @test length(bls) == length(bls.pairs) > 0

    p = bls.pairs[1]
    lbl = bls.labels[1]
    a, b = bls.ant1_names[1], bls.ant2_names[1]
    @test bls[p] == 1
    @test bls[lbl] == 1
    @test bls[(a, b)] == 1
    @test haskey(bls, p) && haskey(bls, lbl) && haskey(bls, (a, b))
    @test !haskey(bls, (-1, -2))
    @test !haskey(bls, "ZZ-ZZ")
    @test_throws KeyError bls[(-1, -2)]
    @test UV.baseline_index(bls, (-1, -2)) == 0
end

@testset "pol_index / pol_at canonicalization" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    leaf = first(values(UV.branches(base)))
    pp = UV.pol_index(leaf, "PP")
    @test pp isa Integer && pp > 0
    # EHT shorthands fold onto the canonical PP/PQ/QP/QQ.
    @test UV.pol_index(leaf, "RR") == pp
    @test UV.pol_index(leaf, "XX") == pp
    @test UV.pol_index(leaf, (RPol(), RPol())) == pp

    @test UV.pol_index(leaf, "QQ") == UV.pol_index(leaf, "LL")
    # Selector roundtrip.
    sel = UV.pol_at("RR")
    @test sel isa DimensionalData.At
    @test getfield(sel, :val) == "PP"  # canonical label
end

@testset "baseline DimStack views" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    leaf = first(values(UV.branches(base)))
    bls = UV.baselines(leaf)
    bl = UV.baseline(leaf, bls.pairs[1])

    @test bl isa DimensionalData.DimStack
    @test Set(keys(bl)) == Set((:vis, :weights, :uvw))

    # Shape & dims
    nch = UV.nchannels(base)
    nti = length(UV.obs_time(leaf))
    npol = length(UV.pol_products(leaf))
    @test size(bl[:vis]) == (nch, nti, npol)
    @test size(bl[:weights]) == (nch, nti, npol)
    @test size(bl[:uvw]) == (nti, 3)

    # Pol selector path matches manual indexing.
    pp_idx = UV.pol_index(leaf, "PP")
    via_selector = parent(bl[:vis][Pol = UV.pol_at("PP")])
    via_manual = parent(view(leaf[:vis], UV.Baseline(bls.pairs[1] |> p -> bls[p])))[:, :, pp_idx]
    @test via_selector == via_manual

    # Metadata carries the antenna identification + scan name.
    md = DimensionalData.metadata(bl)
    @test md.ant1 == bls.ant1_names[1]
    @test md.ant2 == bls.ant2_names[1]
    @test md.label == bls.labels[1]
    @test md.scan_name == UV.scan_name(leaf)

    # Missing baseline → KeyError
    @test_throws KeyError UV.baseline(leaf, "ZZ-ZZ")

    # baselines_per_scan returns OrderedDict keyed by partition.
    per_scan = UV.baselines_per_scan(base, bls.pairs[1])
    @test per_scan isa OrderedCollections.OrderedDict
    @test length(per_scan) == 2  # synthetic_uvdata has 2 leaves, both share this baseline

    # baseline(uvset, bl) concatenates Ti.
    full = UV.baseline(base, bls.pairs[1])
    @test full isa DimensionalData.DimStack
    expected_ti = sum(length(UV.obs_time(l)) for l in values(UV.branches(base)))
    @test size(full[:vis], 2) == expected_ti
    @test size(full[:uvw], 1) == expected_ti
end

# Every extension must precompile and load. An extension method that shares a
# signature with a stub in `src/` is overwritten on load, which precompilation
# rejects outright — so a missing extension here means the package is broken for
# everyone who loads that trigger, not merely missing a feature. Runs last: each
# extension only activates once its trigger package is loaded, and the suite
# loads HDF5 from `synthetic_uvset.jl` rather than at the top of this file.
@testset "extensions load" begin
    for name in (
            :GustavoFITSFilesExt,
            :GustavoHDF5Ext, :GustavoMakieExt,
        )
        @test Base.get_extension(Gustavo, name) !== nothing
    end
end
