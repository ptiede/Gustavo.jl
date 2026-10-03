using Gustavo
using Test
using LinearAlgebra
using Statistics
using Random
using StructArrays
using Dates
using OrderedCollections
using Distributions: LogNormal, Normal, MvNormal
using FITSFiles
import XRadio
using CairoMakie
using DimensionalData
using DimensionalData: DimArray, DimStack, dims, Ti
using Gustavo.UVData: Polarization, Frequency, UVW, BaselineID, UVSet, pol_products, feed_pairs
using Gustavo.UVData: antennas, baselines, source_name, scan_name, frequencies, timestamps
using PolarizedTypes: RPol, LPol

# Test helper: reconstruct the legacy off1/off2 index tables from a ComponentPlan.
include("plan_offsets.jl")

include("test_synthetic_ps.jl")
include("test_xradio_bridge.jl")
include("test_ms_baselines.jl")
include("test_ms_geometry.jl")
include("test_autocorrelations.jl")

# FLAG and WEIGHT as independent layers.
include("test_flags.jl")

# The `AbstractArray` promise: generic where it is made, declared where it is not.
include("test_generic_axes.jl")

include("test_calibration.jl")

# Gauge conventions: which constraint fixes each component's additive freedom.
include("test_gauge.jl")

include("test_antenna_tables.jl")

# Per-baseline FFT fringe search (Phase 3 of the fringe-fitter refactor).
include("test_fringe_search.jl")

# Per-feed stationization with closure (Phase 4 of the fringe-fitter refactor).
include("test_stationize.jl")

# OU / Matérn-1/2 state-space phase smoother primitives (underpins adhoc :gp).
include("test_statespace.jl")

# Fitting a bandpass block under a component prior (the OU prior rides the
# primitives above).
include("test_prior_fits.jl")

# Globally-closing adhoc phasing (Phase 5 of the fringe-fitter refactor).
include("test_adhoc.jl")

# End-to-end CalibrationSolution + fit pipeline (Phase 6).
include("test_pipeline.jl")

# The calibration surface: fit and calibrate, gauges, provenance, defaults.
# Reuses _build_fringe_ps + CAL/FP/UVP aliases from test_pipeline.jl.
include("test_pipeline_config.jl")

# Solve-step interface (step protocol, selections, solutions of several steps,
# fit/calibrate). Reuses the same aliases.
include("test_interface.jl")

# Reading scan groups, and the corrections.
include("test_each_group.jl")
include("test_corrections.jl")
include("test_mapsets.jl")
include("test_group_tables.jl")
include("test_axis_order.jl")

# The BaselineFringeFit step on the new engine (M3 gates): θ ≡ frozen stage A,
# cross-feed rate opt-in, fit-on-subset masking, corrections before the step.
include("test_fringe_step.jl")

# The Bandpass step on the new engine (M4 gates): fringe+bandpass θ
# vs the frozen monolith, step-selection extraction + portable calibrate!.
include("test_bandpass_step.jl")

# Bandpass(smoother = JointSmoother()): the alternating complex-visibility +
# per-scan source-coherence solve, vs. the closure-based PerTrackSmoother
# default, and the shape specs acting as priors inside its gain update.
include("test_joint_bandpass.jl")

# The AdhocPhase step + output sink: multi-scan solves of the three steps in
# turn, and calibrate.
include("test_smoother_step.jl")

# Fringe diagnostics + Makie plot stubs (Phase 8).
include("test_fringe_diagnostics.jl")

# fringe_station_solutions θ-decode + the `rel_time` model option.
# Reuses _build_fringe_ps + CAL/FP/UVP aliases from test_pipeline.jl.
include("test_fringe_station_solutions.jl")

# Per-station weight correction (scale_weights!) through a full solve.
# Reuses _build_fringe_ps + the FP/CAL/UVP aliases from test_pipeline.jl.
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
    vis = DimArray(vis, (Ti(obs_time_synth), Polarization(pol_labels_synth), Frequency(channel_freqs_synth)))
    weights = DimArray(weights, (Ti(obs_time_synth), Polarization(pol_labels_synth), Frequency(channel_freqs_synth)))
    uvw = DimArray(zeros(Float32, 2, 3), (Ti(obs_time_synth), UVW(["U", "V", "W"])))

    UV = Gustavo.UVData
    nominal_basis_v = [(RPol(), LPol()), (RPol(), LPol())]
    pol_angles_v = [(0.0f0, 0.0f0), (0.3f0, 1.2f0)]
    antennas_v = [
        UV.Antenna(;
            name = "AA",
            station_xyz = zeros(3),
            mount = UV.MountAltAz(),
            nominal_basis = nominal_basis_v[1],
            pol_angles = pol_angles_v[1],
        ),
        UV.Antenna(;
            name = "AX",
            station_xyz = zeros(3),
            mount = UV.MountNasmythR((1.5, 0.0, 0.0)),
            nominal_basis = nominal_basis_v[2],
            pol_angles = pol_angles_v[2],
        ),
    ]
    antennas = UV.AntennaTable(
        StructArray(antennas_v), "TEST",
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
    return uvset
end

@testset "Gustavo.jl" begin
    for sub in (:UVData, :Calibration, :Fring)
        @test isdefined(Gustavo, sub)
        @test getfield(Gustavo, sub) isa Module
    end
end

@testset "top-level export surface" begin
    top = names(Gustavo)

    # A bare `using Gustavo` spans the production path: read a set, then fit and
    # calibrate it.
    for n in (:UVSet, :load_uvfits, :fit, :calibrate)
        @test n in top
    end

    # The corrections, and reading scan groups in a step.
    for n in (
            :normalize_by_autocorrelations!, :scale_weights!, :flag_channels!, :calibrate!,
            :each_group, :ExecutionConfig,
        )
        @test n in top
    end

    # The three submodules, named at the top level.
    for n in (:UVData, :Calibration, :Fring)
        @test n in top
    end

    # Every axis name a stored or returned array can carry, so scripts index
    # leaves with a bare `using Gustavo`. `Ti` is DimensionalData's dim under
    # both names — the same binding, so no ambiguity when both are loaded.
    for n in (:Polarization, :Frequency, :AntennaName, :BaselineID, :Ti, :UVW, :Feed, :Scan, :AntennaPair, :FeedPair)
        @test n in top
        @test getproperty(Gustavo, n) <: DimensionalData.Dimension
    end
    @test Gustavo.Ti === DimensionalData.Ti
    # A leaf and a Measurement Set subset share their axis types, so a kernel
    # indexing by name reads either.
    for n in (:Frequency, :BaselineID, :Polarization)
        @test getproperty(Gustavo, n) === getproperty(XRadio, n)
    end

    # Solver-internal parameter bookkeeping: still reachable, no longer exported.
    @test !(:ComponentPlan in names(Gustavo.Calibration))
    @test Gustavo.Calibration.ComponentPlan isa Type

    # The term-authoring interface: the hooks a new `AbstractGainTerm`
    # implements, exported as Gustavo's documented extension point.
    for n in (:term_axes, :param_shapes, :term_eval, :freq_coordinate, :time_coordinate)
        @test n in names(Gustavo.Calibration)
    end
    @test !(:nparams_per_block in names(Gustavo.Calibration))
    @test Gustavo.Calibration.nparams_per_block isa Function
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
        @test UV.metadata(leaf).scan_name == string(s)
        @test ndims(parent(leaf[:vis])) == 4
        @test ndims(parent(leaf[:uvw])) == 3
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
    xc = CALIB.ConstrainedWLS(A, iv32, C)(b32)
    @test isfinite(xc[1]) && isfinite(xc[2])

    # A Float32 system stays Float32: the penalty is a tuning quantity and does
    # not set the precision; the constraint rows do.
    A32 = Float32.(A)
    @test eltype(CALIB.weighted_least_squares(A32, b32, iv32)) == Float32
    xr32 = CALIB.weighted_regularized_least_squares(A32, b32, iv32, [0.2, 0.7])
    @test eltype(xr32) == Float32
    @test xr32 ≈ CALIB.weighted_regularized_least_squares(A, b32, iv32, [0.2, 0.7]) rtol = 1.0e-5
    @test eltype(CALIB.weighted_regularized_least_squares(A32, b32, iv32, Float64[1.0 -1.0])) == Float32
    xc32 = CALIB.ConstrainedWLS(A32, iv32, Float32.(C))(b32)
    @test eltype(xc32) == Float32
    @test xc32 ≈ xc rtol = 1.0e-5
    @test eltype(CALIB.ConstrainedWLS(A32, iv32, C)(b32)) == Float64
end

@testset "Constrained WLS imposes the constraints exactly" begin
    CALIB = Gustavo.Calibration
    rng = MersenneTwister(176)
    # A phase-difference system on 8 nodes gauged by a dense zero-sum row.
    edges = [(u, v) for u in 1:8 for v in (u + 1):8]
    A = zeros(length(edges), 8)
    for (i, (u, v)) in enumerate(edges)
        A[i, u], A[i, v] = 1.0, -1.0
    end
    xtrue = randn(rng, 8)
    xtrue .-= sum(xtrue) / 8
    w = rand(rng, length(edges)) .+ 0.5
    b = A * xtrue .+ 0.01 .* randn(rng, length(edges))
    C = ones(1, 8)
    F = CALIB.ConstrainedWLS(A, w, C)
    x64 = F(b)
    @test abs(sum(x64)) < 1.0e-12
    @test x64 ≈ xtrue atol = 0.05
    x32 = CALIB.ConstrainedWLS(Float32.(A), Float32.(w), Float32.(C))(Float32.(b))
    @test eltype(x32) == Float32
    @test maximum(abs, x32 .- x64) < 1.0e-5
    # One factorization serves any right-hand side.
    @test F(2 .* b) ≈ 2 .* x64
    # With no constraints it is the unconstrained solve.
    @test CALIB.ConstrainedWLS(A[:, 1:7], w, zeros(0, 7))(b) ≈ CALIB.weighted_least_squares(A[:, 1:7], b, w)

    @test_throws "linearly dependent" CALIB.ConstrainedWLS(A, w, [C; 2 .* C])
    unobserved = copy(A)
    unobserved[:, 8] .= 0
    @test_throws "does not determine" CALIB.ConstrainedWLS(unobserved, w, [1.0 zeros(1, 7)])
end

@testset "A factored WLS solver serves any right-hand side" begin
    CALIB = Gustavo.Calibration
    rng = MersenneTwister(1761)
    A = randn(rng, 12, 4)
    w = rand(rng, 12) .+ 0.5
    b = randn(rng, 12)
    for T in (Float64, Float32)
        solve_ls = CALIB.FactoredWLS(T.(A), T.(w))
        x = copy(solve_ls(T.(b)))
        @test eltype(x) == T
        @test x ≈ CALIB.weighted_least_squares(A, b, w) rtol = 10 * eps(T)
        @test solve_ls(2 .* T.(b)) ≈ 2 .* x
    end
    @test eltype(CALIB.FactoredWLS{Float64}(Float32.(A), Float32.(w))(b)) == Float64
    @test_throws DimensionMismatch CALIB.FactoredWLS(A, w[1:11])
    @test_throws "does not determine" CALIB.FactoredWLS([A zeros(12)], w)
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

    # Per-leaf Polarization slice on a UVSet: pull leaf, then DimTree's selector.
    leaf1 = UV.branches(raw)[UV.partition_key(; source_key = :src_TEST, scan_name = "1")]
    sliced = leaf1[Polarization = At("PP")]
    @test sliced isa DimensionalData.AbstractDimTree
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
        pol_angles = (0.0f0, 0.0f0),
    )
    @test ant() == ant()
    @test hash(ant()) == hash(ant())
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
    vis_da = DimArray(vis, (Ti(obs_t), Polarization(pol_lab), Frequency(chan_freq)))
    weights_da = DimArray(weights, (Ti(obs_t), Polarization(pol_lab), Frequency(chan_freq)))
    uvw_da = DimArray(zeros(Float32, 2, 3), (Ti(obs_t), UVW(["U", "V", "W"])))

    # Reuse antennas / array_obs from the single-source fixture. Same
    # structure, just two SPWs.
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
        antennas = UV.metadata(first(values(UV.branches(base)))).antennas,
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
    @test UV.metadata(UV.branches(uvset)[:src_TEST_spw_0_scan_1]).freq_setup == fs_a
    @test UV.metadata(UV.branches(uvset)[:src_TEST_spw_1_scan_1]).freq_setup == fs_b
end

@testset "Phase 1.6 invariants: BaselineIndex + UVMetadata" begin
    UV = Gustavo.UVData
    # decode_baseline must not exist in the UVData public namespace.
    @test !isdefined(UV, :decode_baseline)
    # BaselineIndex must not carry AIPS-only fields.
    @test !(:codes in fieldnames(UV.BaselineIndex))
    @test !(:unique_codes in fieldnames(UV.BaselineIndex))
    @test :pairs_per_record in fieldnames(UV.BaselineIndex)
end

@testset "AIPS leak check: setup_name is MSv4, no record_freqid" begin
    UV = Gustavo.UVData
    fs = UV.metadata(first(values(UV.branches(synthetic_uvdata())))).freq_setup
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
        antennas = info.antennas,
        array_obs = UV.metadata(base).array_obs,
        freq_setups = UV.FrequencySetup[info.freq_setup],
        source_name = "TEST", ra = 0.0, dec = 0.0,
    )
    @test_throws Exception UV.UVSet(bad_flat)
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
    # existing leaf and confirm both coexist in one set.
    branches = Gustavo.UVData.DimensionalData.TreeDict()
    for (k, leaf) in UV.branches(base)
        branches[k] = leaf
        info = UV.metadata(leaf)
        if info.scan_name == "1"
            sub_info = UV.update(info; sub_scan_name = "B")
            sub_leaf = UV._build_leaf(
                leaf[:vis], leaf[:weights], leaf[:uvw], leaf[:flags];
                partition_info = sub_info,
            )
            sub_key = UV.partition_key(sub_info)
            branches[sub_key] = sub_leaf
        end
    end
    multi_sa = Gustavo.UVData.DimensionalData.rebuild(base; branches = branches)
    @test haskey(UV.branches(multi_sa), :src_TEST_spw_0_scan_1)
    @test haskey(UV.branches(multi_sa), :src_TEST_spw_0_scan_1_B)
end

@testset "Phase 1.7 invariants: per-leaf antennas, ArrayConfig deletion" begin
    UV = Gustavo.UVData
    @test !isdefined(UV, :ArrayConfig)
    @test fieldnames(UV.UVMetadata) == (:array_obs,)
    @test :antennas in fieldnames(UV.PartitionInfo)
    @test :subarray_name in fieldnames(UV.PartitionInfo)
end

@testset "leaves share one AntennaTable" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    leaves_v = collect(values(UV.branches(base)))
    leaf1 = first(leaves_v)
    leaf2 = last(leaves_v)
    # Single-subarray fixture: every leaf points at the same AntennaTable
    # instance (shared reference). Mutation requires constructing a new
    # table — silent inconsistency requires explicit work.
    @test UV.metadata(leaf1).antennas === UV.metadata(leaf2).antennas
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

@testset "tsys_in_window rejects outliers outside the scan window" begin
    BP = Gustavo.UVData
    # Three rows: an "in-scan" row, a slew-time outlier outside the window,
    # and another "in-scan" row. The window-mean must average only the
    # in-window rows and never see the outlier.
    base_dt = DateTime(2022, 3, 27, 0, 0, 0)
    times = [base_dt, base_dt + Minute(2), base_dt + Minute(10), base_dt + Minute(20)]
    cols = [(0, :R), (0, :L)]
    vals = Float64[
        100.0   120.0;
        110.0   130.0;
        1.0e6   1.0e6;        # slew/outlier — outside the scan window below
        105.0   125.0;
    ]
    st = BP.AntabStation(
        "XX",
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

@testset "BaselineIndex lookup sugar" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    leaf = first(values(UV.branches(base)))
    bls = UV.metadata(leaf).baselines
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

@testset "pol_index / pol_at select by feed pair" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    vis = first(values(UV.branches(base)))[:vis]
    @test UV.feed_pairs(vis) == [(1, 1), (1, 2), (2, 1), (2, 2)]
    @test [UV.pol_index(vis, fp) for fp in UV.feed_pairs(vis)] == 1:4
    @test_throws KeyError UV.pol_index(vis, (1, 3))
    @test_throws MethodError UV.pol_index(vis, "RR")

    sel = UV.pol_at(vis, (2, 1))
    @test sel isa DimensionalData.At
    @test getfield(sel, :val) == UV.pol_products(vis)[3]

    @test_throws "P (feed 1) and Q (feed 2)" UV._feed_pairs(["RR"])
end

@testset "a Measurement Set's products resolve through each antenna's receptors" begin
    UV = Gustavo.UVData
    Testing = XRadio.Testing
    names = ["SMA", "LMT", "ALMA"]
    ant = Testing.antenna(names)
    types = parent(ant[:polarization_type])
    types[:, 2] .= ["L", "R"]
    types[:, 3] .= ["X", "Y"]
    ms = Testing.measurement_set(; antennas = names, antenna_xds = ant)
    @test UV.pol_products(ms) == ["RR", "RL", "LR", "LL"]
    @test UV.baselines(ms).pairs == [(1, 2), (1, 3), (2, 3)]

    pairs = UV.feed_pairs(ms)
    # SMA–LMT: LMT's R is its second receptor.
    @test pairs[:, 1] == [(1, 2), (1, 1), (2, 2), (2, 1)]
    # SMA–ALMA: ALMA has no R/L receptor, so R is its first and L its second.
    @test pairs[:, 2] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    # LMT–ALMA
    @test pairs[:, 3] == [(2, 1), (2, 2), (1, 1), (1, 2)]

    order, perm = UV._feed_permutation(pairs)
    @test order == [(1, 1), (1, 2), (2, 1), (2, 2)]
    @test perm[:, 1] == [2, 1, 4, 3]
    @test all(bi -> pairs[perm[:, bi], bi] == order, axes(pairs, 2))

    types[:, 3] .= ["R", "X"]
    @test_throws "product letter `L` names no receptor of antenna `ALMA`, whose receptors are R, X" UV.feed_pairs(
        Testing.measurement_set(; antennas = names, antenna_xds = ant)
    )
    @test_throws "every baseline must relate each of" UV._feed_permutation([(1, 1) (1, 1); (2, 2) (1, 2)])
end

# Every extension must precompile and load. An extension method that shares a
# signature with a stub in `src/` is overwritten on load, which precompilation
# rejects outright — so a missing extension here means the package is broken for
# everyone who loads that trigger, not merely missing a feature. Runs last: each
# extension only activates once its trigger package is loaded.
@testset "extensions load" begin
    for name in (:GustavoFITSFilesExt, :GustavoMakieExt)
        @test Base.get_extension(Gustavo, name) !== nothing
    end
end
