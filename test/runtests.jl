using Gustavo
using Test
using LinearAlgebra
using Statistics
using Random
using Dates
using OrderedCollections
using Distributions: LogNormal, Normal, MvNormal
using FITSFiles
import XRadio
using CairoMakie
using DimensionalData
using DimensionalData: DimArray, DimStack, dims, Ti
using Gustavo: Polarization, Frequency, UVW, BaselineID, feed_pairs
using PolarizedTypes: RPol, LPol

# Test helper: reconstruct the legacy off1/off2 index tables from a ComponentPlan.
include("plan_offsets.jl")

include("test_synthetic_ps.jl")
include("test_measurementset.jl")
include("test_uvfits.jl")
include("test_uvfits_write.jl")
include("test_ms_geometry.jl")
include("test_autocorrelations.jl")
include("test_apriori.jl")

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
# Reuses _build_fringe_ps + CAL/FP aliases from test_pipeline.jl.
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

# The weight convention behind the bandpass SNRs and gain-estimate weights.
include("test_noise_convention.jl")

# The AdhocPhase step + output sink: multi-scan solves of the three steps in
# turn, and calibrate.
include("test_smoother_step.jl")
include("test_gauge_invariance.jl")

# Fringe diagnostics + Makie plot stubs (Phase 8).
include("test_fringe_diagnostics.jl")

# fringe_station_solutions θ-decode + the `rel_time` model option.
# Reuses _build_fringe_ps + CAL/FP aliases from test_pipeline.jl.
include("test_fringe_station_solutions.jl")

# Per-station weight correction (scale_weights!) through a full solve.
# Reuses _build_fringe_ps + the FP/CAL aliases from test_pipeline.jl.
include("test_weight_scale.jl")

# The executor seam: the group scheduler under each outer scheduler — dispatch
# order, task cap, and bit-identical θ/outputs whichever one runs the pass.
include("test_executors.jl")

@testset "Gustavo.jl" begin
    for sub in (:Calibration, :Fring)
        @test isdefined(Gustavo, sub)
        @test getfield(Gustavo, sub) isa Module
    end
end

@testset "top-level export surface" begin
    top = names(Gustavo)

    # A bare `using Gustavo` spans the production path: read a set, then fit and
    # calibrate it.
    for n in (:load_uvfits, :fit, :calibrate)
        @test n in top
    end

    # The corrections, and reading scan groups in a step.
    for n in (
            :normalize_by_autocorrelations!, :scale_weights!, :calibrate!,
            :each_group, :ExecutionConfig,
        )
        @test n in top
    end

    # The two submodules, named at the top level.
    for n in (:Calibration, :Fring)
        @test n in top
    end

    # Every axis name a stored or returned array can carry, so scripts index
    # them with a bare `using Gustavo`. `Ti` is DimensionalData's dim under
    # both names — the same binding, so no ambiguity when both are loaded.
    for n in (:Polarization, :Frequency, :AntennaName, :BaselineID, :Ti, :UVW, :Feed, :Scan, :AntennaPair, :FeedPair)
        @test n in top
        @test getproperty(Gustavo, n) <: DimensionalData.Dimension
    end
    @test Gustavo.Ti === DimensionalData.Ti
    # Gustavo and XRadio share these axis types, so a kernel indexing by name
    # reads a Measurement Set's layers directly.
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

@testset "Weighted LSQ accepts mixed-precision RHS" begin
    # `weights` may be Float32 while `design_matrices` builds `A` in Float64; the
    # solve promotes them to a shared eltype.
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

@testset "Source name sanitization" begin
    ext = Base.get_extension(Gustavo, :GustavoFITSFilesExt)
    @test ext.sanitize_source("TEST") == :src_TEST
    @test ext.sanitize_source("3C273") == :src_3C273    # digit-leading
    @test ext.sanitize_source("Sgr A*") == :src_Sgr_A_  # non-identifier chars
    @test ext.sanitize_source("NGC 4486") == :src_NGC_4486
    @test ext.sanitize_source("") == :src_unknown
    @test ext.sanitize_source("  ") == :src_unknown
end

@testset "feed_pairs of a solver cube" begin
    vis = DimArray(zeros(ComplexF32, 4, 2), (Polarization([(1, 1), (1, 2), (2, 1), (2, 2)]), Ti([0.0, 1.0])))
    @test Gustavo.feed_pairs(vis) == [(1, 1), (1, 2), (2, 1), (2, 2)]
end

@testset "a Measurement Set's products resolve through each antenna's receptors" begin
    Testing = XRadio.Testing
    names = ["SMA", "LMT", "ALMA"]
    ant = Testing.antenna(names)
    types = parent(ant[:polarization_type])
    types[:, 2] .= ["L", "R"]
    types[:, 3] .= ["X", "Y"]
    ms = Testing.measurement_set(; antennas = names, antenna_xds = ant)
    @test XRadio.polarizations(ms) == ["RR", "RL", "LR", "LL"]
    @test XRadio.baselines(ms) == [("SMA", "LMT"), ("SMA", "ALMA"), ("LMT", "ALMA")]

    pairs = Gustavo.feed_pairs(ms)
    # SMA–LMT: LMT's R is its second receptor.
    @test pairs[:, 1] == [(1, 2), (1, 1), (2, 2), (2, 1)]
    # SMA–ALMA: ALMA has no R/L receptor, so R is its first and L its second.
    @test pairs[:, 2] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    # LMT–ALMA
    @test pairs[:, 3] == [(2, 1), (2, 2), (1, 1), (1, 2)]

    types[:, 3] .= ["R", "X"]
    @test_throws "product letter `L` names no receptor of antenna `ALMA`, whose receptors are R, X" Gustavo.feed_pairs(
        Testing.measurement_set(; antennas = names, antenna_xds = ant)
    )
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
