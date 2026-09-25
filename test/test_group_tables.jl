# ── A scan group's sums, and the steps that read it ──────────────────────────
#
# The bandpass and adhoc kernels sum each member of a scan group by label; each
# step fits once per group.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")
@isdefined(CAL) || const CAL = Gustavo.Calibration
@isdefined(FP) || const FP = Gustavo.Fring

_swap_baselines(ms) = Gustavo._with_layers(
    ms; baseline_antenna1_name = ms[:baseline_antenna2_name],
    baseline_antenna2_name = ms[:baseline_antenna1_name],
)

@testset "weighted_sums reduces named dimensions" begin
    rng = MersenneTwister(21)
    axs = (
        Polarization(["RR", "RL", "LR", "LL"]), Frequency([1.0e9, 1.1e9, 1.2e9]),
        BaselineID(0:4), Ti([0.0, 10.0]),
    )
    V = DimArray(randn(rng, ComplexF32, 4, 3, 5, 2), axs)
    W = DimArray(rand(rng, Float32, 4, 3, 5, 2), axs)
    F = DimArray(rand(rng, 4, 3, 5, 2) .< 0.2, axs)
    W[1, 1, 1, 1] = 0
    V[2, 2, 2, 2] = NaN
    ok = @. !F & (W > 0) & isfinite(V)
    for d in (Frequency, Ti, (Frequency, Ti))
        wv, ws = FP.weighted_sums(V, W, F; dims = d)
        @test dims(wv) == DimensionalData.otherdims(V, d)
        @test eltype(wv) == ComplexF32 && eltype(ws) == Float32
        @test parent(wv) ≈ dropdims(sum(ifelse.(parent(ok), parent(W) .* parent(V), 0); dims = DimensionalData.dimnum(V, d)); dims = DimensionalData.dimnum(V, d))
        @test parent(ws) ≈ dropdims(sum(ifelse.(parent(ok), parent(W), 0); dims = DimensionalData.dimnum(V, d)); dims = DimensionalData.dimnum(V, d))
    end
    # Layers are read by name, so their storage order is free.
    Wp = permutedims(W, (Ti, BaselineID, Frequency, Polarization))
    @test FP.weighted_sums(V, Wp, F; dims = Frequency) == FP.weighted_sums(V, W, F; dims = Frequency)
    @test_throws DimensionMismatch FP.weighted_sums(V, set(W, Ti => [0.0, 20.0]), F; dims = Frequency)
    @test_throws "a layer has 3 dimensions" FP.weighted_sums(V, W[Ti = 1], F; dims = Frequency)

    # The per-cell loop is compiled for the reduced dimensions: a call allocates
    # its outputs, not a dispatch per cell.
    big = (
        Polarization(["RR", "RL", "LR", "LL"]), Frequency(1.0e9 .+ (0:31) .* 1.0e6),
        BaselineID(0:9), Ti(0.0:10.0:190.0),
    )
    Vb = DimArray(randn(rng, ComplexF32, 4, 32, 10, 20), big)
    Wb = DimArray(rand(rng, Float32, 4, 32, 10, 20), big)
    Fb = DimArray(falses(4, 32, 10, 20), big)
    for d in (Frequency, Ti)
        FP.weighted_sums(Vb, Wb, Fb; dims = d)
        @test @allocated(FP.weighted_sums(Vb, Wb, Fb; dims = d)) < sizeof(parent(Vb))
    end
end

@testset "per-AP sums by label" begin
    ps, _ = _build_fringe_ps(; nant = 4, nspw = 3, nchan = 4, ntime = 5, nscans = 2)
    geom = CAL.DataGeometry(ps)
    group = first(values(DimensionalData.groupby(ps, XRadio.ByScan())))
    exec = SerialScheduler()
    s = FP._ap_sums(group, geom; executor = exec)
    @test lookup(s.rbar, FP.StationPair) == [(geom.stations[a], geom.stations[b]) for (a, b) in s.bl_pairs]
    @test issorted(s.bl_pairs)
    @test lookup(s.rbar, FP.FeedPair) == [(1, 1), (1, 2), (2, 1), (2, 2)]
    @test lookup(s.rbar, Ti) == geom.times[s.ti]
    @test eltype(s.rbar) == ComplexF32 && eltype(s.wbar) == Float32

    # Each member's sums land on its own labels: the group total over members
    # equals the per-member totals.
    tot = sum(FP.weighted_sums(FP._member_layers(ms)...; dims = Frequency)[1] |> sum for ms in values(group))
    @test sum(s.rbar) ≈ tot

    # The order members are stored in does not change the sums.
    rev = XRadio.ProcessingSet(OrderedDict(reverse(collect(pairs(group)))), DimensionalData.metadata(group))
    @test FP._ap_sums(rev, geom; executor = exec).rbar == s.rbar

    members = OrderedDict(pairs(group))
    k = first(keys(members))
    members[k] = _swap_baselines(read(members[k]))
    swapped = XRadio.ProcessingSet(members, DimensionalData.metadata(group))
    @test_throws "stored in both orders" FP._ap_sums(swapped, geom; executor = exec)
    @test_throws "holds no Measurement Sets" FP._ap_sums(
        XRadio.ProcessingSet(OrderedDict{Symbol, XRadio.MeasurementSet}()), geom; executor = exec,
    )
end

@testset "per-group steps on a ProcessingSet" begin
    rng = MersenneTwister(5)
    nant, nspw, nchan = 4, 2, 8
    bp = 0.3 .* randn(rng, nant, 2, nspw * nchan)
    ps, _ = _build_fringe_ps(; nant, nspw, nchan, ntime = 6, nscans = 3, bandpass = bp, seed = 3)
    serial = ExecutionConfig(inner_executor = SerialScheduler())
    wide = ExecutionConfig(outer_executor = DynamicScheduler(), inner_executor = DynamicScheduler(; nchunks = 4))
    for st in (Bandpass(), Bandpass(smoother = FP.PerTrackSmoother()), AdhocPhase())
        a = fit(st, ps; gauge = PinAntenna(1), exec = serial)
        b = fit(st, ps; gauge = PinAntenna(1), exec = wide)
        @test a.steps[1].θ == b.steps[1].θ
        @test stage_info(a, only(keys(a))).nscans == 3
    end
    @test stage_info(fit(Bandpass(), ps; gauge = PinAntenna(1)), :bandpass).sources == ["SRC1"]
end

# Each scan's visibilities rotated by its own phase per baseline, as a source
# phase that changes from scan to scan.
function _offset_scans(ps, seed)
    rng = MersenneTwister(seed)
    members = OrderedDict{Symbol, XRadio.MeasurementSet}()
    for group in values(DimensionalData.groupby(ps, XRadio.ByScan()))
        φ = 2π .* rand(rng, length(XRadio.baselines(first(values(group)))))
        for (k, ms) in pairs(group)
            m = read(ms)
            V = copy(m[:visibility])
            for bi in axes(V, BaselineID)
                view(V, BaselineID(bi)) .*= cis(Float32(φ[bi]))
            end
            members[k] = Gustavo._with_layers(m; visibility = V)
        end
    end
    return XRadio.ProcessingSet(members, DimensionalData.metadata(ps))
end

@testset "PerTrackSmoother aligns each scan's phase before pooling" begin
    rng = MersenneTwister(5)
    ps, _ = _build_fringe_ps(;
        nant = 4, nspw = 2, nchan = 8, ntime = 6, nscans = 3,
        bandpass = 0.3 .* randn(rng, 4, 2, 16), seed = 3,
    )
    st = Bandpass(smoother = FP.PerTrackSmoother())
    θ = fit(st, ps; gauge = PinAntenna(1)).steps[1].θ
    θoff = fit(st, _offset_scans(ps, 4); gauge = PinAntenna(1)).steps[1].θ
    @test θoff ≈ θ atol = 1.0e-6

    # Pooling follows the data's type unless the smoother names one.
    θ64 = fit(Bandpass(smoother = FP.PerTrackSmoother(eltype = Float64)), ps; gauge = PinAntenna(1)).steps[1].θ
    @test θ64 ≈ θ atol = 1.0e-5
    @test_throws "must be a real floating-point type" FP.PerTrackSmoother(eltype = ComplexF64)
end

@testset "bandpass sums by label" begin
    rng = MersenneTwister(5)
    ps, _ = _build_fringe_ps(;
        nant = 4, nspw = 2, nchan = 8, ntime = 6, nscans = 3,
        bandpass = 0.3 .* randn(rng, 4, 2, 16), seed = 3,
    )
    geom = CAL.DataGeometry(ps)
    groups = collect(values(DimensionalData.groupby(ps, XRadio.ByScan())))
    members = [ms for g in groups for ms in values(g)]
    cross = [p for ms in members for p in FP._member_station_pairs(ms) if p[1] != p[2]]
    stations, _ = FP._station_pairs(cross, geom)
    feeds = sort!(unique!([f for ms in members for f in feed_pairs(ms)]))
    exec = SerialScheduler()

    rl, wl = FP.accumulate_bandpass(first(groups), geom, stations, feeds; executor = exec)
    @test lookup(rl, FP.StationPair) == stations
    @test lookup(rl, FP.FeedPair) == feeds
    @test lookup(rl, Frequency) == geom.channel_freqs
    @test eltype(rl) == ComplexF32 && eltype(wl) == Float32
    tot = sum(first(groups)) do ms
        wv, _ = FP.weighted_sums(FP._member_layers(ms)...; dims = Ti)
        autos = [a == b for (a, b) in FP._member_station_pairs(ms)]
        sum(wv[BaselineID = .!autos])
    end
    @test sum(rl) ≈ tot
    results = [(; rl, wl)]
    @test eltype(FP._pool_scans(results, [1], Float32)[1]) == ComplexF32
    @test eltype(FP._pool_scans(results, [1], Float64)[2]) == Float64

    # A scan holding only the parallel hands lines up with the full set's labels:
    # its cross-hand rows stay empty and the rest are unchanged.
    parallel = XRadio.ProcessingSet(
        OrderedDict(k => read(ms)[Polarization = At(["RR", "LL"])] for (k, ms) in pairs(first(groups))),
        DimensionalData.metadata(first(groups)),
    )
    rp, wp = FP.accumulate_bandpass(parallel, geom, stations, feeds; executor = exec)
    @test dims(rp) == dims(rl)
    @test all(iszero, wp[FeedPair = At([(1, 2), (2, 1)])])
    @test rp[FeedPair = At([(1, 1), (2, 2)])] == rl[FeedPair = At([(1, 1), (2, 2)])]

    @test_throws "stored in both orders" FP._station_pairs([("A1", "A2"), ("A2", "A1")], geom)
    @test_throws "not among the geometry's stations" FP._station_pairs([("A1", "Z9")], geom)
end
