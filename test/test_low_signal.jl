# Low-signal flagging: `signal_information` of data corrected by a
# per-integration gain solution and `flag_low_signal!`.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

@testset "signal_information and flag_low_signal!" begin
    # Feed 2 of A2 keeps 3% of its signal over the second half of the scan.
    ps, _ = _build_fringe_ps(; nant = 4, nspw = 1, nchan = 64, ntime = 24, noise = 0.1, antenna_gains = false)
    late(t) = t >= XRadio.times(first(values(ps)))[13]
    dark(a, b, (fa, fb)) = (a == "A2" && fa == 2) || (b == "A2" && fb == 2)
    for ms in values(ps)
        V = parent(ms[:visibility])
        ts, pairs_, feeds = collect(XRadio.times(ms)), Gustavo.Fring._member_antenna_pairs(ms), feed_pairs(ms)
        tdim, bdim, pdim = map(D -> DimensionalData.dimnum(ms[:visibility], D), (Ti, BaselineID, Polarization))
        for (bl, (a, b)) in pairs(pairs_), p in axes(feeds, 1), (i, t) in pairs(ts)
            (dark(a, b, feeds[p, bl]) && late(t)) || continue
            idx = ntuple(d -> d == tdim ? i : d == bdim ? bl : d == pdim ? p : Colon(), ndims(V))
            V[idx...] .*= 0.03
        end
    end
    calibrate!(fit(Adhoc(; model = default_adhoc_model(; feed = PerFeed()), gauge = PinAntenna(1)), ps), ps)
    info = signal_information(ps)
    @test dims(info) isa Tuple{Gustavo.Scan, AntennaName, Feed, Ti}
    times = lookup(info, Ti)
    @test all(<(0.01), info[AntennaName = At("A2"), Feed = At(2), Ti = At(filter(late, times))])
    @test all(>(0.5), info[AntennaName = At("A2"), Feed = At(2), Ti = At(filter(!late, times))])
    @test all(>(0.5), info[AntennaName = At("A2"), Feed = At(1)])
    @test all(>(0.5), info[AntennaName = At(["A1", "A3", "A4"])])

    # The per-sample loops run on concrete arrays: a loop on layers whose types do
    # not infer allocates many times the layers' size.
    layers = sum(_layer_bytes, values(ps))
    allocations(f) = (f(); @allocated f())
    @test allocations(() -> signal_information(ps)) < layers

    @test allocations(() -> flag_low_signal!(ps, info)) < layers
    for ms in values(ps)
        Fl = Gustavo.Fring._coherence_cube(ms[:flag])
        ts = collect(XRadio.times(ms))
        feeds = feed_pairs(ms)
        for (bl, (a, b)) in pairs(Gustavo.Fring._member_antenna_pairs(ms)), p in axes(feeds, 1), (i, t) in pairs(ts)
            @test all(Fl[:, i, bl, p] .== (dark(a, b, feeds[p, bl]) && late(t)))
        end
    end
end
