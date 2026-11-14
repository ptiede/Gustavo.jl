# Low-signal flagging: `signal_information` from a per-integration gain
# solution and `flag_low_signal!`.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

@testset "signal_information and flag_low_signal!" begin
    # A2 keeps 3% of its signal over the second half of the scan.
    ps, _ = _build_fringe_ps(; nant = 4, nspw = 1, nchan = 64, ntime = 24, noise = 0.1, antenna_gains = false)
    late(t) = t >= XRadio.times(first(values(ps)))[13]
    for ms in values(ps)
        V = parent(ms[:visibility])
        ts, pairs_ = collect(XRadio.times(ms)), Gustavo.Fring._member_antenna_pairs(ms)
        tdim, bdim = DimensionalData.dimnum(ms[:visibility], Ti), DimensionalData.dimnum(ms[:visibility], BaselineID)
        for (bl, p) in pairs(pairs_), (i, t) in pairs(ts)
            ("A2" in p && late(t)) || continue
            idx = ntuple(d -> d == tdim ? i : d == bdim ? bl : Colon(), ndims(V))
            V[idx...] .*= 0.03
        end
    end
    sol = fit(Adhoc(; gauge = PinAntenna(1)), ps)
    info = signal_information(sol, ps)
    @test dims(info) isa Tuple{Gustavo.Scan, AntennaName, Ti}
    times = lookup(info, Ti)
    lost = info[AntennaName = At("A2"), Ti = At(filter(late, times))]
    @test all(<(0.01), lost)
    @test all(>(0.5), info[AntennaName = At("A2"), Ti = At(filter(!late, times))])
    @test all(>(0.5), info[AntennaName = At(["A1", "A3", "A4"])])

    # The per-sample loops run on concrete arrays: a loop on layers whose types do
    # not infer allocates many times the layers' size.
    layers = sum(_layer_bytes, values(ps))
    allocations(f) = (f(); @allocated f())
    @test allocations(() -> signal_information(sol, ps)) < 4 * layers

    @test allocations(() -> flag_low_signal!(ps, info)) < layers
    for ms in values(ps)
        Fl = Gustavo.Fring._coherence_cube(ms[:flag])
        ts = collect(XRadio.times(ms))
        for (bl, p) in pairs(Gustavo.Fring._member_antenna_pairs(ms)), (i, t) in pairs(ts)
            @test all(Fl[:, i, bl, :] .== ("A2" in p && late(t)))
        end
    end
end
