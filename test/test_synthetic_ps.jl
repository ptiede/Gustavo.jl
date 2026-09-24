@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

@testset "synthetic ProcessingSet fixture" begin
    ps, truth = _build_fringe_ps(; nant = 3, nspw = 2, nchan = 4, ntime = 5, nscans = 2)
    @test isempty(XRadio.check(ps; schemas = [Gustavo.UVData.GUSTAVO_VISIBILITY_SCHEMA]))
    @test length(ps) == 4
    @test [only(unique(ms[:scan_name])) for ms in values(ps)] == ["1", "1", "2", "2"]
    @test [XRadio.spectralwindow(ms) for ms in values(ps)] == ["band_1", "band_2", "band_1", "band_2"]

    ms = last(collect(values(ps)))
    p, c, bl, ti = 2, 3, 3, 4
    a, b = truth.bl_pairs[bl]
    fa, fb = truth.feeds[p, bl]
    f = lookup(ms, XRadio.Frequency)[c]
    t = lookup(ms, XRadio.Ti)[ti]
    phase = truth.phi[a, fa] - truth.phi[b, fb] +
        2π * (truth.delay[a, fa] - truth.delay[b, fb]) * (f - truth.f0) +
        2π * (truth.rate[a, fa] - truth.rate[b, fb]) * (t - truth.t0_sec) +
        truth.screen[a, fa, ti, 2] - truth.screen[b, fb, ti, 2]
    @test parent(ms[:visibility])[p, c, bl, ti] ≈ ComplexF32(2.5 * cis(phase))
    @test all(==(1.0f3), parent(ms[:weight]))
    @test !any(parent(ms[:flag]))

    again, _ = _build_fringe_ps(; nant = 3, nspw = 2, nchan = 4, ntime = 5, nscans = 2)
    @test all(parent(x[:visibility]) == parent(y[:visibility]) for (x, y) in zip(values(ps), values(again)))

    dropped, _ = _build_fringe_ps(; nant = 4, nspw = 1, omit_station = 2)
    ms = only(values(dropped))
    @test "A2" ∉ ms[:baseline_antenna1_name] && "A2" ∉ ms[:baseline_antenna2_name]
    @test "A2" in collect(lookup(ms.antenna, XRadio.AntennaName))
end
