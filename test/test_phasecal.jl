# PhaseCal: station phases from the injected phase-calibration tones.
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test
import XRadio
using DimensionalData: branches, lookup, rebuild
@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

const PC_A0 = 2.5

# A track with no station gains, so every visibility is `PC_A0`, in linear feeds.
_pc_track(; kw...) = first(
    _build_fringe_ps(;
        nant = 4, nspw = 2, nchan = 16, ntime = 8, nscans = 2, ref_freq = 6.6e9, chan_bw = 2.0e6,
        spw_sep = 1.0e8, polarizations = ["XX", "XY", "YX", "YY"], receptors = ("X", "Y"),
        antenna_gains = false, eltype = ComplexF64, kw...,
    )
)

_pc_scan(ms) = parse(Int, only(XRadio.scans(ms)))
_pc_center(ms) = (ν = XRadio.frequencies(ms); sum(ν) / length(ν))

# Each visibility times e^{-i(ψ_a - ψ_b)}: an instrumental phase ψ(station, feed,
# ν, t) entering the conjugated visibilities MSv4 stores.
function _pc_corrupt!(ms, ψ)
    vis = copy(parent(ms[:visibility]))
    feeds = Gustavo.feed_pairs(ms)
    ν, t = XRadio.frequencies(ms), XRadio.times(ms)
    a1, a2 = collect(ms[:baseline_antenna1_name]), collect(ms[:baseline_antenna2_name])
    for p in axes(vis, 1), c in axes(vis, 2), b in axes(vis, 3), ti in axes(vis, 4)
        fa, fb = feeds[p, b]
        vis[p, c, b, ti] *= cis(-(ψ(a1[b], fa, ν[c], t[ti], ms) - ψ(a2[b], fb, ν[c], t[ti], ms)))
    end
    ms[:visibility] = rebuild(ms[:visibility], vis)
    return ms
end

# A `phase_calibration` sub-dataset whose tones at `tone_offsets` (Hz from the
# window's center) measure ψ at epochs `tpc`, each `interval` seconds long.
# `untoned(name)` leaves a station's tones unmeasured.
function _pc_tones!(ms, ψ, tone_offsets, tpc; interval = 30.0, untoned = Returns(false))
    ant = branches(ms)[:antenna]
    order = (XRadio.ReceptorLabel, XRadio.AntennaName)
    names = collect(String, lookup(ant[:polarization_type], XRadio.AntennaName))
    νk = _pc_center(ms) .+ tone_offsets
    nk, nt, na = length(νk), length(tpc), length(names)
    tones = fill(complex(NaN, NaN), nk, 2, nt, na)
    freqs = fill(NaN, nk, 2, nt, na)
    for (ia, a) in pairs(names), it in 1:nt, f in 1:2, k in 1:nk
        untoned(a) && continue
        tones[k, f, it, ia] = 0.05 * cis(ψ(a, f, νk[k], tpc[it], ms))
        freqs[k, f, it, ia] = νk[k]
    end
    ms.phase_calibration = XRadio.subdataset(
        :phase_calibration;
        antenna_name = names, station_name = copy(names), receptor_label = ["pol_0", "pol_1"],
        polarization_type = Array(parent(permutedims(ant[:polarization_type], order))),
        mount = collect(String, parent(ant[:mount])), telescope_name = collect(String, parent(ant[:telescope_name])),
        time_phase_cal = collect(Float64, tpc), tone_label = ["tone_$(k - 1)" for k in 1:nk],
        phase_cal = tones, phase_cal_tone_frequency = freqs,
        phase_cal_cable_cal = zeros(nt, na), phase_cal_interval = fill(interval, nt, na),
        array_metadata = (; time_phase_cal = (; units = "s", scale = "utc", format = "unix", type = "time")),
    )
    return ms
end

_pc_flat(ps) = maximum(maximum(abs, parent(ms[:visibility]) .- PC_A0) for ms in values(ps))
_pc_delays(sol) = parent(only(sol[:phase, :phasecal_delay].components).params)

@testset "PhaseCal" begin
    station(a) = parse(Int, a[2:end])
    window(ms) = parse(Int, last(split(XRadio.spectralwindow(ms), "_")))
    five_mhz = (-12.5:5:12.5) .* 1.0e6

    @testset "per scan: a phase and a delay per window, feed and scan" begin
        ψ(a, f, ν, t, ms) = 0.7 * station(a) - 0.4f + 0.3 * window(ms) * _pc_scan(ms) +
            2π * (3.0e-9 * station(a) - 1.5e-9 * f + 2.0e-9 * window(ms)) * (ν - _pc_center(ms))
        ps = _pc_track()
        for ms in values(ps)
            _pc_corrupt!(ms, ψ)
            _pc_tones!(ms, ψ, five_mhz, XRadio.times(ms))
        end
        @test _pc_flat(ps) > 1
        sol = fit(PhaseCal(), ps)
        @test sol.name == :phasecal
        @test _pc_flat(_precal(sol, ps)) < 1.0e-6
        @test isempty(sol.info.untoned_station)
    end

    @testset "per integration: tones interpolated between measurements" begin
        ψ(a, f, ν, t, ms) = 0.5 * station(a) + f + 2.0e-3 * station(a) * (t - 1.6e9) +
            2π * 4.0e-9 * station(a) * (ν - _pc_center(ms))
        ps = _pc_track()
        for ms in values(ps)
            _pc_corrupt!(ms, ψ)
            t = XRadio.times(ms)
            _pc_tones!(ms, ψ, five_mhz, t[[1, 3, 5, 8]])
        end
        @test _pc_flat(_precal(fit(PhaseCal(), ps), ps)) > 1.0e-2
        @test _pc_flat(_precal(fit(PhaseCal(; time = PerIntegration()), ps), ps)) < 1.0e-6
    end

    @testset "one delay for the track, a phase per scan or integration" begin
        ψ(a, f, ν, t, ms) = 0.4 * station(a) * _pc_scan(ms) + 1.0e-3 * f * (t - 1.6e9) +
            2π * (2.0e-9 * station(a) - 1.0e-9 * f + 1.5e-9 * window(ms)) * (ν - _pc_center(ms))
        ps = _pc_track()
        for ms in values(ps)
            _pc_corrupt!(ms, ψ)
            _pc_tones!(ms, ψ, five_mhz, XRadio.times(ms)[[1, 3, 5, 8]])
        end
        sol = fit(PhaseCal(; time = PerIntegration(), delay_time = GlobalTime()), ps)
        @test size(_pc_delays(sol), 4) == 1
        @test _pc_flat(_precal(sol, ps)) < 1.0e-6
        @test _pc_flat(_precal(fit(PhaseCal(; delay_time = GlobalTime()), ps), ps)) > 1.0e-2
        @test_throws "delay_time must be GlobalTime() or the same as time" PhaseCal(; time = PerScan(), delay_time = PerIntegration())
    end

    @testset "a delay stays on one branch of the tone-spacing ambiguity" begin
        # Three tones 10 MHz apart fix a delay modulo 100 ns. Scan 1's 49 ns and
        # scan 2's 51 ns straddle the ±50 ns branch edge, which a search centered
        # on zero would put 98 ns apart; either branch fits the tones.
        τ(ms) = _pc_scan(ms) == 1 ? 49.0e-9 : 51.0e-9
        ψ(a, f, ν, t, ms) = station(a) == 1 ? 0.0 : 2π * τ(ms) * (ν - _pc_center(ms))
        ps = _pc_track()
        for ms in values(ps)
            _pc_corrupt!(ms, ψ)
            _pc_tones!(ms, ψ, (-10:10:10) .* 1.0e6, XRadio.times(ms))
        end
        d = _pc_delays(fit(PhaseCal(), ps))
        @test d[1, :, :, 2, 2:end] .- d[1, :, :, 1, 2:end] ≈ fill(-2.0e-9, 2, 2, 3)
    end

    @testset "lines are referred to the window's center, not the tones'" begin
        ψ(a, f, ν, t, ms) = 0.6 * station(a) - 0.3f + 2π * (2.5e-9 * station(a) + 1.0e-9 * f) * (ν - _pc_center(ms))
        ps = _pc_track()
        for ms in values(ps)
            _pc_corrupt!(ms, ψ)
            _pc_tones!(ms, ψ, (0:5:15) .* 1.0e6, XRadio.times(ms); untoned = ==("A4"))
        end
        sol = @test_logs (:warn, r"no tone measurement") match_mode = :any fit(PhaseCal(), ps)
        @test all(isfinite, parent(only(sol[:phase, :phasecal].components).params))
        for ms in values(_precal(sol, ps))
            a1, a2 = collect(ms[:baseline_antenna1_name]), collect(ms[:baseline_antenna2_name])
            toned = [b for b in eachindex(a1) if "A4" ∉ (a1[b], a2[b])]
            @test maximum(abs, parent(ms[:visibility])[:, :, toned, :] .- PC_A0) < 1.0e-6
        end
    end

    @testset "degenerate tone sets give finite lines" begin
        νc = 6.6e9
        ν = νc .+ (-10:10:10) .* 1.0e6
        z = cis.(2π * 3.0e-9 .* (ν .- νc))
        @test all(isfinite, Gustavo._slope_moments([zero(eltype(z)); z], [νc - 2.0e7; ν], νc, 3.0e-9))
        tr = (; νc, period = 1.0e-7, ngrid = 512, τref = 0.0, τfix = nothing)
        @test all(isfinite, Gustavo._tone_line(z[1:1], ν[1:1], tr))
        @test PhaseCal{PerScan, PerScan}(PerScan(), PerScan()) == PhaseCal()
    end

    @testset "a station without tones keeps unit gain and is reported" begin
        ψ(a, f, ν, t, ms) = 0.3 * station(a) + 2π * 1.0e-9 * station(a) * (ν - _pc_center(ms))
        ps = _pc_track()
        for ms in values(ps)
            _pc_corrupt!(ms, ψ)
            _pc_tones!(ms, ψ, five_mhz, XRadio.times(ms); untoned = a -> a == "A3" && _pc_scan(ms) == 2)
        end
        sol = @test_logs (:warn, r"PhaseCal: 4 .* cells have no tone measurement") match_mode = :any fit(PhaseCal(), ps)
        @test Set(zip(sol.info.untoned_station, sol.info.untoned_scan)) == Set([("A3", "2")])
        out = _precal(sol, ps)
        for (k, ms) in pairs(out)
            a1, a2 = collect(ms[:baseline_antenna1_name]), collect(ms[:baseline_antenna2_name])
            touched = [b for b in eachindex(a1) if "A3" in (a1[b], a2[b]) && a1[b] != a2[b]]
            others = setdiff(eachindex(a1), touched)
            vis = parent(ms[:visibility])
            @test maximum(abs, vis[:, :, others, :] .- PC_A0) < 1.0e-6
            if _pc_scan(ms) == 2
                @test maximum(abs, vis[:, :, touched, :] .- PC_A0) > 1.0e-2
            end
        end
    end
end
