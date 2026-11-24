# ── The fringe closure split by spectral window ──────────────────────────────
#
# Each window of a scan searched on its own, then closed together: a delay and
# phase per window beside a band-wide rate, against a source phase per baseline
# and feed pair shared by the windows.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

using StableRNGs: StableRNG
using Statistics: mean
using Logging

_scan_window(ms) = map(l -> (; l..., window = XRadio.spectralwindow(ms)), XRadio.ByScan()(ms))
_window_dets(ps) = reduce(vcat, values(mapsets(g -> fit(FringeSearch(), g), groupby(ps, _scan_window))))

function _window_model(; phase_frequency = PerSpectralWindow(), rel_time = nothing)
    rel = isnothing(rel_time) ? (;) :
        (; rel_delay = GainComponent(Delay(); Ti = rel_time, Frequency = PerSpectralWindow(), Feed = ExceptFeed(1)))
    return GainModel(;
        phase = (;
            delay = GainComponent(Delay(); Ti = PerScan(), Frequency = PerSpectralWindow(), Feed = rel_time === nothing ? PerFeed() : SharedFeeds()),
            rel...,
            rate = GainComponent(Rate(); Ti = PerScan(), Feed = SharedFeeds()),
            phase = GainComponent(ConstantTerm(); Ti = PerScan(), Frequency = phase_frequency, Feed = PerFeed()),
        ),
    )
end

# Per-window station delays and phases, the phases across the whole circle and
# changing from scan to scan, and a source phase per (scan, baseline, product)
# that every window shares. With `fixed_xy`, each station's second feed keeps
# one phase offset per window from its first in every scan.
function _window_ps(; nant = 5, nspw = 4, nchan = 16, nscans = 3, noise = 0.2, seed = 7, fixed_xy = false)
    rng = StableRNG(seed)
    chan_bw, spw_sep, ref_freq = 2.0e6, 1.0e8, 230.0e9
    nfeed = 2
    offset = 2π .* rand(rng, nant, nfeed, nspw, nscans) .- π
    fixed_xy && (offset[:, 2, :, :] .= offset[:, 1, :, :] .+ (2π .* rand(rng, nant, nspw) .- π))
    delay = 6.0e-9 .* (rand(rng, nant, nfeed, nspw) .- 0.5)
    bp = zeros(nant, nfeed, nspw * nchan, nscans)
    for a in 1:nant, f in 1:nfeed, b in 1:nspw, c in 1:nchan, s in 1:nscans
        ν = (c - 1 - (nchan - 1) / 2) * chan_bw
        bp[a, f, (b - 1) * nchan + c, s] = offset[a, f, b, s] + 2π * delay[a, f, b] * ν
    end
    ps, truth = _build_fringe_ps(; nant, nspw, nchan, nscans, noise, chan_bw, spw_sep, ref_freq, bandpass = bp, eltype = ComplexF64, seed)
    source = Dict{Tuple{String, Int}, Matrix{Float64}}()
    for ms in values(ps)
        scan = only(unique(collect(ms[:scan_name])))
        ψ = get!(() -> 2π .* rand(rng, size(ms[:visibility], 1), size(ms[:visibility], 3)) .- π, source, (scan, 0))
        vis = copy(parent(ms[:visibility]))
        for p in axes(vis, 1), bl in axes(vis, 3)
            view(vis, p, :, bl, :) .*= cis(ψ[p, bl])
        end
        ms[:visibility] = rebuild(ms[:visibility], vis)
    end
    return ps, truth
end

# The phase of each (scan, baseline, product, window) after `sol`, from the
# window's mean phasor, and the mean over its times of the coherence across
# channels, `|Σv| / Σ|v|`.
function _window_phasors(sol, ps)
    out = Dict{Tuple{String, Int, Int}, Vector{Tuple{Float64, Float64}}}()
    cal = calibrate(sol, ps; flag_bad = false, apply_flags = false)
    for ms in values(cal)
        scan = only(unique(collect(ms[:scan_name])))
        vis = parent(ms[:visibility])
        a1, a2 = collect(ms[:baseline_antenna1_name]), collect(ms[:baseline_antenna2_name])
        for p in axes(vis, 1), bl in axes(vis, 3)
            a1[bl] == a2[bl] && continue
            v = view(vis, p, :, bl, :)
            coherence = mean(abs(sum(c)) / sum(abs, c) for c in eachcol(v))
            push!(get!(out, (scan, p, bl), Tuple{Float64, Float64}[]), (angle(sum(v)), coherence))
        end
    end
    return out
end

# The largest departure of a window's phase from its baseline and product's
# mean over the windows.
_window_spread(phasors) = maximum(values(phasors)) do ws
    φ = first.(ws)
    maximum(abs, rem2pi.(φ .- angle(sum(cis, φ)), RoundNearest))
end

@testset "fringe closure: delay and phase per spectral window" begin
    ps, _ = _window_ps()
    dets = _window_dets(ps)
    @test length(dets) == 3 * 4
    sol = @test_logs min_level = Logging.Warn fit(FringeClosure(; model = _window_model(), gauge = PinAntenna(1)), dets)
    @test sol.info.nscan == 3
    @test isempty(sol.info.flagged_ant)

    # Every window of a baseline and product ends at one phase, the source's,
    # with the window's delay taken out: each window's phasor is coherent.
    phasors = _window_phasors(sol, ps)
    @test _window_spread(phasors) < 0.02
    @test minimum(minimum(last, ws) for ws in values(phasors)) > 0.99

    # Each station's first feed has phases summing to zero over a scan's windows.
    ph = only(c for c in sol.components if last(c.path) === :phase).params
    x = parent(ph[Feed = At(1)])
    @test maximum(abs, sum(x; dims = dimnum(ph[Feed = At(1)], Frequency))) < 1.0e-8

    # An inter-feed delay held over the track couples the scans' delays; their
    # phases are still solved scan by scan.
    coupled = fit(FringeClosure(; model = _window_model(; rel_time = GlobalTime()), gauge = PinAntenna(1)), dets)
    @test _window_spread(_window_phasors(coupled, ps)) < 0.02
end

# Each station's gain phase of its second feed against its first at each
# window's center, from the window's summed channel gains, at an integration in
# the middle of each scan: an array over (window, scan, station).
function _window_xy(sol)
    g = sol.geom
    channels = [findall(==(w), g.spw_of_chan) for w in eachindex(g.spw_names)]
    middles = [(i = findall(==(s), g.scan_of_time); i[cld(length(i), 2)]) for s in eachindex(g.scan_names)]
    xy = map(Iterators.product(channels, middles, g.antennas)) do (cs, ti, a)
        gt = gains(sol; Ti = ti)
        center(f) = sum(c -> gt[Frequency = c, Feed = At(f), AntennaName = At(a)], cs)
        angle(center(2) / center(1))
    end
    return xy
end

@testset "fringe closure: a station's inter-feed phase is the same in every scan" begin
    ps, _ = _window_ps(; nscans = 6, fixed_xy = true)
    sol = fit(FringeClosure(; model = _window_model(), gauge = PinAntenna(1)), _window_dets(ps))
    xy = _window_xy(sol)
    @test maximum(abs, rem2pi.(xy .- xy[:, 1:1, :], RoundNearest)) < 0.1
    # Its circular mean over the windows is zero in every scan.
    @test maximum(abs, angle.(sum(cis.(xy); dims = 1))) < 1.0e-6
end

@testset "fringe closure: a phase across the band is rejected" begin
    ps, _ = _window_ps(; nscans = 1)
    @test_throws "source phase absorbs it" fit(
        FringeClosure(; model = _window_model(; phase_frequency = GlobalFrequency()), gauge = PinAntenna(1)), _window_dets(ps),
    )
end

@testset "fringe closure: a detection spanning several windows is rejected" begin
    ps, _ = _window_ps(; nscans = 1)
    @test_throws "span 4 frequency segments" fit(
        FringeClosure(; model = _window_model(), gauge = PinAntenna(1)), fit(FringeSearch(), ps),
    )
end

@testset "fringe closure: a station's window without fringes is flagged there alone" begin
    ps, _ = _window_ps(; nscans = 1)
    for ms in values(ps)
        XRadio.spectralwindow(ms) == "band_2" || continue
        fl = parent(ms[:flag])
        a1, a2 = collect(ms[:baseline_antenna1_name]), collect(ms[:baseline_antenna2_name])
        for bl in eachindex(a1)
            "A3" in (a1[bl], a2[bl]) && (view(fl, :, :, bl, :) .= true)
        end
    end
    sol = fit(FringeClosure(; model = _window_model(), gauge = PinAntenna(1)), _window_dets(ps))
    flagged = unique(zip(sol.info.flagged_ant, sol.info.flagged_window))
    @test sort(flagged) == [("A3", "band_2")]
end

@testset "fringe closure: a source phase only weak rows measure is theirs, not the gauge's" begin
    # Rows over nodes 1 and 2, the first accepted, each with its own source
    # phase (3, 4). Were the second row free to drop, node 4 would read as a
    # freedom for the gauge, and a pin meant for it could land on a station.
    rows = (; A = [[1], [1]], B = [[2], [2]], extra = [3, 4], accept = [true, false])
    @test Gustavo.Fring._held_rows(rows) == [true, true]
    @test Gustavo.Fring._held_rows(merge(rows, (; extra = [3, 3]))) == [true, false]
end
