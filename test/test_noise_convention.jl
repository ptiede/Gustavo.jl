# The noise convention the bandpass solve relies on: with WEIGHT = 1/σ² per real
# component, each real component of the noise in a coherent sum r = Σ w·V has
# variance W = Σ w, so s² = |r|²/W is the inverse variance of angle(r) and
# log|r|, and s² ~ χ²₂ under noise alone.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

const _NC_FR = Gustavo.Fring
const _NC_A0 = 2.5
const _NC_NCHAN = 256

# One scan's `accumulate_bandpass` sums of a 4-station fixture with no station
# gains, so every cell's signal is `_NC_A0` times the fixture's bandpass.
function _nc_sums(; seed, noise, kw...)
    ps, _ = _build_fringe_ps(;
        nant = 4, nspw = 1, nchan = _NC_NCHAN, ntime = 8, noise, seed,
        station_gains = false, eltype = ComplexF64, kw...,
    )
    geom = Gustavo.Calibration.DataGeometry(ps)
    members = collect(values(ps))
    pairs, feeds = _NC_FR._cross_cell_labels(
        map(_NC_FR._member_station_pairs, members), map(feed_pairs, members), geom,
    )
    return _NC_FR.accumulate_bandpass(ps, geom, pairs, feeds)
end

# Every (station pair, feed pair, two-channel segment)'s `_segment_residual`.
function _nc_segments(rl, wl)
    out = Tuple{ComplexF64, Float64}[]
    nchan = size(rl, Frequency)
    for bi in axes(rl, AntennaPair), p in axes(rl, FeedPair), c in 1:2:nchan
        push!(out, _NC_FR._segment_residual(rl, wl, bi, p, c:(c + 1)))
    end
    return out
end

@testset "bandpass noise convention" begin
    # Monte Carlo tolerances are 4 standard errors of the estimate.
    @testset "noise only: s² ~ χ²₂" begin
        silent = fill(-Inf, 4, 2, _NC_NCHAN)
        segs = reduce(vcat, (_nc_segments(_nc_sums(; seed, noise = 2.0, amp_bandpass = silent)...) for seed in 1:2))
        s2 = [abs2(r) / w for (r, w) in segs]
        n = length(s2)
        @test n == 6144
        @test mean(s2) ≈ 2 atol = 4 * 2 / sqrt(n)
        p = exp(-1 / 2)
        @test mean(>=(1), s2) ≈ p atol = 4 * sqrt(p * (1 - p) / n)
    end

    @testset "signal: s² is the inverse variance of angle(r) and log|r|" begin
        segs = reduce(vcat, (_nc_segments(_nc_sums(; seed, noise = 2.0)...) for seed in 11:12))
        n = length(segs)
        W = only(unique(last.(segs)))
        ρ² = _NC_A0^2 * W
        @test 40 < ρ² < 60
        s2 = [abs2(r) / w for (r, w) in segs]
        @test mean(s2) ≈ ρ² + 2 atol = 4 * sqrt(4ρ² + 4) / sqrt(n)
        # First order in 1/ρ²: var(angle r) = var(log|r|) = 1/ρ² = 1/E[s² − 2].
        prec = 1 / (mean(s2) - 2)
        @test var(angle.(first.(segs))) ≈ prec rtol = 4 * sqrt(2 / n) + 2 / ρ²
        @test var(log.(abs.(first.(segs)))) ≈ prec rtol = 4 * sqrt(2 / n) + 2 / ρ²
    end

    @testset "joint gain estimate ĝ = num/den" begin
        # Station A1 carries gain g on both feeds; every other gain is 1 and the
        # source coherence is S = _NC_A0, so each A1 cell has coeff = S.
        g = exp(complex(0.2, 0.6))
        bp = zeros(4, 2, _NC_NCHAN)
        bp[1, :, :] .= angle(g)
        la = zeros(4, 2, _NC_NCHAN)
        la[1, :, :] .= log(abs(g))
        ĝs = ComplexF64[]
        dens = Float64[]
        for seed in 21:24
            rl, wl = _nc_sums(; seed, noise = 15.0, bandpass = bp, amp_bandpass = la)
            spairs = lookup(rl, AntennaPair)
            fps = lookup(rl, FeedPair)
            for feed in 1:2, c in axes(rl, Frequency)
                num, den = zero(ComplexF64), 0.0
                for bi in eachindex(spairs), p in eachindex(fps)
                    first_end = spairs[bi][1] == "A1" && fps[p][1] == feed
                    second_end = spairs[bi][2] == "A1" && fps[p][2] == feed
                    (first_end || second_end) || continue
                    coeff = complex(_NC_A0)
                    cell = (AntennaPair(bi), FeedPair(p), Frequency(c))
                    rc, wc = rl[cell...], wl[cell...]
                    num += conj(coeff) * (first_end ? rc : conj(rc))
                    den += wc * abs2(coeff)
                end
                push!(ĝs, num / den)
                push!(dens, den)
            end
        end
        n = length(ĝs)
        den = only(unique(dens))
        @test 3 < den * abs2(g) < 5
        δ = ĝs .- g
        @test mean(abs2, real.(δ)) ≈ 1 / den rtol = 4 * sqrt(2 / n)
        @test mean(abs2, imag.(δ)) ≈ 1 / den rtol = 4 * sqrt(2 / n)
        q = den .* abs2.(ĝs)
        @test mean(q) ≈ den * abs2(g) + 2 atol = 4 * sqrt((4den * abs2(g) + 4) / n)
    end
end
