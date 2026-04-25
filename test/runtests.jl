using Gustavo
using Test
using LinearAlgebra
using Statistics
using Random
using StructArrays

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
    vis = reshape(vis, 2, 4, 4)
    weights = fill(1.0, size(vis))

    UV = Gustavo.UVFITS
    antennas = UV.AntennaTable(
        StructArray{UV.Antenna}((
            name        = ["AA", "AX"],
            station_xyz = [zeros(3), zeros(3)],
            mount_type  = [0, 0],
            axis_offset = [0.0, 0.0],
            diameter    = [0.0, 0.0],
            feed_a      = ["R", "R"],
            feed_b      = ["L", "L"],
            pola_angle  = [0.0, 0.0],
            polb_angle  = [0.0, 0.0],
        )),
        zeros(3), "TEST", 1.0e9, "2000-01-01",
        0.0, 360.0, 0.0, "UTC", "ITRF", "RIGHT",
    )
    metadata = UV.ObsMetadata(
        "TEST", "TEST", "test", "2000-01-01", 2000.0, "JY",
        0.0, 0.0, 1.0e9,
        collect(1.0:4.0), fill(1.0, 4), 1.0, fill(1, 4),
        [-1, -2, -3, -4], ["11", "22", "12", "21"],
    )

    return Gustavo.Bandpass.UVData(
        vis,
        weights,
        zeros(2, 3),
        [0.0, 1.0],
        [1, 2],
        [1.0, 1.0],
        [(1, 2)],
        Dict(1.0 => 1),
        [1.0],
        StructArray(lower=[0.0, 1.0], upper=[1.0, 2.0]),
        antennas,
        metadata,
        [],
    )
end

function synthetic_bandpass_avg_uvdata()
    BP = Gustavo.Bandpass
    nant = 3
    bl_pairs = [(1, 2), (1, 3), (2, 3)]
    ant_names = ["AA", "AX", "NN"]
    pol_codes = [-1, -2, -3, -4]
    pol_labels = ["11", "22", "12", "21"]
    nscan = 2
    nchan = 6

    gains_true = Array{ComplexF64}(undef, nant, 2, nchan)
    for c in 1:nchan
        gains_true[1, 1, c] = (0.95 + 0.02c) * cis(0.03 * (c - 1))
        gains_true[2, 1, c] = (1.05 + 0.01c) * cis(0.14 + 0.04 * (c - 1))
        gains_true[3, 1, c] = (0.88 - 0.015c) * cis(-0.09 - 0.05 * (c - 1))

        gains_true[1, 2, c] = gains_true[1, 1, c] * (1.10 - 0.01c) * cis(0.20 + 0.02 * (c - 1))
        gains_true[2, 2, c] = gains_true[2, 1, c] * (0.92 + 0.015c) * cis(-0.15 + 0.01 * (c - 1))
        gains_true[3, 2, c] = gains_true[3, 1, c] * (1.04 - 0.005c) * cis(0.11 - 0.03 * (c - 1))
    end

    source_true = reshape(
        ComplexF64[
            1.5 * cis(0.2), 0.3 * cis(-0.4), 0.25 * cis(0.1), 0.9 * cis(0.3),
            0.7 * cis(-0.1), 0.2 * cis(0.5), 0.15 * cis(-0.2), 1.2 * cis(-0.3),

            1.1 * cis(0.4), 0.35 * cis(0.2), 0.18 * cis(-0.5), 0.8 * cis(0.15),
            0.9 * cis(-0.2), 0.25 * cis(0.35), 0.12 * cis(-0.1), 1.0 * cis(-0.25),

            0.8 * cis(0.1), 0.28 * cis(-0.15), 0.21 * cis(0.22), 1.1 * cis(0.18),
            1.3 * cis(-0.05), 0.18 * cis(0.4), 0.14 * cis(-0.3), 0.95 * cis(0.27),
        ],
        nscan, length(bl_pairs), 2, 2
    )

    vis = zeros(ComplexF64, nscan, length(bl_pairs), length(pol_codes), nchan)
    for s in 1:nscan, (bi, (a, b)) in enumerate(bl_pairs), pol in eachindex(pol_codes), c in 1:nchan
        fa, fb = BP.stokes_feed_pair(pol_codes[pol])
        vis[s, bi, pol, c] = gains_true[a, fa, c] * source_true[s, bi, fa, fb] * conj(gains_true[b, fb, c])
    end
    weights = ones(Float64, size(vis))

    UV = Gustavo.UVFITS
    antennas = UV.AntennaTable(
        StructArray{UV.Antenna}((
            name        = ant_names,
            station_xyz = [zeros(3) for _ in 1:nant],
            mount_type  = zeros(Int, nant),
            axis_offset = zeros(nant),
            diameter    = zeros(nant),
            feed_a      = fill("R", nant),
            feed_b      = fill("L", nant),
            pola_angle  = zeros(nant),
            polb_angle  = zeros(nant),
        )),
        zeros(3), "TEST", 1.0e9, "2000-01-01",
        0.0, 360.0, 0.0, "UTC", "ITRF", "RIGHT",
    )
    metadata = UV.ObsMetadata(
        "TEST", "TEST", "test", "2000-01-01", 2000.0, "JY",
        0.0, 0.0, 1.0e9,
        collect(1.0:nchan), fill(1.0, nchan), 1.0, fill(1, nchan),
        pol_codes, pol_labels,
    )

    return BP.UVData(
        vis,
        weights,
        zeros(nscan, 3),
        collect(0.0:(nscan - 1)),
        collect(1:nscan),
        collect(1.0:length(bl_pairs)),
        bl_pairs,
        Dict(Float64(i) => i for i in 1:length(bl_pairs)),
        collect(1.0:length(bl_pairs)),
        StructArray(lower=Float64.(0:(nscan - 1)), upper=Float64.(1:nscan)),
        antennas,
        metadata,
        [],
    )
end

@testset "Gustavo.jl" begin
    @test isdefined(Gustavo, :Bandpass)
    @test Gustavo.Bandpass isa Module
    @test isdefined(Gustavo.Bandpass, :solve_bandpass)
end

@testset "Bandpass composite basis" begin
    BP = Gustavo.Bandpass
    segmentation = BP.BlockFrequencySegmentation(4)
    model = BP.CompositeBandpassModel(
        BP.SegmentedBandpassModel(BP.FlatBandpassModel(), segmentation),
        BP.SegmentedBandpassModel(BP.PolynomialBandpassModel(1), segmentation),
    )

    x = collect(1.0:8.0)
    valid = trues(length(x))
    basis = Vector{Vector{Float64}}()
    for component in BP.model_components(model, segmentation)
        append!(basis, BP.component_design_columns(component, x, valid))
    end
    A = hcat(basis...)

    @test A[1:4, 3] ≈ [-1.0, -1 / 3, 1 / 3, 1.0]
    @test A[5:8, 4] ≈ [-1.0, -1 / 3, 1 / 3, 1.0]

    valid = Bool[1, 0, 0, 0, 1, 1, 1, 1]
    basis = Vector{Vector{Float64}}()
    for component in BP.model_components(model, segmentation)
        append!(basis, BP.component_design_columns(component, x, valid))
    end
    A = hcat(basis...)

    @test size(A, 2) == 3
    @test rank(A[valid, :]) == size(A, 2)
end

@testset "Bandpass time segmentation" begin
    BP = Gustavo.Bandpass
    model = BP.StationBandpassModel(
        reference = BP.FeedBandpassModel(
            phase = BP.BandpassSpec(
                BP.PolynomialBandpassModel(1);
                segmentation = BP.BandpassSegmentation(
                    BP.PerScanTimeSegmentation(),
                    BP.GlobalFrequencySegmentation()
                )
            ),
            amplitude = BP.BandpassSpec(
                BP.PolynomialBandpassModel(1);
                segmentation = BP.BandpassSegmentation(
                    BP.GlobalTimeSegmentation(),
                    BP.GlobalFrequencySegmentation()
                )
            )
        ),
        relative = BP.FeedBandpassModel(
            phase = BP.BandpassSpec(
                BP.PolynomialBandpassModel(1);
                segmentation = BP.BandpassSegmentation(
                    BP.GlobalTimeSegmentation(),
                    BP.GlobalFrequencySegmentation()
                )
            ),
            amplitude = BP.BandpassSpec(
                BP.PolynomialBandpassModel(1);
                segmentation = BP.BandpassSegmentation(
                    BP.GlobalTimeSegmentation(),
                    BP.GlobalFrequencySegmentation()
                )
            )
        ),
    )

    @test BP.phase_is_per_scan(model.reference)
    @test !BP.amplitude_is_per_scan(model.reference)
    @test !BP.phase_is_per_scan(model.relative)
    @test occursin("abs(phase=poly1, phase_time=per_scan, amp=poly1, amp_time=global)", BP.station_model_summary("AA", model))

    gain_slice = ComplexF64[
        2.0 * cis(0.1) 3.0 * cis(0.2)
        4.0 * cis(0.3) 5.0 * cis(0.4);

        6.0 * cis(0.5) 7.0 * cis(0.6)
        8.0 * cis(0.7) 9.0 * cis(0.8)
    ]
    gain_slice = reshape(gain_slice, 2, 2, 2)

    scan_gains = ComplexF64[
        20.0 * cis(1.1) 30.0 * cis(1.2)
        40.0 * cis(1.3) 50.0 * cis(1.4);

        60.0 * cis(1.5) 70.0 * cis(1.6)
        80.0 * cis(1.7) 90.0 * cis(1.8)
    ]
    scan_gains = reshape(scan_gains, 2, 2, 2)
    solved = trues(2, 2, 2)
    phase_variable_mask = Bool[1 0; 0 0]
    amplitude_variable_mask = Bool[0 0; 0 1]

    BP.merge_scan_gains!(gain_slice, scan_gains, solved, phase_variable_mask, amplitude_variable_mask)

    @test abs(gain_slice[1, 1, 1]) ≈ 2.0
    @test angle(gain_slice[1, 1, 1]) ≈ 1.1
    @test abs(gain_slice[2, 2, 2]) ≈ 90.0
    @test angle(gain_slice[2, 2, 2]) ≈ 0.8
end

@testset "Bandpass stability plots" begin
    BP = Gustavo.Bandpass
    data = synthetic_uvdata()
    corr = BP.with_visibilities(data, data.vis .* (1.0 + 0.0im), data.weights)
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

    pol_idx, pol_labels = BP.resolve_plot_polarizations(data; pol = :parallel)
    @test pol_idx == [1, 2]
    @test pol_labels == ["11", "22"]

    pol_idx, pol_labels = BP.resolve_plot_polarizations(data; pol = ["22", "12"])
    @test pol_idx == [2, 3]
    @test pol_labels == ["22", "12"]

    @test !isnothing(BP.plot_stability(data, corr, ("AA", "AX"); quantity = :phase, pol = "11"))
    @test !isnothing(BP.plot_stability(data, corr, ("AA", "AX"); quantity = :amplitude, pol = :all, relative = true))
    @test !isnothing(BP.plot_gain_solutions(gains, data))
    @test !isnothing(BP.plot_gain_solutions(gains, data; quantity = :amplitude, pol = 1, sites = "AA", relative = false))
    @test !isnothing(BP.plot_gain_solutions(gains, data; quantity = :phase, pol = [2], sites = ["AX"]))

    fig = BP.plot_stability(data, corr, ("AA", "AX"); quantity = :phase, pol = "11")
    @test_nowarn show(IOBuffer(), MIME("image/png"), fig)
end

@testset "Amplitude stability summary" begin
    BP = Gustavo.Bandpass
    vis_block = ComplexF64[
        1.0 + 0.0im 2.0 + 0.0im;
        -1.0 + 0.0im -2.0 + 0.0im
    ]
    weight_block = ones(Float64, 2, 2)
    groups = [1, 2]

    summary = BP.scan_averaged_amplitude_series(vis_block, weight_block; relative = false, groups = groups)
    @test summary ≈ [1.0, 2.0]

    noise_vis = reshape(ComplexF64[1.0 + 0.0im, 2.0 + 0.0im], 1, 2)
    noise_weights = fill(2.0, 1, 2)

    _, amp_noise = BP.amplitude_series_with_noise(noise_vis, noise_weights; relative = false)
    @test amp_noise ≈ fill(1 / sqrt(2), 2)

    rel_amp, rel_amp_noise = BP.amplitude_series_with_noise(noise_vis, noise_weights; relative = true)
    @test rel_amp ≈ [1.0, 2.0]
    @test rel_amp_noise ≈ [0.0, 1 / sqrt(2)]

    phase, phase_noise = BP.phase_series_with_noise(noise_vis, noise_weights; relative = false)
    @test phase ≈ [0.0, 0.0]
    @test phase_noise ≈ [1 / sqrt(2), 1 / (2sqrt(2))]

    rel_phase, rel_phase_noise = BP.phase_series_with_noise(noise_vis, noise_weights; relative = true)
    @test rel_phase ≈ [0.0, 0.0]
    @test rel_phase_noise ≈ [0.0, 1 / (2sqrt(2))]
end

@testset "Diagnostics series y-limits" begin
    BP = Gustavo.Bandpass

    ylims = BP.finite_series_ylims(([1.0, 2.0, NaN], [4.0]); pad_fraction = 0.1, min_pad = 0.0)
    @test collect(ylims) ≈ [0.7, 4.3]

    ylims_noise = BP.finite_series_ylims(([1.0, 1.0],), ([10.0, 0.2],); pad_fraction = 0.1, min_pad = 0.0, noise_cap_fraction = 0.5)
    @test collect(ylims_noise) ≈ [0.4, 1.6]

    @test isnothing(BP.finite_series_ylims(([NaN], [Inf, -Inf])))
end

@testset "Parallel-hand log-ratio weights" begin
    BP = Gustavo.Bandpass

    vis = zeros(ComplexF64, 2, 1, 2)
    vis[1, 1, 1] = 2.0 + 0.0im
    vis[1, 1, 2] = 4.0 + 0.0im
    vis[2, 1, 1] = 0.0 + 0.0im
    vis[2, 1, 2] = 3.0 + 0.0im

    weights = zeros(Float64, 2, 1, 2)
    weights[1, 1, 1] = 9.0
    weights[1, 1, 2] = 16.0
    weights[2, 1, 1] = 25.0
    weights[2, 1, 2] = 36.0

    ratios, row_weights, rows = BP.collect_parallel_hand_rows(vis, weights, 1, 1, 2)
    weights_shifted_ref = copy(weights)
    weights_shifted_ref[1, 1, 1] = 1.0e6
    _, shifted_ref_row_weights, _ = BP.collect_parallel_hand_rows(vis, weights_shifted_ref, 1, 1, 2)

    expected_variance = inv(16.0 * abs2(4.0 + 0.0im)) + inv(9.0 * abs2(2.0 + 0.0im))
    expected_weight = inv(sqrt(expected_variance))
    shifted_ref_variance = inv(16.0 * abs2(4.0 + 0.0im)) + inv(1.0e6 * abs2(2.0 + 0.0im))
    shifted_ref_weight = inv(sqrt(shifted_ref_variance))

    @test ratios == ComplexF64[2.0 + 0.0im]
    @test row_weights ≈ [expected_weight]
    @test shifted_ref_row_weights ≈ [shifted_ref_weight]
    @test shifted_ref_row_weights[1] > row_weights[1]
    @test rows == [1]

    double_ratio_weight = BP.propagated_log_double_ratio_weight(
        5.0 + 0.0im, 25.0,
        4.0 + 0.0im, 16.0,
        2.0 + 0.0im, 9.0,
        3.0 + 0.0im, 36.0
    )
    expected_double_variance = (
        inv(25.0 * abs2(5.0 + 0.0im)) +
        inv(16.0 * abs2(4.0 + 0.0im)) +
        inv(9.0 * abs2(2.0 + 0.0im)) +
        inv(36.0 * abs2(3.0 + 0.0im))
    )
    @test double_ratio_weight ≈ inv(sqrt(expected_double_variance))
end

@testset "Zero-mean bandpass gauge" begin
    BP = Gustavo.Bandpass

    gains = reshape(
        ComplexF64[
            exp(1.0) * cis(0.7), exp(2.0) * cis(1.2), exp(3.0) * cis(1.7),
            exp(-0.5) * cis(-0.2), exp(0.0) * cis(0.3), exp(0.5) * cis(0.8),
        ],
        1, 2, 3
    )
    support = ones(Float64, 1, 2, 3)

    BP.apply_zero_mean_bandpass_gauge!(gains, support, 2)

    for feed in 1:2
        log_amp = log.(abs.(gains[1, feed, :]))
        phase = BP.unwrap_phase_track(vec(angle.(gains[1, feed, :])), 2)
        @test abs(sum(log_amp) / length(log_amp)) < 1.0e-12
        @test abs(sum(phase) / length(phase)) < 1.0e-12
    end

    gains4 = reshape(
        ComplexF64[
            exp(1.0) * cis(0.2), exp(2.0) * cis(0.4), exp(3.0) * cis(0.6),
            exp(0.0) * cis(-0.1), exp(0.5) * cis(0.1), exp(1.0) * cis(0.3),

            exp(1.5) * cis(0.7), exp(2.5) * cis(0.9), exp(3.5) * cis(1.1),
            exp(-0.5) * cis(-0.4), exp(0.0) * cis(-0.2), exp(0.5) * cis(0.0),
        ],
        2, 1, 2, 3
    )
    BP.apply_zero_mean_bandpass_gauge!(gains4, support, 2)

    for scan in 1:2, feed in 1:2
        log_amp = log.(abs.(gains4[scan, 1, feed, :]))
        phase = BP.unwrap_phase_track(vec(angle.(gains4[scan, 1, feed, :])), 2)
        @test abs(sum(log_amp) / length(log_amp)) < 1.0e-12
        @test abs(sum(phase) / length(phase)) < 1.0e-12
    end
end

@testset "Reference-antenna bandpass gauge" begin
    BP = Gustavo.Bandpass

    gains = reshape(
        ComplexF64[
            exp(1.0) * cis(0.7), exp(2.0) * cis(1.2), exp(3.0) * cis(1.7),
            exp(-0.5) * cis(-0.2), exp(0.0) * cis(0.3), exp(0.5) * cis(0.8),

            exp(0.1) * cis(-0.4), exp(0.2) * cis(-0.1), exp(0.3) * cis(0.2),
            exp(-0.7) * cis(0.5), exp(-0.2) * cis(0.8), exp(0.1) * cis(1.1),
        ],
        2, 2, 3
    )
    support = ones(Float64, 2, 2, 3)
    gains_gauged = copy(gains)

    BP.apply_bandpass_gauge!(gains_gauged, support, 2, BP.ReferenceAntennaBandpassGauge(2))

    for feed in 1:2
        log_amp = log.(abs.(gains_gauged[2, feed, :]))
        phase = BP.unwrap_phase_track(vec(angle.(gains_gauged[2, feed, :])), 2)
        @test abs(sum(log_amp) / length(log_amp)) < 1.0e-12
        @test abs(sum(phase) / length(phase)) < 1.0e-12

        for c in axes(gains, 3)
            @test gains_gauged[1, feed, c] / gains_gauged[2, feed, c] ≈ gains[1, feed, c] / gains[2, feed, c]
        end
    end

    setup = BP.prepare_bandpass_solver(synthetic_bandpass_avg_uvdata(), 1; gauge = BP.ReferenceAntennaBandpassGauge(2))
    @test setup.gauge isa BP.ReferenceAntennaBandpassGauge
    @test setup.gauge.ref_ant == 2
end

@testset "Solve parallel-hand channel ratios" begin
    BP = Gustavo.Bandpass

    nant = 3
    bl_pairs = [(1, 2), (1, 3), (2, 3)]
    npol = 2
    nchan = 6
    c0 = 1

    gains_true = Array{ComplexF64}(undef, nant, nchan)
    for c in 1:nchan
        gains_true[1, c] = (0.95 + 0.03c) * cis(0.04 * (c - 1))
        gains_true[2, c] = (1.10 + 0.02c) * cis(0.10 + 0.05 * (c - 1))
        gains_true[3, c] = (0.85 - 0.02c) * cis(-0.08 - 0.04 * (c - 1))
    end
    A_amp, A_phase = BP.design_matrices(bl_pairs, nant)
    station_models = [BP.StationBandpassModel() for _ in 1:nant]

    Vscan = ones(ComplexF64, length(bl_pairs), npol, nchan)
    source_scan = ComplexF64[1.5 * cis(0.3), 0.6 * cis(-0.4), 1.2 * cis(0.1)]
    for (bi, (a, b)) in enumerate(bl_pairs), pol in 1:npol, c in 1:nchan
        Vscan[bi, pol, c] = source_scan[bi] * gains_true[a, c] * conj(gains_true[b, c])
    end
    Wscan = ones(Float64, size(Vscan))
    gains_scan = ones(ComplexF64, nant, 2, nchan)

    for c in 1:nchan
        c == c0 && continue
        BP.solve_parallel_channel!(
            gains_scan, nothing, Vscan, Wscan, bl_pairs, nant, 1, c0, c, A_amp, A_phase,
            station_models, (1, 2); min_baselines = 3
        )
    end

    for feed in 1:2
        @test gains_scan[:, feed, c0] ≈ ones(ComplexF64, nant) atol = 1.0e-10
        for c in 1:nchan
            c == c0 && continue
            for (bi, (a, b)) in enumerate(bl_pairs)
                solved_ratio = gains_scan[a, feed, c] * conj(gains_scan[b, feed, c])
                expected_ratio = (
                    gains_true[a, c] * conj(gains_true[b, c]) /
                    (gains_true[a, c0] * conj(gains_true[b, c0]))
                )
                @test solved_ratio ≈ expected_ratio atol = 1.0e-10
            end
        end
    end

    nscan = 2
    Vtemplate = ones(ComplexF64, nscan, length(bl_pairs), npol, nchan)
    source_template = reshape(
        ComplexF64[
            1.5 * cis(0.3), 0.6 * cis(-0.4), 1.2 * cis(0.1),
            0.7 * cis(-0.2), 1.1 * cis(0.5), 0.9 * cis(-0.3),
        ],
        nscan, length(bl_pairs)
    )
    for s in 1:nscan, (bi, (a, b)) in enumerate(bl_pairs), pol in 1:npol, c in 1:nchan
        Vtemplate[s, bi, pol, c] = source_template[s, bi] * gains_true[a, c] * conj(gains_true[b, c])
    end
    Wtemplate = ones(Float64, size(Vtemplate))
    gains_template = ones(ComplexF64, nant, 2, nchan)

    for c in 1:nchan
        c == c0 && continue
        BP.solve_parallel_channel!(
            gains_template, nothing, Vtemplate, Wtemplate, bl_pairs, nant, 1, c0, c, A_amp, A_phase,
            station_models, (1, 2); min_baselines = 3
        )
    end

    for feed in 1:2
        @test gains_template[:, feed, c0] ≈ ones(ComplexF64, nant) atol = 1.0e-10
        for c in 1:nchan
            c == c0 && continue
            for (bi, (a, b)) in enumerate(bl_pairs)
                solved_ratio = gains_template[a, feed, c] * conj(gains_template[b, feed, c])
                expected_ratio = (
                    gains_true[a, c] * conj(gains_true[b, c]) /
                    (gains_true[a, c0] * conj(gains_true[b, c0]))
                )
                @test solved_ratio ≈ expected_ratio atol = 1.0e-10
            end
        end
    end
end

@testset "Joint ALS bandpass refinement" begin
    BP = Gustavo.Bandpass

    nant = 3
    bl_pairs = [(1, 2), (1, 3), (2, 3)]
    nchan = 6
    pol_codes = [-1, -2, -3, -4]
    npol = length(pol_codes)
    c0 = 1

    gains_true = Array{ComplexF64}(undef, nant, 2, nchan)
    for c in 1:nchan
        gains_true[1, 1, c] = (0.95 + 0.02c) * cis(0.03 * (c - 1))
        gains_true[2, 1, c] = (1.05 + 0.01c) * cis(0.14 + 0.04 * (c - 1))
        gains_true[3, 1, c] = (0.88 - 0.015c) * cis(-0.09 - 0.05 * (c - 1))

        gains_true[1, 2, c] = gains_true[1, 1, c] * (1.10 - 0.01c) * cis(0.20 + 0.02 * (c - 1))
        gains_true[2, 2, c] = gains_true[2, 1, c] * (0.92 + 0.015c) * cis(-0.15 + 0.01 * (c - 1))
        gains_true[3, 2, c] = gains_true[3, 1, c] * (1.04 - 0.005c) * cis(0.11 - 0.03 * (c - 1))
    end

    source_true = zeros(ComplexF64, length(bl_pairs), 2, 2)
    source_true[1, :, :] .= ComplexF64[1.5 * cis(0.2) 0.3 * cis(-0.4); 0.25 * cis(0.1) 0.9 * cis(0.3)]
    source_true[2, :, :] .= ComplexF64[0.7 * cis(-0.1) 0.2 * cis(0.5); 0.15 * cis(-0.2) 1.2 * cis(-0.3)]
    source_true[3, :, :] .= ComplexF64[1.1 * cis(0.4) 0.35 * cis(0.2); 0.18 * cis(-0.5) 0.8 * cis(0.15)]

    V = zeros(ComplexF64, length(bl_pairs), npol, nchan)
    for (bi, (a, b)) in enumerate(bl_pairs), pol in 1:npol, c in 1:nchan
        fa, fb = BP.stokes_feed_pair(pol_codes[pol])
        V[bi, pol, c] = gains_true[a, fa, c] * source_true[bi, fa, fb] * conj(gains_true[b, fb, c])
    end
    W = ones(Float64, size(V))

    gains_init = ones(ComplexF64, nant, 2, nchan)
    A_amp, A_phase = BP.design_matrices(bl_pairs, nant)
    station_models = [BP.StationBandpassModel() for _ in 1:nant]
    parallel_pols = (1, 2)

    for c in 1:nchan
        c == c0 && continue
        BP.solve_parallel_channel!(
            gains_init, nothing, V, W, bl_pairs, nant, 1, c0, c, A_amp, A_phase,
            station_models, parallel_pols; min_baselines = 3
        )
    end

    source_init = BP.allocate_source_coherencies(V)
    BP.solve_source_coherencies!(source_init, gains_init, V, W, bl_pairs, pol_codes)
    objective_before = BP.joint_bandpass_objective(gains_init, source_init, V, W, bl_pairs, pol_codes)

    gains_before = copy(gains_init)
    support = BP.antenna_feed_support_weights(W, bl_pairs, pol_codes, nant)
    gains_expected = copy(gains_true)
    BP.apply_zero_mean_bandpass_gauge!(gains_expected, support, c0)
    gains_before_gauged = copy(gains_before)
    BP.apply_zero_mean_bandpass_gauge!(gains_before_gauged, support, c0)
    error_before = norm(gains_before_gauged .- gains_expected)

    BP.refine_joint_bandpass_als!(
        gains_init, nothing, V, W, bl_pairs, pol_codes, c0;
        max_iterations = 12, tolerance = 1.0e-10
    )

    source_final = BP.allocate_source_coherencies(V)
    BP.solve_source_coherencies!(source_final, gains_init, V, W, bl_pairs, pol_codes)
    objective_after = BP.joint_bandpass_objective(gains_init, source_final, V, W, bl_pairs, pol_codes)

    gains_estimated = copy(gains_init)
    BP.apply_zero_mean_bandpass_gauge!(gains_estimated, support, c0)
    error_after = norm(gains_estimated .- gains_expected)

    @test objective_after <= objective_before + 1.0e-10
    @test error_after < error_before
end

@testset "Prepared bandpass solver lifecycle" begin
    BP = Gustavo.Bandpass
    data = synthetic_bandpass_avg_uvdata()
    ref_ant = 1
    station_models = [BP.StationBandpassModel() for _ in data.antennas]

    setup = BP.prepare_bandpass_solver(
        data,
        ref_ant;
        min_baselines = 3,
        station_models = station_models,
        phase_ref_ant = ref_ant
    )
    state = BP.initialize_bandpass_state(setup)
    objective_before = BP.bandpass_state_objective(state)

    BP.refine_bandpass!(setup, state, BP.BandpassALS(iterations = 2, tolerance = 1.0e-10))
    objective_after = BP.bandpass_state_objective(state)
    result = BP.finalize_bandpass_state(setup, state; apply_relative_correction = false)

    gains_direct, c0_direct, xy_direct = BP.solve_bandpass(
        data,
        ref_ant;
        min_baselines = 3,
        station_models = station_models,
        phase_ref_ant = ref_ant,
        apply_relative_correction = false,
        joint_als_iterations = 2,
        joint_als_tolerance = 1.0e-10
    )

    @test objective_after <= objective_before + 1.0e-10
    @test result.c0 == c0_direct == setup.c0
    @test Array(result.gains) ≈ Array(gains_direct)
    @test Array(result.xy_correction) ≈ Array(xy_direct)

    stats = BP.bandpass_fit_stats(setup, state)
    merged_source = BP.allocate_source_coherencies(data.vis)
    BP.solve_source_coherencies!(merged_source, state.gains, data.vis, data.weights, data.bl_pairs, data.metadata.pol_codes)
    expected_chi2 = BP.joint_bandpass_objective(state.gains, merged_source, data.vis, data.weights, data.bl_pairs, data.metadata.pol_codes)
    @test stats.chi2 ≈ expected_chi2
    @test stats.nvis > 0
    @test stats.nreal == 2 * stats.nvis
    @test stats.nparams > 0
    @test stats.dof > 0
    @test stats.chi2_per_visibility ≈ stats.chi2 / stats.nvis
    @test stats.chi2_per_real_component ≈ stats.chi2 / stats.nreal
    @test stats.reduced_chi2 ≈ stats.chi2 / stats.dof
end

@testset "Bandpass fit stats parameter counting" begin
    BP = Gustavo.Bandpass

    @test BP.constrained_real_track_parameter_count(Bool[]) == 0
    @test BP.constrained_real_track_parameter_count(trues(1)) == 0
    @test BP.constrained_real_track_parameter_count(trues(4)) == 3

    V = ones(ComplexF64, 1, 1, 2, 3)
    W = ones(Float64, 1, 1, 2, 3)
    pol_codes = [-1, -2]
    @test BP.observed_source_parameter_count(V, W, pol_codes) == 4

    data = synthetic_bandpass_avg_uvdata()
    station_models = [
        BP.StationBandpassModel(),
        BP.StationBandpassModel(
            reference = BP.FeedBandpassModel(
                phase = BP.BandpassSpec(
                    BP.PerChannelBandpassModel();
                    segmentation = BP.BandpassSegmentation(
                        BP.PerScanTimeSegmentation(),
                        BP.GlobalFrequencySegmentation()
                    )
                )
            )
        ),
        BP.StationBandpassModel(),
    ]
    setup = BP.prepare_bandpass_solver(
        data,
        1;
        min_baselines = 3,
        station_models = station_models,
        phase_ref_ant = 1
    )
    state = BP.initialize_bandpass_state(setup)

    expected_gain_params = 0
    for ant in axes(state.gains_template, 1), feed in axes(state.gains_template, 2)
        valid_template = isfinite.(view(state.gains_template, ant, feed, :))
        if !setup.amplitude_variable_mask[ant, feed]
            expected_gain_params += BP.constrained_real_track_parameter_count(valid_template)
        end
        if !setup.phase_variable_mask[ant, feed]
            expected_gain_params += BP.constrained_real_track_parameter_count(valid_template)
        end
        for s in axes(state.scan_gains, 1)
            valid_scan = isfinite.(view(state.scan_gains, s, ant, feed, :)) .& view(state.scan_solved, s, ant, feed, :)
            if setup.amplitude_variable_mask[ant, feed]
                expected_gain_params += BP.constrained_real_track_parameter_count(valid_scan)
            end
            if setup.phase_variable_mask[ant, feed]
                expected_gain_params += BP.constrained_real_track_parameter_count(valid_scan)
            end
        end
    end

    @test BP.effective_gain_parameter_count(setup, state) == expected_gain_params
end

@testset "Bandpass residual stats and plot" begin
    BP = Gustavo.Bandpass
    data = synthetic_bandpass_avg_uvdata()
    ref_ant = 1
    station_models = [BP.StationBandpassModel() for _ in data.antennas]

    setup = BP.prepare_bandpass_solver(
        data,
        ref_ant;
        min_baselines = 3,
        station_models = station_models,
        phase_ref_ant = ref_ant
    )
    state = BP.initialize_bandpass_state(setup)
    BP.refine_bandpass!(setup, state, BP.BandpassALS(iterations = 2, tolerance = 1.0e-10))

    fit_stats = BP.bandpass_fit_stats(setup, state)
    residual_rows = BP.bandpass_residual_stats(setup, state; by = :baseline)
    scan_rows = BP.bandpass_residual_stats(setup, state; by = :scan_baseline)

    @test !isempty(residual_rows)
    @test !isempty(scan_rows)
    @test sum(getindex.(residual_rows, :nvis)) == fit_stats.nvis
    @test sum(getindex.(scan_rows, :nvis)) == fit_stats.nvis
    @test sum(getindex.(residual_rows, :chi2)) ≈ fit_stats.chi2
    @test sum(getindex.(scan_rows, :chi2)) ≈ fit_stats.chi2
    @test all(hasproperty.(residual_rows, :median_abs_normalized_residual))
    @test all(row -> row.normalized_residual_rms ≈ sqrt(row.chi2_per_real_component), residual_rows)

    fig = BP.plot_baseline_bandpass_residuals(setup, state, ("AA", "AX"); pol = :parallel)
    png = repr(MIME("image/png"), fig)
    @test !isempty(png)
end

@testset "Bandpass initializer methods" begin
    BP = Gustavo.Bandpass
    data = synthetic_bandpass_avg_uvdata()
    ref_ant = 1
    station_models = [BP.StationBandpassModel() for _ in data.antennas]

    setup = BP.prepare_bandpass_solver(
        data,
        ref_ant;
        min_baselines = 3,
        station_models = station_models,
        phase_ref_ant = ref_ant
    )

    state_default = BP.initialize_bandpass_state(setup)
    state_ratio = BP.initialize_bandpass_state(setup, BP.RatioBandpassInitializer())
    @test state_default.gains_template ≈ state_ratio.gains_template
    @test state_default.scan_gains ≈ state_ratio.scan_gains
    @test state_default.scan_solved == state_ratio.scan_solved

    rng = MersenneTwister(1234)
    random_initializer = BP.RandomBandpassInitializer(
        rng = rng,
        amplitude_sigma = 0.03,
        phase_sigma = 0.1,
        scan_perturbation = 0.01
    )
    random_state = BP.initialize_bandpass_state(setup, random_initializer)

    @test size(random_state.gains_template) == (length(data.antennas), 2, length(data.metadata.channel_freqs))
    @test size(random_state.scan_gains) == (length(data.scans), length(data.antennas), 2, length(data.metadata.channel_freqs))
    @test all(random_state.scan_solved)
    @test isfinite(BP.bandpass_state_objective(random_state))
    @test norm(random_state.gains_template .- state_ratio.gains_template) > 0
end

@testset "Gain amplitude sanitization" begin
    BP = Gustavo.Bandpass
    gains = ComplexF64[
        0.9 * cis(0.1) 1.0 * cis(0.2) 1.1 * cis(0.3) 0.004 * cis(0.4) 1.2 * cis(0.5);
        1.0 * cis(0.0) 1.0 * cis(0.1) 1.0 * cis(0.2) 1.0 * cis(0.3) 1.0 * cis(0.4);
    ]
    gains = reshape(gains, 1, 2, 5)

    support = ones(Float64, 1, 2, 5)
    repaired = BP.sanitize_gain_amplitudes!(gains, support, 2; collapse_fraction = 0.05, min_gain_amplitude = 1.0e-2, neighbor_window = 1)

    @test length(repaired) == 1
    @test repaired[1].channel == 4
    @test abs(gains[1, 1, 4]) ≈ median([1.1, 1.2])
    @test angle(gains[1, 1, 4]) ≈ 0.4
    @test abs(gains[1, 2, 4]) ≈ 1.0

    @test_logs (:warn, r"repaired collapsed gain amplitudes") BP.warn_sanitized_gain_amplitudes(
        repaired, ["AA"]; context = "scan 1"
    )
end
