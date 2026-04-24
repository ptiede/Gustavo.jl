using Gustavo
using Test
using LinearAlgebra
using Statistics

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

    return Gustavo.Bandpass.UVData(
        vis,
        weights,
        [0.0, 1.0],
        [1, 2],
        [1.0, 1.0],
        [(1, 2)],
        Dict(1.0 => 1),
        [1.0],
        ["AA", "AX"],
        [1, 2],
        [-1, -2, -3, -4],
        ["11", "22", "12", "21"],
        collect(1.0:4.0),
        (2, 4, 4),
        Int[],
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
        reference=BP.FeedBandpassModel(
            phase=BP.BandpassSpec(
                BP.PolynomialBandpassModel(1);
                segmentation=BP.BandpassSegmentation(
                    BP.PerScanTimeSegmentation(),
                    BP.GlobalFrequencySegmentation())),
            amplitude=BP.BandpassSpec(
                BP.PolynomialBandpassModel(1);
                segmentation=BP.BandpassSegmentation(
                    BP.GlobalTimeSegmentation(),
                    BP.GlobalFrequencySegmentation()))),
        relative=BP.FeedBandpassModel(
            phase=BP.BandpassSpec(
                BP.PolynomialBandpassModel(1);
                segmentation=BP.BandpassSegmentation(
                    BP.GlobalTimeSegmentation(),
                    BP.GlobalFrequencySegmentation())),
            amplitude=BP.BandpassSpec(
                BP.PolynomialBandpassModel(1);
                segmentation=BP.BandpassSegmentation(
                    BP.GlobalTimeSegmentation(),
                    BP.GlobalFrequencySegmentation()))),
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
    gains = reshape(ComplexF64[
        1.0 * cis(0.1), 2.0 * cis(0.2),
        3.0 * cis(0.3), 4.0 * cis(0.4),
        5.0 * cis(0.5), 6.0 * cis(0.6),
        7.0 * cis(0.7), 8.0 * cis(0.8),
        1.1 * cis(0.2), 2.1 * cis(0.3),
        3.1 * cis(0.4), 4.1 * cis(0.5),
        5.1 * cis(0.6), 6.1 * cis(0.7),
        7.1 * cis(0.8), 8.1 * cis(0.9),
    ], 2, 2, 2, 2)

    pol_idx, pol_labels = BP.resolve_plot_polarizations(data; pol=:parallel)
    @test pol_idx == [1, 2]
    @test pol_labels == ["11", "22"]

    pol_idx, pol_labels = BP.resolve_plot_polarizations(data; pol=["22", "12"])
    @test pol_idx == [2, 3]
    @test pol_labels == ["22", "12"]

    @test !isnothing(BP.plot_stability(data, corr, ("AA", "AX"); quantity=:phase, pol="11"))
    @test !isnothing(BP.plot_stability(data, corr, ("AA", "AX"); quantity=:amplitude, pol=:all, relative=true))
    @test !isnothing(BP.plot_gain_solutions(gains, data))
    @test !isnothing(BP.plot_gain_solutions(gains, data; quantity=:amplitude, pol=1, sites="AA", relative=false))
    @test !isnothing(BP.plot_gain_solutions(gains, data; quantity=:phase, pol=[2], sites=["AX"]))
end

@testset "Amplitude stability summary" begin
    BP = Gustavo.Bandpass
    vis_block = ComplexF64[
        1.0 + 0.0im 2.0 + 0.0im;
        -1.0 + 0.0im -2.0 + 0.0im
    ]
    weight_block = ones(Float64, 2, 2)
    groups = [1, 2]

    summary = BP.scan_averaged_amplitude_series(vis_block, weight_block; relative=false, groups=groups)
    @test summary ≈ [1.0, 2.0]

    noise_vis = reshape(ComplexF64[1.0 + 0.0im, 2.0 + 0.0im], 1, 2)
    noise_weights = fill(2.0, 1, 2)

    _, amp_noise = BP.amplitude_series_with_noise(noise_vis, noise_weights; relative=false)
    @test amp_noise ≈ fill(1 / sqrt(2), 2)

    phase, phase_noise = BP.phase_series_with_noise(noise_vis, noise_weights; relative=false)
    @test phase ≈ [0.0, 0.0]
    @test phase_noise ≈ [1 / sqrt(2), 1 / (2sqrt(2))]
end

@testset "Gain amplitude sanitization" begin
    BP = Gustavo.Bandpass
    gains = ComplexF64[
        0.9 * cis(0.1) 1.0 * cis(0.2) 1.1 * cis(0.3) 0.004 * cis(0.4) 1.2 * cis(0.5);
        1.0 * cis(0.0) 1.0 * cis(0.1) 1.0 * cis(0.2) 1.0 * cis(0.3) 1.0 * cis(0.4);
    ]
    gains = reshape(gains, 1, 2, 5)

    support = ones(Float64, 1, 2, 5)
    repaired = BP.sanitize_gain_amplitudes!(gains, support, 2; collapse_fraction=0.05, min_gain_amplitude=1e-2, neighbor_window=1)

    @test length(repaired) == 1
    @test repaired[1].channel == 4
    @test abs(gains[1, 1, 4]) ≈ median([1.1, 1.2])
    @test angle(gains[1, 1, 4]) ≈ 0.4
    @test abs(gains[1, 2, 4]) ≈ 1.0

    @test_logs (:warn, r"repaired collapsed gain amplitudes") BP.warn_sanitized_gain_amplitudes(
        repaired, ["AA"]; context="scan 1")
end
