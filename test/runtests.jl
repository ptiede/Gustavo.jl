using Gustavo
using Test
using LinearAlgebra
using Statistics
using Random
using StructArrays
using FITSFiles: Card
using CairoMakie
using DimensionalData: DimArray, DimStack, dims, Ti
using Gustavo.UVData: Integration, Pol, IF, UVW, Scan, Baseline, UVSet, pol_products
using PolarizedTypes: RPol, LPol

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
    vis = ComplexF32.(reshape(vis, 2, 4, 4))
    weights = fill(1.0f0, size(vis))
    obs_time_synth = [0.0, 1.0]
    # Internal MSv4-canonical correlation order. AIPS Stokes axis on disk is
    # [-1,-2,-3,-4] (RR/LL/RL/LR) which maps to ["PP","QQ","PQ","QP"]; the
    # FITS read path permutes to MSv4 ["PP","PQ","QP","QQ"]. Since the
    # synthetic vis values are placeholders, we just label the dim in MSv4
    # order — the round-trip test exercises the read/write permutation.
    pol_labels_synth = ["PP", "PQ", "QP", "QQ"]
    channel_freqs_synth = collect(1.0:4.0)
    vis = DimArray(vis, (Integration(obs_time_synth), Pol(pol_labels_synth), IF(channel_freqs_synth)))
    weights = DimArray(weights, (Integration(obs_time_synth), Pol(pol_labels_synth), IF(channel_freqs_synth)))
    uvw = DimArray(zeros(Float32, 2, 3), (Integration(obs_time_synth), UVW(["U", "V", "W"])))

    UV = Gustavo.UVData
    nominal_basis_v = [(RPol(), LPol()), (RPol(), LPol())]
    response_v = [Diagonal(ones(ComplexF32, 2)) for _ in 1:2]
    pol_angles_v = [(0.0f0, 0.0f0), (0.0f0, 0.0f0)]
    antennas_v = [
        UV.Antenna(;
            name = "AA",
            station_xyz = zeros(3),
            mount = UV.MountAltAz(),
            nominal_basis = nominal_basis_v[1],
            response = response_v[1],
            pol_angles = pol_angles_v[1],
        ),
        UV.Antenna(;
            name = "AX",
            station_xyz = zeros(3),
            mount = UV.MountAltAz(),
            nominal_basis = nominal_basis_v[2],
            response = response_v[2],
            pol_angles = pol_angles_v[2],
        ),
    ]
    antennas = UV.AntennaTable(
        StructArray(antennas_v), zeros(3), "TEST",
        (POLCALA = [Float32[], Float32[]], POLCALB = [Float32[], Float32[]]),
    )
    array_config = UV.ArrayConfig(
        "2000-01-01", 0.0f0, 360.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
        "UTC", "ITRF", "RIGHT", "APPROX",
        Int32(1), Int32(0), Int32(4), Int32(0), Int32(1),
    )
    freq_setup = UV.FrequencySetup(;
        name = "FRQSEL_1",
        ref_freq = 1.0e9,
        channel_freqs = collect(1.0:4.0),
        ch_widths = fill(1.0f0, 4),
        total_bandwidths = fill(1.0f0, 4),
        sidebands = Int32.(fill(1, 4)),
    )
    array_obs = UV.ObsArrayMetadata(;
        telescope = "TEST", instrume = "TEST",
        date_obs = "2000-01-01", equinox = 2000.0f0, bunit = "JY",
        freq_setup,
    )

    # Memo-117-shaped primary HDU cards: NAXIS=7 random-groups layout plus
    # CTYPE cards for the regular axes (so STOKES/FREQ/RA/DEC round-trip
    # through `parse_stokes_axis` and friends), plus PTYPE cards for the
    # canonical random parameters.
    primary_cards = Card[
        Card("NAXIS", 7),
        Card("OBJECT", "TEST"),
        Card("TELESCOP", "TEST"),
        Card("INSTRUME", "TEST"),
        Card("DATE-OBS", "2000-01-01"),
        Card("CTYPE2", "COMPLEX"),
        Card("CRVAL2", 1.0), Card("CDELT2", 1.0), Card("CRPIX2", 1.0),
        Card("CTYPE3", "STOKES"),
        Card("CRVAL3", -1.0), Card("CDELT3", -1.0), Card("CRPIX3", 1.0),
        Card("CTYPE4", "FREQ"),
        Card("CRVAL4", 1.0e9), Card("CDELT4", 1.0), Card("CRPIX4", 1.0),
        Card("CTYPE5", "IF"),
        Card("CTYPE6", "RA"), Card("CRVAL6", 0.0),
        Card("CTYPE7", "DEC"), Card("CRVAL7", 0.0),
        Card("PTYPE1", "UU---SIN"),
        Card("PTYPE2", "VV---SIN"),
        Card("PTYPE3", "WW---SIN"),
        Card("PTYPE4", "BASELINE"),
        Card("PTYPE5", "DATE"),
    ]
    return UVSet(
        (
            vis = vis,
            weights = weights,
            uvw = uvw,
            obs_time = obs_time_synth,
            scan_idx = [1, 2],
            baselines = UV.BaselineIndex([1, 1], [(1, 2)], Dict(1 => 1), [1]; antenna_names = ["AA", "AX"]),
            # Single DATE PTYPE column carrying obs_time/24 (hours → days).
            # Whole-JD + fractional-day style needs two PTYPE columns which
            # FITSFiles can't deduplicate on the round-trip path.
            date_param = reshape(Float32[0.0, 1 / 24], :, 1),
            extra_columns = NamedTuple(),
            scans = StructArray(lower = [0.0, 1.0], upper = [1.0, 2.0]),
            antennas = antennas,
            array_config = array_config,
            array_obs = array_obs,
            source_name = "TEST",
            ra = 0.0, dec = 0.0,
            primary_cards = primary_cards,
            basename = "synthetic",
        )
    )
end

function synthetic_bandpass_avg_uvdata()
    BP = Gustavo.Bandpass
    nant = 3
    bl_pairs = [(1, 2), (1, 3), (2, 3)]
    ant_names = ["AA", "AX", "NN"]
    pol_labels = ["PP", "PQ", "QP", "QQ"]   # MSv4-canonical order
    nscan = 2
    nchan = 6

    gains_true = Array{ComplexF64}(undef, nant, 2, nchan)
    for c in 1:nchan
        gains_true[1, 1, c] = (0.95 + 0.02c) * cis(0.03 * (c - 1))
        gains_true[2, 1, c] = (1.05 + 0.01c) * cis(0.14 + 0.04 * (c - 1))
        gains_true[3, 1, c] = (0.88 - 0.015c) * cis(-0.09 - 0.05 * (c - 1))

        gains_true[1, 2, c] = gains_true[1, 1, c] * (1.1 - 0.01c) * cis(0.2 + 0.02 * (c - 1))
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

    vis_arr = zeros(ComplexF64, nscan, length(bl_pairs), length(pol_labels), nchan)
    for s in 1:nscan, (bi, (a, b)) in enumerate(bl_pairs), pol in eachindex(pol_labels), c in 1:nchan
        fa, fb = BP.correlation_feed_pair(pol_labels[pol])
        vis_arr[s, bi, pol, c] = gains_true[a, fa, c] * source_true[s, bi, fa, fb] * conj(gains_true[b, fb, c])
    end
    weights_arr = ones(Float64, size(vis_arr))
    bl_labels = [string(ant_names[a], "-", ant_names[b]) for (a, b) in bl_pairs]
    scan_centers = (Float64.(0:(nscan - 1)) .+ Float64.(1:nscan)) ./ 2
    channel_freqs_synth = collect(1.0:nchan)
    vis = DimArray(vis_arr, (Scan(scan_centers), Baseline(bl_labels), Pol(pol_labels), IF(channel_freqs_synth)))
    weights = DimArray(weights_arr, (Scan(scan_centers), Baseline(bl_labels), Pol(pol_labels), IF(channel_freqs_synth)))
    uvw = DimArray(zeros(nscan, 3), (Integration(collect(0.0:(nscan - 1))), UVW(["U", "V", "W"])))

    UV = Gustavo.UVData
    antennas_v = [
        UV.Antenna(;
            name = ant_names[i],
            station_xyz = zeros(3),
            mount = UV.MountAltAz(),
            nominal_basis = (RPol(), LPol()),
            response = Diagonal(ones(ComplexF32, 2)),
            pol_angles = (0.0f0, 0.0f0),
        )
        for i in 1:nant
    ]
    antennas = UV.AntennaTable(
        StructArray(antennas_v), zeros(3), "TEST",
        (POLCALA = [Float32[] for _ in 1:nant], POLCALB = [Float32[] for _ in 1:nant]),
    )
    array_config = UV.ArrayConfig(
        "2000-01-01", 0.0f0, 360.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
        "UTC", "ITRF", "RIGHT", "APPROX",
        Int32(1), Int32(0), Int32(nchan), Int32(0), Int32(1),
    )
    freq_setup = UV.FrequencySetup(;
        name = "FRQSEL_1",
        ref_freq = 1.0e9,
        channel_freqs = collect(1.0:nchan),
        ch_widths = fill(1.0f0, nchan),
        total_bandwidths = fill(1.0f0, nchan),
        sidebands = Int32.(fill(1, nchan)),
    )
    array_obs = UV.ObsArrayMetadata(;
        telescope = "TEST", instrume = "TEST",
        date_obs = "2000-01-01", equinox = 2000.0f0, bunit = "JY",
        freq_setup,
    )

    baselines_idx = UV.BaselineIndex(
        collect(1:length(bl_pairs)),
        bl_pairs,
        Dict(i => i for i in 1:length(bl_pairs)),
        collect(1:length(bl_pairs));
        antenna_names = ant_names,
    )
    scans_struct = StructArray(lower = Float64.(0:(nscan - 1)), upper = Float64.(1:nscan))
    return BP.BandpassDataset(
        vis, weights, uvw, baselines_idx,
        antennas, array_obs, scans_struct,
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
    UV = Gustavo.UVData
    data = synthetic_uvdata()
    corr = UV.apply((leaf, _info, _meta) -> UV.with_visibilities(leaf, parent(leaf[:vis]) .* (1.0 + 0.0im), parent(leaf[:weights])), data)

    # Plot helpers operate on UVSet directly. The resolver tests below
    # also accept a UVSet (via the _DataLike accessors).
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
    @test pol_idx == [1, 4]
    @test pol_labels == ["PP", "QQ"]

    pol_idx, pol_labels = BP.resolve_plot_polarizations(data; pol = ["QQ", "PQ"])
    @test pol_idx == [4, 2]
    @test pol_labels == ["QQ", "PQ"]

    @test !isnothing(BP.plot_stability(data, corr, ("AA", "AX"); quantity = :phase, pol = "PP"))
    @test !isnothing(BP.plot_stability(data, corr, ("AA", "AX"); quantity = :amplitude, pol = :all, relative = true))
    @test !isnothing(BP.plot_gain_solutions(gains, data))
    @test !isnothing(BP.plot_gain_solutions(gains, data; quantity = :amplitude, pol = 1, sites = "AA", relative = false))
    @test !isnothing(BP.plot_gain_solutions(gains, data; quantity = :phase, pol = [2], sites = ["AX"]))
    fig_embed = Figure(size = (1400, 500))
    @test !isnothing(BP.plot_stability(fig_embed[1, 1], data, corr, ("AA", "AX"); quantity = :phase, pol = "PP"))
    @test !isnothing(BP.plot_gain_solutions(fig_embed[1, 2], gains, data; quantity = :phase, pol = [2], sites = ["AX"]))

    fig = BP.plot_stability(data, corr, ("AA", "AX"); quantity = :phase, pol = "PP")
    @test_nowarn show(IOBuffer(), MIME("image/png"), fig)
    @test_nowarn show(IOBuffer(), MIME("image/png"), fig_embed)
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
    @test isequal(BP.shared_track(([1.0, 2.0, NaN], [1.0, 2.0, NaN])), [1.0, 2.0, NaN])
    @test isnothing(BP.shared_track(([1.0, 2.0], [1.0, 3.0])))
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

    # Wrap as DimArrays (Baseline × Pol × IF) so collect_parallel_hand_rows'
    # dim-agnostic loop sees named axes — exercising the same code path the
    # solver hits on real data.
    bl_dim = Baseline(["A", "B"])
    pol_dim = Pol(["PP"])
    if_dim = IF([1.0, 2.0])
    vis_d = DimArray(vis, (bl_dim, pol_dim, if_dim))
    weights_d = DimArray(weights, (bl_dim, pol_dim, if_dim))

    ratios, row_weights, rows = BP.collect_parallel_hand_rows(vis_d, weights_d, 1, 1, 2)
    weights_shifted_ref = copy(weights)
    weights_shifted_ref[1, 1, 1] = 1.0e6
    weights_shifted_d = DimArray(weights_shifted_ref, (bl_dim, pol_dim, if_dim))
    _, shifted_ref_row_weights, _ = BP.collect_parallel_hand_rows(vis_d, weights_shifted_d, 1, 1, 2)

    expected_variance = inv(16.0 * abs2(4.0 + 0.0im)) + inv(9.0 * abs2(2.0 + 0.0im))
    expected_weight = inv(expected_variance)
    shifted_ref_variance = inv(16.0 * abs2(4.0 + 0.0im)) + inv(1.0e6 * abs2(2.0 + 0.0im))
    shifted_ref_weight = inv(shifted_ref_variance)

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
    @test double_ratio_weight ≈ inv(expected_double_variance)
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

    validated = BP.validate_bandpass_gauge(BP.ReferenceAntennaBandpassGauge(2), 3)
    @test validated isa BP.ReferenceAntennaBandpassGauge
    @test validated.ref_ant == 2
    @test_throws ErrorException BP.validate_bandpass_gauge(BP.ReferenceAntennaBandpassGauge(4), 3)
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
        gains_true[2, c] = (1.1 + 0.02c) * cis(0.1 + 0.05 * (c - 1))
        gains_true[3, c] = (0.85 - 0.02c) * cis(-0.08 - 0.04 * (c - 1))
    end
    A_amp, A_phase = BP.design_matrices(bl_pairs, nant)
    station_models = [BP.StationBandpassModel() for _ in 1:nant]

    Vscan_arr = ones(ComplexF64, length(bl_pairs), npol, nchan)
    source_scan = ComplexF64[1.5 * cis(0.3), 0.6 * cis(-0.4), 1.2 * cis(0.1)]
    for (bi, (a, b)) in enumerate(bl_pairs), pol in 1:npol, c in 1:nchan
        Vscan_arr[bi, pol, c] = source_scan[bi] * gains_true[a, c] * conj(gains_true[b, c])
    end
    Wscan_arr = ones(Float64, size(Vscan_arr))
    Vscan = DimArray(Vscan_arr, (Baseline(["AB", "AC", "BC"]), Pol(["PP", "QQ"]), IF(Float64.(1:nchan))))
    Wscan = DimArray(Wscan_arr, dims(Vscan))
    gains_scan = ones(ComplexF64, nant, 2, nchan)
    ref_gauge = BP.ReferenceAntennaBandpassGauge(1)

    for c in 1:nchan
        c == c0 && continue
        BP.solve_parallel_channel!(
            gains_scan, nothing, Vscan, Wscan, bl_pairs, nant, ref_gauge, c0, c, A_amp, A_phase,
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
    Vtemplate_arr = ones(ComplexF64, nscan, length(bl_pairs), npol, nchan)
    source_template = reshape(
        ComplexF64[
            1.5 * cis(0.3), 0.6 * cis(-0.4), 1.2 * cis(0.1),
            0.7 * cis(-0.2), 1.1 * cis(0.5), 0.9 * cis(-0.3),
        ],
        nscan, length(bl_pairs)
    )
    for s in 1:nscan, (bi, (a, b)) in enumerate(bl_pairs), pol in 1:npol, c in 1:nchan
        Vtemplate_arr[s, bi, pol, c] = source_template[s, bi] * gains_true[a, c] * conj(gains_true[b, c])
    end
    Wtemplate_arr = ones(Float64, size(Vtemplate_arr))
    Vtemplate = DimArray(Vtemplate_arr, (Scan(Float64.(1:nscan)), Baseline(["AB", "AC", "BC"]), Pol(["PP", "QQ"]), IF(Float64.(1:nchan))))
    Wtemplate = DimArray(Wtemplate_arr, dims(Vtemplate))
    gains_template = ones(ComplexF64, nant, 2, nchan)

    for c in 1:nchan
        c == c0 && continue
        BP.solve_parallel_channel!(
            gains_template, nothing, Vtemplate, Wtemplate, bl_pairs, nant, ref_gauge, c0, c, A_amp, A_phase,
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
    pol_products_v = ["PP", "PQ", "QP", "QQ"]
    npol = length(pol_products_v)
    c0 = 1

    gains_true = Array{ComplexF64}(undef, nant, 2, nchan)
    for c in 1:nchan
        gains_true[1, 1, c] = (0.95 + 0.02c) * cis(0.03 * (c - 1))
        gains_true[2, 1, c] = (1.05 + 0.01c) * cis(0.14 + 0.04 * (c - 1))
        gains_true[3, 1, c] = (0.88 - 0.015c) * cis(-0.09 - 0.05 * (c - 1))

        gains_true[1, 2, c] = gains_true[1, 1, c] * (1.1 - 0.01c) * cis(0.2 + 0.02 * (c - 1))
        gains_true[2, 2, c] = gains_true[2, 1, c] * (0.92 + 0.015c) * cis(-0.15 + 0.01 * (c - 1))
        gains_true[3, 2, c] = gains_true[3, 1, c] * (1.04 - 0.005c) * cis(0.11 - 0.03 * (c - 1))
    end

    source_true = zeros(ComplexF64, length(bl_pairs), 2, 2)
    source_true[1, :, :] .= ComplexF64[1.5 * cis(0.2) 0.3 * cis(-0.4); 0.25 * cis(0.1) 0.9 * cis(0.3)]
    source_true[2, :, :] .= ComplexF64[0.7 * cis(-0.1) 0.2 * cis(0.5); 0.15 * cis(-0.2) 1.2 * cis(-0.3)]
    source_true[3, :, :] .= ComplexF64[1.1 * cis(0.4) 0.35 * cis(0.2); 0.18 * cis(-0.5) 0.8 * cis(0.15)]

    V_arr = zeros(ComplexF64, length(bl_pairs), npol, nchan)
    for (bi, (a, b)) in enumerate(bl_pairs), pol in 1:npol, c in 1:nchan
        fa, fb = BP.correlation_feed_pair(pol_products_v[pol])
        V_arr[bi, pol, c] = gains_true[a, fa, c] * source_true[bi, fa, fb] * conj(gains_true[b, fb, c])
    end
    W_arr = ones(Float64, size(V_arr))
    V = DimArray(V_arr, (Baseline(["AB", "AC", "BC"]), Pol(pol_products_v), IF(Float64.(1:nchan))))
    W = DimArray(W_arr, dims(V))

    gains_init = ones(ComplexF64, nant, 2, nchan)
    A_amp, A_phase = BP.design_matrices(bl_pairs, nant)
    station_models = [BP.StationBandpassModel() for _ in 1:nant]
    parallel_pols = (1, 2)

    for c in 1:nchan
        c == c0 && continue
        BP.solve_parallel_channel!(
            gains_init, nothing, V, W, bl_pairs, nant, BP.ReferenceAntennaBandpassGauge(1), c0, c, A_amp, A_phase,
            station_models, parallel_pols; min_baselines = 3
        )
    end

    source_init = BP.allocate_source_coherencies(V)
    BP.solve_source_coherencies!(source_init, gains_init, V, W, bl_pairs, pol_products_v)
    objective_before = BP.joint_bandpass_objective(gains_init, source_init, V, W, bl_pairs, pol_products_v)

    gains_before = copy(gains_init)
    support = BP.antenna_feed_support_weights(W, bl_pairs, pol_products_v, nant)
    gains_expected = copy(gains_true)
    BP.apply_zero_mean_bandpass_gauge!(gains_expected, support, c0)
    gains_before_gauged = copy(gains_before)
    BP.apply_zero_mean_bandpass_gauge!(gains_before_gauged, support, c0)
    error_before = norm(gains_before_gauged .- gains_expected)

    BP.refine_joint_bandpass_als!(
        gains_init, nothing, V, W, bl_pairs, pol_products_v, c0;
        max_iterations = 12, tolerance = 1.0e-10
    )

    source_final = BP.allocate_source_coherencies(V)
    BP.solve_source_coherencies!(source_final, gains_init, V, W, bl_pairs, pol_products_v)
    objective_after = BP.joint_bandpass_objective(gains_init, source_final, V, W, bl_pairs, pol_products_v)

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
        station_models = station_models
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
    BP.solve_source_coherencies!(merged_source, state.gains, data.vis, data.weights, data.baselines.pairs, pol_products(data))
    expected_chi2 = BP.joint_bandpass_objective(state.gains, merged_source, data.vis, data.weights, data.baselines.pairs, pol_products(data))
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
    pol_products_v = ["PP", "QQ"]
    @test BP.observed_source_parameter_count(V, W, pol_products_v) == 4

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
        station_models = station_models
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
        station_models = station_models
    )
    state = BP.initialize_bandpass_state(setup)
    BP.refine_bandpass!(setup, state, BP.BandpassALS(iterations = 2, tolerance = 1.0e-10))

    fit_stats = BP.bandpass_fit_stats(setup, state)
    residual_rows = BP.bandpass_residual_stats(setup, state; by = :baseline)
    scan_rows = BP.bandpass_residual_stats(setup, state; by = :scan_baseline)
    result = BP.finalize_bandpass_state(setup, state; apply_relative_correction = false)
    final_fit_stats = BP.bandpass_fit_stats(setup, result.gains)
    final_residual_rows = BP.bandpass_residual_stats(setup, result.gains; by = :baseline)

    @test !isempty(residual_rows)
    @test !isempty(scan_rows)
    @test !isempty(final_residual_rows)
    @test sum(getindex.(residual_rows, :nvis)) == fit_stats.nvis
    @test sum(getindex.(scan_rows, :nvis)) == fit_stats.nvis
    @test sum(getindex.(residual_rows, :chi2)) ≈ fit_stats.chi2
    @test sum(getindex.(scan_rows, :chi2)) ≈ fit_stats.chi2
    @test sum(getindex.(final_residual_rows, :chi2)) ≈ final_fit_stats.chi2
    @test all(hasproperty.(residual_rows, :median_abs_normalized_residual))
    @test all(row -> row.normalized_residual_rms ≈ sqrt(row.chi2_per_real_component), residual_rows)
    @test final_fit_stats.nparams === missing
    @test final_fit_stats.dof === missing
    @test final_fit_stats.reduced_chi2 === missing

    bi = findfirst(==((1, 2)), data.baselines.pairs)
    observed, observed_weights, gain_model, normalized_residual, weights = BP.baseline_bandpass_diagnostics(setup, result.gains, bi, 1)
    source = BP.fit_bandpass_source_coherencies(setup, result.gains)
    for s in axes(data.vis, 1), c in axes(data.vis, 4)
        w = data.weights[s, bi, 1, c]
        v = data.vis[s, bi, 1, c]
        if w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))
            a, b = data.baselines.pairs[bi]
            m = result.gains[s, a, 1, c] * source[s, bi, 1, 1] * conj(result.gains[s, b, 1, c])
            @test normalized_residual[s, c] ≈ sqrt(w) * (v - m)
        end
    end
    @test size(observed) == size(observed_weights) == size(gain_model) == size(normalized_residual) == size(weights)

    fig_bandpass = BP.plot_baseline_bandpass(setup, result.gains, ("AA", "AX"); pol = :parallel)
    @test !isempty(repr(MIME("image/png"), fig_bandpass))

    fig_embed = Figure(size = (1800, 700))
    @test !isnothing(BP.plot_baseline_bandpass(fig_embed[1, 1], setup, result.gains, ("AA", "AX"); pol = :parallel))
    @test !isnothing(BP.plot_baseline_bandpass_residuals(fig_embed[1, 2], setup, result.gains, ("AA", "AX"); pol = :parallel))
    @test !isempty(repr(MIME("image/png"), fig_embed))

    fig = BP.plot_baseline_bandpass_residuals(setup, result.gains, ("AA", "AX"); pol = :parallel)
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
        station_models = station_models
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

    @test size(random_state.gains_template) == (length(data.antennas), 2, length(data.metadata.freq_setup.channel_freqs))
    @test size(random_state.scan_gains) == (length(data.scans), length(data.antennas), 2, length(data.metadata.freq_setup.channel_freqs))
    @test all(random_state.scan_solved)
    @test isfinite(BP.bandpass_state_objective(random_state))
    @test norm(random_state.gains_template .- state_ratio.gains_template) > 0
end

@testset "Gain amplitude sanitization" begin
    BP = Gustavo.Bandpass

    # c0 = 1. Channel 3 (ant 1, feed 1) is genuinely broken (zero magnitude);
    # channel 5 has a legitimately tiny but finite amplitude (e.g. IF-edge
    # rolloff). Only the zero-magnitude gain gets sanitized; the small-but-
    # finite value is left alone but surfaced separately by
    # `inspect_collapsed_gain_amplitudes`.
    gains = ComplexF64[
        1.0 + 0.0im   0.9 * cis(0.1)  0.0 + 0.0im     1.1 * cis(0.3)  0.004 * cis(0.4);
        1.0 + 0.0im   1.0 * cis(0.1)  1.0 * cis(0.2)  1.0 * cis(0.3)  1.0 * cis(0.4);
    ]
    gains = reshape(gains, 1, 2, 5)
    support = ones(Float64, 1, 2, 5)

    repaired = BP.sanitize_gain_amplitudes!(gains, support, 1; neighbor_window = 1)

    @test length(repaired) == 1
    @test repaired[1].channel == 3
    @test abs(gains[1, 1, 3]) ≈ median([0.9, 1.1])  # local neighbors at c±1 (c0=1 excluded)
    @test isfinite(gains[1, 1, 5]) && abs(gains[1, 1, 5]) ≈ 0.004  # finite-tiny untouched

    @test_logs (:warn, r"repaired") BP.warn_sanitized_gain_amplitudes(
        repaired, ["AA"]; context = "scan 1"
    )

    suspects = BP.inspect_collapsed_gain_amplitudes(
        gains, support, 1;
        collapse_fraction = 0.05, min_gain_amplitude = 1.0e-2, neighbor_window = 1
    )
    @test length(suspects) == 1
    @test suspects[1].channel == 5
    @test suspects[1].amplitude ≈ 0.004
    @test_logs (:warn, r"collapsed gain amplitudes detected") BP.warn_collapsed_gain_amplitudes(
        suspects, ["AA"]; context = "scan 1"
    )
end

@testset "write_uvfits HDU construction" begin
    UV = Gustavo.UVData
    data = synthetic_uvdata()

    # End-to-end: write_uvfits should run on the synthetic dataset.
    tmp = tempname() * ".uvfits"
    try
        @test UV.write_uvfits(tmp, data) == tmp
        @test isfile(tmp)
        @test filesize(tmp) > 0
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
end

@testset "UVData round-trip preserves canonical fields" begin
    UV = Gustavo.UVData
    data = synthetic_uvdata()

    fs_orig = UV.metadata(data).array_obs.freq_setup
    @test eltype(fs_orig.channel_freqs) == Float64
    @test eltype(fs_orig.ch_widths) == Float32
    @test eltype(fs_orig.total_bandwidths) == Float32
    @test eltype(fs_orig.sidebands) == Int32

    tmp = tempname() * ".uvfits"
    try
        UV.write_uvfits(tmp, data)
        round_set = UV.load_uvfits(tmp)
        @test round_set isa UV.UVSet

        # Compare leaves directly: load → write → reload should preserve
        # vis/weights/uvw cubes per partition and the baseline codes.
        for (k, leaf_orig) in pairs(UV.branches(data))
            leaf_round = UV.branches(round_set)[k]
            @test parent(leaf_round[:uvw]) == Float32.(parent(leaf_orig[:uvw]))
            @test parent(leaf_round[:weights]) == Float32.(parent(leaf_orig[:weights]))
            @test UV.baselines(leaf_round).unique_codes == UV.baselines(leaf_orig).unique_codes
        end

        fs_round = UV.metadata(round_set).array_obs.freq_setup
        @test fs_round.channel_freqs == fs_orig.channel_freqs
        @test fs_round.ch_widths == fs_orig.ch_widths
        @test fs_round.total_bandwidths == fs_orig.total_bandwidths
        @test fs_round.sidebands == fs_orig.sidebands
        @test pol_products(round_set) == pol_products(data)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
end

@testset "UVDataset extra_columns + date_param preservation" begin
    UV = Gustavo.UVData
    base = synthetic_uvdata()
    base_root = UV.metadata(base)

    # Materialise the flat record stream we need to feed UVSet's NamedTuple
    # factory. Walk leaves in order to assemble per-record arrays with the
    # same shape the load path produces.
    leaves_collected = collect(values(UV.branches(base)))
    obs_times = Float64[]
    scan_indices = Int[]
    bl_codes = Int[]
    vis_chunks = Any[]
    weights_chunks = Any[]
    uvw_chunks = Any[]
    for leaf in leaves_collected
        info = UV.metadata(leaf)
        ti_lookup = collect(UV.obs_time(leaf))
        bls = info.baselines
        vis_p = parent(leaf[:vis])
        w_p = parent(leaf[:weights])
        uvw_p = parent(leaf[:uvw])
        for (ti, bi) in info.record_order
            push!(obs_times, ti_lookup[ti])
            push!(scan_indices, info.scan_idx)
            push!(bl_codes, round(Int, bls.unique_codes[bi]))
            push!(vis_chunks, vis_p[ti, bi, :, :])
            push!(weights_chunks, w_p[ti, bi, :, :])
            push!(uvw_chunks, uvw_p[ti, bi, :])
        end
    end
    nint = length(obs_times)
    npol = size(vis_chunks[1], 1)
    nchan = size(vis_chunks[1], 2)
    vis_flat = Array{ComplexF32, 3}(undef, nint, npol, nchan)
    weights_flat = Array{Float32, 3}(undef, nint, npol, nchan)
    uvw_flat = Array{Float32, 2}(undef, nint, 3)
    for i in 1:nint
        vis_flat[i, :, :] .= vis_chunks[i]
        weights_flat[i, :, :] .= weights_chunks[i]
        uvw_flat[i, :] .= uvw_chunks[i]
    end
    pol_labels = pol_products(base)
    chan_freqs = base_root.array_obs.freq_setup.channel_freqs
    vis_da = DimArray(vis_flat, (UV.Integration(obs_times), UV.Pol(pol_labels), UV.IF(chan_freqs)))
    weights_da = DimArray(weights_flat, (UV.Integration(obs_times), UV.Pol(pol_labels), UV.IF(chan_freqs)))
    uvw_da = DimArray(uvw_flat, (UV.Integration(obs_times), UV.UVW(["U", "V", "W"])))

    unique_codes = sort(unique(bl_codes))
    bl_lookup = Dict(c => i for (i, c) in enumerate(unique_codes))
    bl_pairs = [UV.decode_baseline(c) for c in unique_codes]
    bls = UV.BaselineIndex(bl_codes, bl_pairs, bl_lookup, unique_codes; antenna_names = base_root.antennas.name)

    inttim_values = Float32.(0.5 .* (1:nint))
    primary_cards = Card[
        Card("PTYPE1", "UU---SIN"),
        Card("PTYPE2", "VV---SIN"),
        Card("PTYPE3", "WW---SIN"),
        Card("PTYPE4", "BASELINE"),
        Card("PTYPE5", "DATE"),
        Card("PTYPE6", "DATE"),
        Card("PTYPE7", "INTTIM"),
        Card("PCOUNT", 7),
    ]
    extras = (var"INTTIM" = inttim_values,)
    date_param = hcat(fill(Float32(2_459_664.5), nint), Float32.(obs_times ./ 24))

    uvset = UV.UVSet(
        (
            vis = vis_da, weights = weights_da, uvw = uvw_da,
            obs_time = obs_times, scan_idx = scan_indices,
            baselines = bls,
            date_param = date_param, extra_columns = extras,
            scans = base_root.scans, antennas = base_root.antennas,
            array_config = base_root.array_config,
            array_obs = base_root.array_obs,
            source_name = "TEST", ra = 0.0, dec = 0.0,
            primary_cards = primary_cards,
            basename = "synthetic",
        )
    )

    # Verify date_param + INTTIM round-trip through the leaves.
    leaves_collected = collect(values(UV.branches(uvset)))
    cat_dates = vcat([UV.date_param(l) for l in leaves_collected]...)
    cat_inttim = vcat([UV.extra_columns(l).INTTIM for l in leaves_collected]...)
    @test cat_dates == date_param
    @test cat_inttim == inttim_values

    tmp = tempname() * ".uvfits"
    try
        @test UV.write_uvfits(tmp, uvset) == tmp
        @test filesize(tmp) > 0
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
end

@testset "antenna_phase_weights uses |v|² · w" begin
    # Regression: under the new inverse-variance WLS convention, channel
    # weights for the polynomial bandpass fits must be the inverse variance
    # of the *phasetrack*, i.e. Σ_bl |v|² · w_raw (`log_visibility_precision`),
    # not bare Σ w_raw. With bare-w, high-SNR channels (AA-AX-rich) get
    # sqrt-compressed weight relative to the old code and the polynomial
    # leaves a visible AA-AX phase residual.
    BP = Gustavo.Bandpass
    bl_pairs = [(1, 2)]
    nchan = 2
    npol = 1
    # |v|² differs across channels: 4 at c=1, 1 at c=2. w_raw is uniform.
    V = DimArray(
        reshape(ComplexF64[2.0 + 0.0im, 1.0 + 0.0im], 1, 1, nchan),
        (Baseline(["AB"]), Pol(["PP"]), IF(Float64.(1:nchan)))
    )
    W = DimArray(fill(1.0, 1, 1, nchan), dims(V))

    cw = BP.antenna_phase_weights(V, W, bl_pairs, 2, 1)
    # log_visibility_precision = w·|v|² → 4.0 at c=1, 1.0 at c=2 per baseline,
    # accumulated to both antennas.
    @test cw[1, 1] ≈ 4.0
    @test cw[1, 2] ≈ 1.0
    @test cw[2, 1] ≈ 4.0
    @test cw[2, 2] ≈ 1.0
end

@testset "UVSet partition tree shape" begin
    UV = Gustavo.UVData
    uvset = synthetic_uvdata()
    @test uvset isa UV.UVSet
    @test length(UV.branches(uvset)) == length(UV.metadata(uvset).scans)
    for s in 1:length(UV.metadata(uvset).scans)
        key = UV.partition_key(:TEST, s)
        @test haskey(UV.branches(uvset), key)
        leaf = UV.branches(uvset)[key]
        @test UV.scan_idx(leaf) == s
        @test ndims(parent(leaf[:vis])) == 4
        @test ndims(parent(leaf[:uvw])) == 3
    end
end

@testset "UVSet apply" begin
    UV = Gustavo.UVData
    uvset = synthetic_uvdata()

    # Identity apply preserves shape and globals.
    same = UV.apply((p, _info, _root) -> p, uvset)
    @test same isa UV.UVSet
    @test length(UV.branches(same)) == length(UV.branches(uvset))
    @test UV.metadata(same).antennas === UV.metadata(uvset).antennas
    for (key, _) in UV.branches(uvset)
        @test haskey(UV.branches(same), key)
    end

    # Transformed leaves: scale weights by 2.
    scaled = UV.apply(uvset) do leaf, _info, _root
        new_w = parent(leaf[:weights]) .* 2
        UV.with_visibilities(leaf, parent(leaf[:vis]), new_w)
    end
    for (key, leaf) in UV.branches(scaled)
        orig = UV.branches(uvset)[key]
        @test parent(leaf[:weights]) ≈ 2 .* parent(orig[:weights])
    end
end

@testset "UVSet selection ergonomics" begin
    UV = Gustavo.UVData
    using DimensionalData: At
    uvset = synthetic_uvdata()

    # Single-scan selection returns a leaf DimTree.
    p1 = UV.select_scan(uvset, "TEST", 1)
    @test p1 isa UV.DimTree
    @test p1 === UV.branches(uvset)[UV.partition_key(:TEST, 1)]

    # Multi-scan selection via select_partition returns a sub-UVSet.
    sub_s1 = UV.select_partition(uvset; scan = 1)
    @test sub_s1 isa UV.UVSet
    @test length(UV.branches(sub_s1)) == 1

    # Source filter — single source "TEST" still returns the full set.
    sub_test = UV.select_source(uvset, "TEST")
    @test length(UV.branches(sub_test)) == length(UV.branches(uvset))
    sub_other = UV.select_source(uvset, "NONEXISTENT")
    @test isempty(UV.branches(sub_other))

    # Station filter — synthetic fixture has antennas "AA","AX" so the only
    # baseline AA-AX touches both. Filtering by "AA" keeps it; by "QQ"
    # drops everything.
    sub_aa = UV.select_station(uvset, "AA")
    @test sub_aa isa UV.UVSet
    @test length(UV.branches(sub_aa)) == length(UV.branches(uvset))
    for (_, leaf) in UV.branches(sub_aa)
        for bi in eachindex(UV.baselines(leaf).pairs)
            @test "AA" in (UV.baselines(leaf).ant1_names[bi], UV.baselines(leaf).ant2_names[bi])
        end
    end
    sub_qq = UV.select_station(uvset, "QQ")
    @test isempty(UV.branches(sub_qq))

    # Single-baseline filter.
    sub_bl = UV.select_baseline(uvset, "AA-AX")
    @test sub_bl isa UV.UVSet
    @test length(UV.branches(sub_bl)) == length(UV.branches(uvset))
    for (_, leaf) in UV.branches(sub_bl)
        @test UV.baselines(leaf).labels == ["AA-AX"]
        @test size(parent(leaf[:vis]), 2) == 1   # Baseline axis collapsed to 1
    end

    # Time window — synthetic fixture has obs_time = [0.0, 1.0], one per scan.
    win = UV.time_window(uvset, 0.5, 1.5)
    @test win isa UV.UVSet
    @test length(UV.branches(win)) == 1
    @test haskey(UV.branches(win), UV.partition_key(:TEST, 2))

    # Flag layer is computed at construction (= weights ≤ 0).
    leaf_1 = UV.branches(uvset)[UV.partition_key(:TEST, 1)]
    @test eltype(leaf_1[:flag]) == Bool
    @test parent(leaf_1[:flag]) == (parent(leaf_1[:weights]) .<= 0)
end

@testset "scan_average(uvset) collapses Ti to length 1" begin
    UV = Gustavo.UVData
    uvset = synthetic_uvdata()
    avg_set = UV.scan_average(uvset)

    @test avg_set isa UV.UVSet
    @test length(UV.branches(avg_set)) == length(UV.branches(uvset))
    for (key, leaf) in UV.branches(avg_set)
        @test size(parent(leaf[:vis]), 1) == 1
        @test length(UV.obs_time(leaf)) == 1
    end

    # The reducer-style entry point composes via apply.
    via_apply = UV.apply(UV.TimeAverage(), uvset)
    for (key, leaf) in UV.branches(via_apply)
        @test parent(leaf[:vis]) == parent(UV.branches(avg_set)[key][:vis])
    end
end

@testset "apply_bandpass(uvset) preserves shape per partition" begin
    UV = Gustavo.UVData
    BP = Gustavo.Bandpass
    uvset = synthetic_uvdata()

    nant = length(UV.metadata(uvset).antennas)
    nchan = length(UV.metadata(uvset).array_obs.freq_setup.channel_freqs)
    nscan = length(UV.metadata(uvset).scans)
    gains = ComplexF64[
        cis(0.1 * a + 0.05 * c + 0.02 * f + 0.01 * s) * (0.95 + 0.01 * (a + c))
            for s in 1:nscan, a in 1:nant, f in 1:2, c in 1:nchan
    ]

    corr_set = BP.apply_bandpass(uvset, gains)
    @test corr_set isa UV.UVSet
    @test length(UV.branches(corr_set)) == length(UV.branches(uvset))
    for (key, leaf) in UV.branches(corr_set)
        orig = UV.branches(uvset)[key]
        @test size(parent(leaf[:vis])) == size(parent(orig[:vis]))
    end
end

@testset "Weighted LSQ accepts mixed-precision RHS" begin
    # Regression: Memo-117 cleanup made `weights` Float32 while `A` is built in
    # Float64 by `design_matrices`. LinearSolve's QR `ldiv!` errored on the
    # Float64 factorization against a Float32 RHS — promote to a shared eltype.
    BP = Gustavo.Bandpass
    A = Float64[1.0 0.0; 1.0 1.0; 1.0 2.0; 1.0 3.0]
    b32 = Float32[1.0, 2.0, 3.1, 3.9]
    iv32 = Float32[1.0, 1.0, 1.0, 1.0]
    x = BP.weighted_least_squares(A, b32, iv32)
    @test eltype(x) == Float64
    @test x ≈ A \ Float64.(b32) rtol = 1.0e-6

    # Same for the regularized and constrained variants.
    xr = BP.weighted_regularized_least_squares(A, b32, iv32, [0.0, 0.0])
    @test xr ≈ A \ Float64.(b32) rtol = 1.0e-6
    C = Float64[1.0 0.0]
    d = Float64[0.5]
    xc = BP.weighted_constrained_least_squares(A, b32, iv32, C, d)
    @test isfinite(xc[1]) && isfinite(xc[2])
end

@testset "Bandpass solver works with Float32 weights" begin
    # Regression: `prepare_bandpass_solver` + `initialize_bandpass_state` was
    # erroring inside `weighted_least_squares` because `data.weights` is now
    # Float32 (Memo 117) and the design matrix is Float64. This testset drives
    # the full template-solve path on the synthetic averaged dataset using a
    # Float32 weights cube to lock the regression.
    BP = Gustavo.Bandpass
    UV = Gustavo.UVData
    avg = synthetic_bandpass_avg_uvdata()
    # Force Float32 weights (the production path now produces these on load).
    w32 = Float32.(parent(avg.weights))
    avg = BP.BandpassDataset(
        avg.vis, DimArray(w32, dims(avg.weights)), avg.uvw,
        avg.baselines, avg.antennas, avg.metadata, avg.scans,
    )
    @test eltype(avg.weights) == Float32

    setup = BP.prepare_bandpass_solver(
        avg, 1;
        station_models = BP.build_station_models(avg.antennas.name, Dict{String, BP.StationBandpassModel}()),
        min_baselines = 1,
    )
    state = BP.initialize_bandpass_state(setup, BP.RatioBandpassInitializer())
    @test all(isfinite, state.gains)
end

@testset "DimArray slicing" begin
    using DimensionalData
    UV = Gustavo.UVData
    BP = Gustavo.Bandpass
    raw = synthetic_uvdata()
    avg = synthetic_bandpass_avg_uvdata()

    # Per-leaf Pol slice on a UVSet: pull leaf, then DimTree's selector.
    leaf1 = UV.select_scan(raw, "TEST", 1)
    sliced = leaf1[Pol = At("PP")]
    @test sliced isa DimensionalData.AbstractDimTree

    # BandpassDataset Baseline slice returns a DimStack.
    sel3 = avg[Baseline = At("AA-AX")]
    @test ndims(sel3[:vis]) == 3
    @test parent(sel3[:uvw]) == parent(avg.uvw)

    # Baseline-tuple positional indexing on a BandpassDataset.
    bl_vis = BP.baseline_visibilities(avg, ("AA", "AX"))
    @test ndims(bl_vis) == 3
end

# Build a second single-source UVSet so we can exercise multi-source
# composition via merge_uvsets. The fixture mirrors `synthetic_uvdata`
# but with source name "3C273" and slightly shifted obs_times.
function synthetic_uvdata_3c273()
    base = synthetic_uvdata()
    UV = Gustavo.UVData
    leaves_orig = collect(values(UV.branches(base)))
    new_branches = Gustavo.UVData.DimensionalData.TreeDict()
    for leaf in leaves_orig
        info = UV.metadata(leaf)
        info_3c = merge(
            info,
            (;
                source_name = "3C273",
                source_key = UV.sanitize_source("3C273"),
                field_name = "3C273",
                ra = 187.27791666666667,
                dec = 12.391122222222222,
                partition_name = "synthetic_0_3C273_$(info.scan_name)",
            ),
        )
        new_leaf = UV._build_leaf(
            leaf[:vis], leaf[:weights], leaf[:uvw], leaf[:flag];
            partition_info = info_3c,
        )
        new_key = UV.partition_key(info_3c.source_key, info_3c.scan_idx)
        new_branches[new_key] = new_leaf
    end
    return Gustavo.UVData.DimensionalData.rebuild(base; branches = new_branches)
end

@testset "multi-source UVSet" begin
    UV = Gustavo.UVData
    set_test = synthetic_uvdata()
    set_3c = synthetic_uvdata_3c273()

    multi = UV.merge_uvsets(set_test, set_3c)
    @test multi isa UV.UVSet
    @test length(UV.branches(multi)) == 4
    @test UV.sources(multi) == ["TEST", "3C273"]
    @test sort(UV.scan_ids(multi, "TEST")) == [1, 2]
    @test sort(UV.scan_ids(multi, "3C273")) == [1, 2]
    # summary returns one row per leaf with correct source/scan info.
    rows = Base.summary(multi)
    @test length(rows) == 4
    @test count(r -> r.source_name == "TEST", rows) == 2
    @test count(r -> r.source_name == "3C273", rows) == 2

    # select_source narrows to one source's leaves.
    sub_test = UV.select_source(multi, "TEST")
    @test length(UV.branches(sub_test)) == 2
    @test all(UV.metadata(l).source_name == "TEST" for (_, l) in UV.branches(sub_test))

    # select_scan returns the leaf for (source, scan_idx).
    leaf = UV.select_scan(multi, "3C273", 1)
    @test UV.metadata(leaf).source_name == "3C273"
    @test UV.metadata(leaf).scan_idx == 1

    # Tab-completable Partitions accessor.
    ps = UV.partitions(multi)
    @test :TEST_scan_1 in propertynames(ps)
    @test :M3C273_scan_1 in propertynames(ps)
    @test ps.M3C273_scan_1 === UV.branches(multi)[:M3C273_scan_1]

    # apply preserves tree shape; per-leaf metadata flows through.
    averaged = UV.apply(UV.TimeAverage(), multi)
    @test length(UV.branches(averaged)) == 4
    @test all(size(parent(l[:vis]), 1) == 1 for (_, l) in UV.branches(averaged))

    # write_uvfits errors clearly on multi-source input but works after select_source.
    tmp = tempname() * ".uvfits"
    try
        @test_throws ErrorException UV.write_uvfits(tmp, multi)
        @test UV.write_uvfits(tmp, UV.select_source(multi, "TEST")) == tmp
        @test isfile(tmp)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
end

@testset "Source name sanitization" begin
    UV = Gustavo.UVData
    @test UV.sanitize_source("TEST") == :TEST
    @test UV.sanitize_source("3C273") == :M3C273       # digit-leading
    @test UV.sanitize_source("Sgr A*") == :Sgr_A_      # non-identifier chars
    @test UV.sanitize_source("NGC 4486") == :NGC_4486
    @test UV.sanitize_source("") == :unknown
    @test UV.sanitize_source("  ") == :unknown
    @test UV.partition_key(:M3C273, 5) == :M3C273_scan_5
end
