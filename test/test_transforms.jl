# ── Transform chain semantics ────────────────────────────────────────────────
#
# The materialization transform chain (ApplySolution → StationWeightScale →
# FlagChannels + the `CalFunction` open hook): the cube and leaf choke paths
# must agree bit-for-bit through the same chain, user data is never mutated,
# and the view metadata/mutation semantics hold. (The frozen-monolith precal
# oracle these were originally gated against was deleted at M5 after the parity
# gates passed — see test_smoother_step.jl for the end-to-end gates.)

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")

@testset "Transform chain semantics" begin
    uvset, _ = _build_fringe_uvset()
    geom = CAL.build_geometry(uvset)

    # A real solution to divide out (per-scan delays/rates + per-AP adhoc — the
    # time-VARYING precal branch; `_precal_time_constant` guards the fast path).
    precal = fit(
        FringeFit(model = FringeModel()) |> Bandpass() |>
            TemporalSmoother(FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)),
        uvset,
    )
    nant = length(UVP.union_antennas(uvset).name)
    ws = [1.0 + 0.25 * i for i in 1:nant]
    nchan = length(geom.channel_freqs)
    mask = falses(nchan); mask[1] = true; mask[end] = true

    chain = [FP.ApplySolution(precal), FP.StationWeightScale(ws), FP.FlagChannels(mask)]
    st = FP.scan_stream(uvset; geom = geom, transforms = chain)

    @testset "cube path ≡ leaf path through the same chain" begin
        for spec in st.groups
            stack, win = FP.materialize_cube(st, spec; executor = DynamicScheduler(; nchunks = 2))
            keyed = FP.materialize_leaves(st, spec; executor = DynamicScheduler(; nchunks = 2))
            # Rebuild the cube from the transformed leaves: identical data —
            # the two choke paths apply the chain identically.
            stack_l, _ = ST._stacked_scan_group([m for (_, m) in keyed], geom)
            @test isequal(stack[:vis], stack_l[:vis])
            @test isequal(stack[:weights], stack_l[:weights])
            # The mask really zero-weighted the flagged global channels.
            for (c, gc) in enumerate(win.chan_idx)
                mask[gc] && @test all(iszero, @view stack[:weights][c, :, :, :])
            end
        end
    end

    @testset "eager source data is never mutated" begin
        snap = [copy(parent(leaf[:weights])) for (_, leaf) in UVP.branches(uvset)]
        FP.materialize_leaves(st, st.groups[1]; executor = DynamicScheduler(; nchunks = 2))
        FP.materialize_cube(st, st.groups[1])
        @test all(
            isequal(s, parent(leaf[:weights]))
                for (s, (_, leaf)) in zip(snap, UVP.branches(uvset))
        )
    end

    @testset "CalFunction: stack metadata + targeted mutation" begin
        seen = []
        probe = FP.CalFunction() do stack, win
            push!(
                seen, (;
                    source = source_name(stack), scan = scan_name(stack),
                    nchan = length(frequencies(stack)),
                    ant_names = String.(antennas(stack).name),
                    ci = copy(win.chan_idx), ti = copy(win.ti_idx),
                    pol_products = pol_products(stack),
                )
            )
        end
        stp = FP.scan_stream(uvset; geom = geom, transforms = (probe,))
        stack, win = FP.materialize_cube(stp, stp.groups[1])
        @test length(seen) == 1
        m = seen[1]
        @test m.source == stp.groups[1].source && m.scan == stp.groups[1].scan
        @test m.nchan == length(frequencies(stack)) &&
            m.ci == win.chan_idx && m.ti == win.ti_idx
        @test m.ant_names == stp.ant_names
        @test m.pol_products == pol_products(stack)

        # Targeted per-datum correction: halve the weights of baseline (1, 2).
        fix = FP.CalFunction() do stack, win
            for (bi, (a, b)) in enumerate(baselines(stack).pairs)
                minmax(a, b) == (1, 2) && (stack[:weights][Baseline = bi] .*= 0.5)
            end
        end
        stf = FP.scan_stream(uvset; geom = geom, transforms = (fix,))
        stack_f, _ = FP.materialize_cube(stf, stf.groups[1])
        stack_0, _ = FP.materialize_cube(FP.scan_stream(uvset; geom = geom), stf.groups[1])
        W0 = stack_0[:weights]
        for (bi, (a, b)) in enumerate(baselines(stack_0).pairs)
            expect = minmax(a, b) == (1, 2) ? Float32(0.5) .* W0[:, :, bi, :] : W0[:, :, bi, :]
            @test isequal(stack_f[:weights][:, :, bi, :], expect)
        end
        @test isequal(stack_f[:vis], stack_0[:vis])

        # The chain hits the leaf path too.
        keyed_f = FP.materialize_leaves(stf, stf.groups[1])
        @test length(keyed_f) == length(stf.groups[1].leaves)
    end

    @testset "materialized group = leaf-shaped DimStack + geometry window" begin
        stv = FP.scan_stream(uvset; geom = geom)
        stack, win = FP.materialize_cube(stv, stv.groups[1])
        @test stack isa DimStack
        @test win isa CAL.GeometryWindow && win.geom === geom
        # The layers ALIAS the cubes the decode wrote — mutation flows both ways.
        parent(stack[:weights])[1, 1, 1, 1] = 42
        @test stack[:weights][1, 1, 1, 1] == 42.0f0
        # Leaf-format axes: lookups carry freqs/times/products; UVData helpers
        # and DimensionalData selectors work on the stack directly.
        @test collect(lookup(stack[:vis], Frequency)) == frequencies(stack)
        @test collect(lookup(stack[:vis], Ti)) == timestamps(stack)
        @test pol_products(stack[:vis]) == pol_products(stack)
        p1 = pol_products(stack)[1]
        @test parent(stack[:vis][Pol = At(p1)]) == parent(stack[:vis])[:, :, :, 1]
        # The window addresses the same channels in the solve's index space.
        @test geom.channel_freqs[win.chan_idx] ≈ frequencies(stack)
        # The metadata is the first band leaf's own PartitionInfo — moved, not
        # rebuilt: real BaselineIndex, antenna table, source/scan identity.
        md = DimensionalData.metadata(stack)
        @test md isa UVP.PartitionInfo
        @test md.baselines isa UVP.BaselineIndex
        @test md.baselines.pairs == baselines(stack).pairs
        @test md.baselines[baselines(stack).pairs[2]] == 2
        @test md.source_name == source_name(stack)
        @test md.scan_name == scan_name(stack)
        @test String.(md.antennas.name) == stv.ant_names
    end

    @testset "leaf path: stack selected off the leaf, flag re-derived" begin
        lk = ReentrantLock()
        captured = []
        probe = FP.CalFunction() do stack, win
            lock(lk) do
                push!(
                    captured, (;
                        stack, source = source_name(stack),
                        ant_names = String.(antennas(stack).name),
                    )
                )
            end
        end
        stp = FP.scan_stream(uvset; geom = geom, transforms = (probe, FP.FlagChannels(mask)))
        keyed = FP.materialize_leaves(stp, stp.groups[1])
        @test length(captured) == length(keyed)
        # The stack is `leaf[(:vis, :weights)]` — the leaf's own
        # PartitionInfo comes along with the selection, nothing is rebuilt.
        for c in captured
            @test DimensionalData.metadata(c.stack) isa UVP.PartitionInfo
            @test c.ant_names == stp.ant_names
            @test c.source == stp.groups[1].source
        end
        # The transformed arrays flow through to the returned leaves unchanged.
        @test Set(objectid(parent(c.stack[:vis])) for c in captured) ==
            Set(objectid(parent(l[:vis])) for (_, l) in keyed)
        # FlagChannels zeroed whole channels above, so the transformed weights
        # carry the flag (a cell is flagged iff its weight is ≤ 0).
        for (_, l) in keyed
            @test any(parent(l[:weights]) .<= 0)
        end
    end

    @testset "validation errors" begin
        # Station identity is what stream construction checks: a name is the
        # whole of it, so a solution that records none, or none this set shares,
        # can never correct anything and is refused before any data is read.
        anon = CAL.CalibrationSolution(
            precal.steps, precal.geom, Base.structdiff(precal.info, (; ant_names = nothing)),
        )
        @test_throws "records no station names" FP.scan_stream(
            uvset; transforms = [FP.ApplySolution(anon)],
        )
        strangers = CAL.CalibrationSolution(
            precal.steps, precal.geom, merge(precal.info, (; ant_names = ["XX", "YY", "ZZ", "WW"])),
        )
        @test_throws "shares no station with this set" FP.scan_stream(
            uvset; transforms = [FP.ApplySolution(strangers)],
        )

        # Placement is checked per window instead, as each is materialized:
        # `precal`'s bandpass cuts the channel-INDEX axis, so it means nothing
        # on a set that indexes different channels.
        other, _ = _build_fringe_uvset(nspw = 3, nchan = 4)
        sto = FP.scan_stream(other; transforms = [FP.ApplySolution(precal)])
        @test_throws "identical channel layout" FP.materialize_cube(sto, sto.groups[1])
        # A scan the solve never saw is refused rather than served by the
        # neighbouring scan the solution does have.
        twoscan, _ = _build_fringe_uvset(nscans = 2)
        st2 = FP.scan_stream(twoscan; transforms = [FP.ApplySolution(precal)])
        @test_throws "is not in the solution" [
            FP.materialize_cube(st2, g) for g in st2.groups
        ]
        @test_throws ErrorException FP.StationWeightScale([1.0, -1.0])
        st_len = FP.scan_stream(uvset; geom = geom, transforms = [FP.FlagChannels(falses(3))])
        @test_throws ErrorException FP.materialize_cube(st_len, st_len.groups[1])
    end
end

# ── AprioriPreCal: a-priori SEFD scaling inside the streaming pass ────────────
#
# The transform's whole reason to exist is that the solvers see the SCALED data,
# where the `AprioriAmplitude` pipeline step scales only the output. Its
# correctness gate is therefore parity with the eager whole-set verb: streaming
# a scan group through the transform must land on the same visibilities and
# weights `apply_calibration(uvset, antab)` produces for those samples.

@testset "AprioriPreCal" begin
    uvset, _ = _build_fringe_uvset(nant = 4, nspw = 2, nchan = 8, ntime = 6)
    ant_names = String.(UVP.union_antennas(uvset).name)
    geom = CAL.build_geometry(uvset)

    rdate = DimensionalData.metadata(uvset).array_obs.rdate
    base_dt = DateTime(Date(rdate))
    ts = sort!(unique(reduce(vcat, [collect(UVP.obs_time(l)) for l in values(UVP.branches(uvset))])))
    times = [base_dt + Millisecond(round(Int, t * 3_600_000)) for t in ts]
    times = [times[1] - Hour(1); times; times[end] + Hour(1)]      # pad the window

    # Per-station Tsys, distinct per station so a mixed-up station mapping shows
    # up as a wrong factor rather than cancelling.
    tsys_of = Dict(nm => 100.0 * (i + 1) for (i, nm) in pairs(ant_names))
    function _antab(names; tsys_of = tsys_of)
        stns = Dict{String, UVP.AntabStation}()
        for nm in names
            vals = repeat([tsys_of[nm] tsys_of[nm]], length(times), 1)
            stns[nm] = UVP.AntabStation(
                nm, UVP.AntabGainCurve((1.0, 1.0), [1.0]),
                UVP.AntabTsysSeries(times, [(0, :R), (0, :L)], vals), 0,
            )
        end
        return UVP.AntabCalibration("synthetic", "synth", 2000, stns)
    end
    antab = _antab(ant_names)

    # Synthetic station_xyz are not real ECEF coords, so elevation is ill-defined;
    # the gain curve is flat, so disabling the horizon cut isolates SEFD scaling.
    precal = ST.AprioriPreCal(uvset, antab; min_elevation_deg = -Inf)
    eager = UVP.apply_calibration(uvset, antab; min_elevation_deg = -Inf)

    @testset "streaming ≡ eager apply_calibration" begin
        st_raw = FP.scan_stream(uvset; geom = geom)
        st_pre = FP.scan_stream(uvset; geom = geom, transforms = [precal])
        st_eag = FP.scan_stream(eager; geom = geom)
        for (g_pre, g_eag) in zip(st_pre.groups, st_eag.groups)
            s_pre, _ = FP.materialize_cube(st_pre, g_pre)
            s_eag, _ = FP.materialize_cube(st_eag, g_eag)
            @test isequal(parent(s_pre[:vis]), parent(s_eag[:vis]))
            @test isequal(parent(s_pre[:weights]), parent(s_eag[:weights]))
        end
        # ... and it actually did something: the scaled amplitudes differ from raw.
        s_raw, _ = FP.materialize_cube(st_raw, st_raw.groups[1])
        s_pre, _ = FP.materialize_cube(st_pre, st_pre.groups[1])
        @test !isequal(parent(s_raw[:vis]), parent(s_pre[:vis]))
    end

    @testset "eager form matches the verb" begin
        direct = FP.apply_transform(uvset, precal)
        for (k, leaf) in UVP.branches(direct)
            @test isequal(parent(leaf[:vis]), parent(UVP.branches(eager)[k][:vis]))
        end
    end

    # An ANTAB numbers channels WITHIN a spw (ALMA's `'L1|R1' … 'Ln|Rn'`), so a
    # scan group spanning two spws must map each of its columns back to a
    # per-spw channel number. Getting that wrong reads the neighbouring band's
    # Tsys — a wrong answer with no symptom, which is what these two gate.
    nchan_spw = 8
    perchan = let stns = Dict{String, UVP.AntabStation}()
        cols = [(c, :both) for c in 1:nchan_spw]
        for (i, nm) in pairs(ant_names)
            # Strongly channel-dependent, and distinct per station.
            row = [100.0 * (i + 1) * (1 + 0.5 * c) for c in 1:nchan_spw]
            stns[nm] = UVP.AntabStation(
                nm, UVP.AntabGainCurve((1.0, 1.0), [1.0]),
                UVP.AntabTsysSeries(times, cols, repeat(row', length(times), 1)), nchan_spw,
            )
        end
        UVP.AntabCalibration("synthetic", "synth", 2000, stns)
    end
    pre_pc = ST.AprioriPreCal(uvset, perchan; min_elevation_deg = -Inf)

    @testset "per-channel Tsys: streaming ≡ eager across spws" begin
        eager_pc = UVP.apply_calibration(uvset, perchan; min_elevation_deg = -Inf)
        st_pre = FP.scan_stream(uvset; geom = geom, transforms = [pre_pc])
        st_eag = FP.scan_stream(eager_pc; geom = geom)
        for (g_pre, g_eag) in zip(st_pre.groups, st_eag.groups)
            s_pre, _ = FP.materialize_cube(st_pre, g_pre)
            s_eag, _ = FP.materialize_cube(st_eag, g_eag)
            @test isequal(parent(s_pre[:vis]), parent(s_eag[:vis]))
        end
    end

    @testset "the solve sees scaled data" begin
        # The point of the transform: a step fit through it is fit on calibrated
        # amplitudes. A channel-dependent SEFD is what shows it — a station's
        # flat scaling is degenerate with the bandpass gauge and absorbed by it.
        b_raw = fit(Bandpass(), uvset)
        b_pre = fit(pre_pc |> Bandpass(), uvset)
        @test !isapprox(
            abs.(CAL.gains(b_raw; Ti = 1)), abs.(CAL.gains(b_pre; Ti = 1)); rtol = 1.0e-3,
        )
        # The transform is recorded on the solution, so `calibrate` replays it.
        @test any(t -> t isa ST.AprioriPreCal, b_pre.transforms)
    end

    @testset "station coverage is settled at stream construction" begin
        partial = _antab(ant_names[1:(end - 1)])
        @test_throws "has no record for stations" FP.scan_stream(
            uvset; geom = geom,
            transforms = [ST.AprioriPreCal(uvset, partial; on_missing_station = :error)],
        )
        strangers = _antab(["XX", "YY"]; tsys_of = Dict("XX" => 100.0, "YY" => 100.0))
        @test_throws "shares no station with this set" FP.scan_stream(
            uvset; geom = geom, transforms = [ST.AprioriPreCal(uvset, strangers)],
        )
        @test_throws "on_missing_station must be" ST.AprioriPreCal(
            uvset, antab; on_missing_station = :shrug,
        )
    end
end
