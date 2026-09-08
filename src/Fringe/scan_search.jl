# ── Fringe search over one materialized scan group ───────────────────────────
#
# The domain half of the streaming pass: given a materialized scan `DimStack`
# and its `DataGeometry`, run the per-baseline kernels in `search.jl` over every
# (baseline, product) cell. The caller (`Streaming`/the pipeline) owns the
# group/budget/transform machinery and supplies the cube and geometry here.

# One recorded search row: baseline antennas, correlation product, SNR, the
# family-wise false-alarm probability, and whether that PFA accepted it as a real
# fringe. Every measured cell gets a row, so `detected` — not the row's presence
# — is what marks a detection.
#
# `phase` (rad) is the measured constant phase at the epoch the search referenced
# (`scan_phase_epoch`). It is the only stage-A observable a station solve cannot
# be inverted for: the parallel-hand pair of one baseline gives the inter-feed
# offset DIFFERENCE `ρ_a − ρ_b` directly as `QQ − PP`, with the source and
# atmospheric terms cancelling, at parallel-hand SNR and without cross-hand data
# or any station fit in between.
const DetectionRow = @NamedTuple{
    a::Int, b::Int, pol::String, snr::Float64, pfa::Float64,
    delay::Float64, rate::Float64, phase::Float64, detected::Bool,
    snr_steer::Float64, pfa_steer::Float64,
    delay_steer::Float64, rate_steer::Float64, steered::Bool,
}

"""
    search_scan(data, geom::DataGeometry, params::FringeSearch;
                Vsearch = data[:vis], ngroups = 1,
                executor = SerialScheduler(), t0 = geom.t0 * 3600.0) -> DimStack

Fringe-search every cross-baseline (baseline, product) of a materialized
scan group. `data` is the group's `DimStack` as [`materialize_cube`](@ref)
returns it; autocorrelation baselines are dropped, so the result covers only
interferometric baselines. `geom` supplies the reference frequency `f0` and
the default phase epoch. `Vsearch` lets a caller search a residual cube in
place of the raw one.

`ngroups` sizes the false-alarm family each cell's `pfa` is computed over:
the family of `ncross×npol×ngroups` searches shares one budget (Bonferroni),
so a recorded `pfa` is directly comparable to `Stationization.pfa_max`. The
default `ngroups = 1` scopes the family to this scan; a whole-track solve
passes its scan count.

`t0` (seconds) is the epoch the detection phases are referenced to
(delay/rate/SNR are epoch-invariant). Quoting a phase a lever arm from the
data costs it `2π·σ_rate·Δt`, so a caller comparing phases against a model
must reference them where the model's constant lives
([`scan_phase_epoch`](@ref)); a standalone caller wants the scan midpoint
(`mean(timestamps(data)) * 3600`). The default is `geom`'s track epoch,
which is right only for a single-scan geometry.

Returns a `DimStack` over `Baseline × Pol` whose layers are the seven
[`Detection`](@ref) fields (`:delay`/`:rate`/`:phase`/`:amp`/`:snr`/`:pfa`/`:valid`),
so one cell `det[bi, p]` reads back as a `Detection` `NamedTuple`; the
`Baseline` lookup carries the surviving `(a, b)` pairs. The layers' element
type tracks `real(eltype(Vsearch))`. Results are bit-identical to the serial
loop regardless of the fan-out `executor`.
"""
function search_scan(
        data::AbstractDimStack, geom::DataGeometry, params::FringeSearch;
        Vsearch = data[:vis], ngroups::Integer = 1,
        executor = SerialScheduler(), t0::Real = geom.t0 * 3600.0,
    )
    # Search only interferometric baselines: drop autocorrelations once here
    # rather than guarding `a == b` per cell. `keep[j]` is the column of the
    # surviving baseline `j` in `data` — the two sets diverge exactly when `data`
    # still carries autocorrelations (they are dropped at load by default).
    bls = UVData.baselines(data)
    keep = findall(pr -> pr[1] != pr[2], bls.pairs)
    bl_pairs = bls.pairs[keep]
    pols = pol_products(data)
    ncross = length(keep)
    npol = length(pols)

    # The search's compute type: a ComplexF32 streaming caller runs its FFT
    # natively in Float32 (see search.jl's precision-scope note); the detection
    # layers below track the same precision.
    C = eltype(Vsearch)
    T = real(C)
    dims = (Baseline(bl_pairs), Pol(pols))
    delay = zeros(T, dims...)
    rate = similar(delay)
    phase = similar(delay)
    amp = similar(delay)
    snr = similar(delay)
    # `valid` is a dense `Matrix{Bool}`, not a `BitArray`: the fan-out writes
    # distinct cells concurrently, and adjacent bits of a BitArray share a word,
    # so `zeros(Bool, …)` is race-free where `falses(…)` is not.
    valid = zeros(Bool, dims...)
    pfa = similar(delay)
    scube = DimensionalData.DimStack((; delay, rate, phase, amp, snr, pfa, valid))

    fg = frequencies(data)
    times = timestamps(data) .* 3600.0
    ax = _search_axes(fg, times, params, C)
    # The false-alarm family: every ncross×npol×ngroups search sharing one budget.
    # Scaling one search's cell count by the family size is the Bonferroni
    # correction, so each cell's recorded `pfa` is a family-wise probability and is
    # directly comparable to `Stationization.pfa_max`.
    nsearch = max(ncross * npol, 1) * max(ngroups, 1)
    family_cells = _search_cells(fg, times, params) * nsearch

    f0 = geom.f0
    t0_sec = Float64(t0)
    Wg = data[:weights]
    # One reusable workspace per task (not per cell): the grids are tens of MB, so
    # a task processing many baselines allocates its scratch once. Each (j, p)
    # writes a distinct cell of every layer, so the concurrent writes never overlap.
    workspace = TaskLocalValue{FringeWorkspace{C}}(() -> FringeWorkspace(C))
    cells = [(j, p) for p in 1:npol for j in 1:ncross]
    tforeach(cells; scheduler = executor) do (j, p)
        bi = keep[j]
        scube[j, p] = _baseline_fringe_search(
            view(Vsearch, :, :, bi, p), view(Wg, :, :, bi, p),
            fg, times, f0, t0_sec, ax, workspace[], params, family_cells,
        )
    end
    _warn_edge_peaks(scube, params, ax)
    return scube
end
