# ── Fringe search over one materialized scan group ───────────────────────────
#
# The domain half of the streaming pass: given a materialized scan `DimStack`
# and its `DataGeometry`, run the per-baseline kernels in `search.jl` over every
# (baseline, product) cell. The caller (`Streaming`/the pipeline) owns the
# group/budget/transform machinery and supplies the cube and geometry here.

# One recorded fringe detection row: baseline antennas, correlation product,
# SNR, and the PER-BASELINE false-alarm probability (single-search null).
const DetectionRow = @NamedTuple{a::Int, b::Int, pol::String, snr::Float64, pfa::Float64}

"""
    search_scan(data, geom::DataGeometry, params::FringeSearch;
                Vsearch = data[:vis], ngroups = 1,
                executor = SerialScheduler(), t0 = geom.t0 * 3600.0) -> DimStack

Fringe-search every cross-baseline (baseline, product) of a materialized scan
group — the public stage-A search. `data` is the group's `DimStack` as
[`materialize_cube`](@ref) returns it; the search reads data only and needs no
geometry window. Autocorrelation baselines (antenna `a == a`, total power) are
dropped up front, so the result covers only interferometric baselines. `geom`
supplies the reference frequency `f0` and the default phase epoch `t0`.
`Vsearch` lets a caller search a residual cube in place of the raw one
(`rounds > 1`). `ngroups` sets the family-wise Bonferroni denominator:
`params.pfa_max` budgets the whole family of `ncross×npol×ngroups` searches, so
each individual search runs at `pfa_max` divided by that count; the default
`ngroups = 1` is standalone per-scan gating (the QA convention), a whole-track
solve passes its scan count. `t0` (seconds) is the epoch the detection PHASES
are referenced to — delay/rate/SNR are epoch-invariant; the default is `geom`'s
track epoch, a standalone QA caller typically wants the scan midpoint
(`mean(timestamps(data)) * 3600`). Results are bit-identical to the serial loop
regardless of the fan-out `executor`.

Returns a `DimStack` over `Baseline × Pol` whose layers are the six
[`Detection`](@ref) fields (`:delay`/`:rate`/`:phase`/`:amp`/`:snr`/`:valid`),
so one cell `det[bi, p]` reads back as a `Detection` `NamedTuple`, and its
`Baseline` lookup carries the surviving `(a, b)` antenna pairs. The layers'
element type tracks `Vsearch`'s own precision (`real(eltype(Vsearch))`) — a
`ComplexF32` cube produces `Float32` layers. The scan-level aggregates (max
SNR, effective cell count, the detection table) are not stored here; a caller
derives them from the cube's layers plus a cheap `_search_cells` recompute when
it needs them.

Grid geometry (and its shared FFT plan) is built ONCE per scan; the independent
per-(baseline, product) searches fan out over `executor`, each task reusing one
`FringeWorkspace` for scratch. Each search result is written straight into its
cell of the DimStack — no intermediate detection matrix is built.
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
    # `valid` is a dense `Matrix{Bool}`, NOT a `BitArray`: the fan-out writes
    # distinct cells concurrently, and adjacent bits of a BitArray share a word,
    # so `zeros(Bool, …)` is race-free where `falses(…)` is not.
    valid = zeros(Bool, dims...)
    scube = DimensionalData.DimStack((; delay, rate, phase, amp, snr, valid))

    fg = frequencies(data)
    times = timestamps(data) .* 3600.0
    ax = _search_axes(fg, times, params, C)
    # Family-wise PFA (Bonferroni): split pfa_max across all ncross×npol×ngroups
    # searches and resolve the per-search SNR gate once (cheap — no FFT plan).
    nsearch = max(ncross * npol, 1) * max(ngroups, 1)
    snr_gate = _snr_gate(fg, times, params, nsearch)

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
            fg, times, f0, t0_sec, ax, workspace[], params, snr_gate,
        )
    end
    return scube
end
