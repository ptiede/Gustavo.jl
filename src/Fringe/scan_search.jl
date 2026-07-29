# ── Fringe search over one materialized scan group ───────────────────────────
#
# The domain half of the streaming pass: `Streaming` owns the group/budget/
# transform machinery and hands a materialized scan `DimStack` to the kernels in
# `search.jl`. This file is the join — it is the only place `Streaming`'s types
# and the fringe kernels meet.

# One recorded fringe detection row: baseline antennas, correlation product,
# SNR, and the PER-BASELINE false-alarm probability (single-search null).
const DetectionRow = @NamedTuple{a::Int, b::Int, pol::String, snr::Float64, pfa::Float64}

"""
    ScanSearchResult

One scan group's fringe-search outcome: the per-(baseline, product) detection
matrix `det`, the group's `max_snr`, `cells1` (independent search cells of ONE
baseline×product search — the null for a single detection's PFA), `ncells` (the
scan-level effective cells: `cells1 ×` the number of cross-baseline×product
searches — the null for the scan's max SNR), and the valid detections as
`rows` (`(; a, b, pol, snr, pfa)`).
"""
struct ScanSearchResult
    det::Matrix{FringeDetection}
    max_snr::Float64
    cells1::Float64    # effective cells are FRACTIONAL (oversampled grids
    ncells::Float64    # divide by the oversampling) — never Int on real data
    rows::Vector{DetectionRow}
end

"""
    search_scan(stream::ScanStream, stack, search::FringeSearch;
                Vsearch = stack[:vis], ngroups = length(stream.groups),
                executor = stream.inner_executor, t0 = stream.geom.t0 * 3600.0) -> ScanSearchResult

Fringe-search every (baseline, product) of a materialized scan group — the
public stage-A search. `stack` is the group's `DimStack` as
[`materialize_cube`](@ref) returns it; the search reads data only and needs no
geometry window. `Vsearch` lets a caller search a residual cube in place of the
raw one (`rounds > 1`). `ngroups` sets the family-wise Bonferroni denominator:
`search.pfa_max` budgets the whole family of `ncross×npol×ngroups` searches, so
each individual search runs at `pfa_max` divided by that count; pass
`ngroups = 1` for standalone per-scan gating (the QA convention). `t0` (seconds)
is the epoch the detection PHASES are referenced to — delay/rate/SNR are
epoch-invariant; the default is the solve's track epoch, a standalone QA caller
typically wants the scan midpoint (`mean(timestamps(stack)) * 3600`). Results
are bit-identical to the serial loop regardless of the inner `executor`.

`stream` must carry a [`FringeWorkspace`](@ref) pool — build it with
`scan_stream(uvset; workspace = FringeWorkspace)`.
"""
function search_scan(
        stream::ScanStream, stack::AbstractDimStack, search::FringeSearch;
        Vsearch = stack[:vis], ngroups::Integer = length(stream.groups),
        executor = stream.inner_executor, t0::Real = stream.geom.t0 * 3600.0,
    )
    return _search_scan_cube(
        UVData.baselines(stack).pairs, pol_products(stack),
        frequencies(stack), timestamps(stack), stack[:weights],
        Vsearch, stream.geom.f0, Float64(t0), search,
        _search_pool(stream), executor, ngroups,
    )
end

# The stream's workspace pool, or a message naming the fix. The pool's element
# type is set by `scan_stream`'s `workspace` factory, which the streaming layer
# defaults to none — searching through a stream that was built without one
# would otherwise block forever on an empty channel.
function _search_pool(stream::ScanStream)
    pool = stream.pool
    eltype(pool) === FringeWorkspace && return pool
    throw(
        ArgumentError(
            "search_scan needs a FringeWorkspace pool, but this stream's pool holds " *
                "$(eltype(pool)) — build it with " *
                "`scan_stream(uvset; workspace = FringeWorkspace)`."
        )
    )
end

# Core search over one stacked cube. Grid geometry is built ONCE per group; the
# independent per-(baseline, product) searches fan out over the inner `executor`,
# each borrowing a workspace so FFTW plans survive scan-to-scan.
function _search_scan_cube(
        bl_pairs, pols, fg, tg, Wg, Vsearch, f0, t0_sec, search,
        pool::Channel{FringeWorkspace}, executor, ngroups::Integer,
    )
    nbl = length(bl_pairs)
    npol = length(pols)
    det = Matrix{FringeDetection}(undef, nbl, npol)
    times = tg .* 3600.0
    ax = _search_axes(fg, times, search)
    ncross = count(pr -> pr[1] != pr[2], bl_pairs)
    cells1 = _search_cells(ax, search)
    ncells = cells1 * max(ncross * npol, 1)
    # Family-wise PFA gate (Bonferroni): each search runs at pfa_max/nsearches
    # so the acceptance threshold scales itself with array size, product count,
    # scan count, and (through cells1) bandwidth/duration.
    nsearch = max(ncross * npol, 1) * max(ngroups, 1)
    search = isfinite(search.pfa_max) ?
        FringeSearch(search.delay_window, search.rate_window, search.oversample,
                     search.snr_min, search.quad_interp, search.algorithm,
                     search.pfa_max / nsearch) : search

    pairs = [(bi, p) for p in 1:npol for bi in 1:nbl]
    nchunk = clamp(Executors.inner_nchunks(executor), 1, length(pairs))
    chunks = collect(Iterators.partition(pairs, cld(length(pairs), nchunk)))
    tforeach(chunks; scheduler = executor) do chunk
        ws = take!(pool)
        try
            for (bi, p) in chunk
                a, b = bl_pairs[bi]
                if a == b
                    det[bi, p] = FringeDetection(0.0, 0.0, 0.0, 0.0, 0.0, false)
                    continue
                end
                det[bi, p] = _baseline_fringe_search(
                    view(Vsearch, :, :, bi, p), view(Wg, :, :, bi, p),
                    fg, times, f0, t0_sec, ax, ws, search,
                )
            end
        finally
            put!(pool, ws)
        end
    end

    # Assemble the scalar outputs SEQUENTIALLY in the original (product-major)
    # order so the recorded detection table matches the serial loop's exactly.
    maxsnr = 0.0
    rows = DetectionRow[]
    for p in 1:npol, bi in 1:nbl
        d = det[bi, p]
        d.valid || continue
        maxsnr = max(maxsnr, d.snr)
        push!(rows, (; a = bl_pairs[bi][1], b = bl_pairs[bi][2],
            pol = pols[p], snr = d.snr, pfa = fringe_pfa(d.snr, cells1)))
    end
    return ScanSearchResult(det, maxsnr, cells1, ncells, rows)
end
