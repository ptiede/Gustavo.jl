# ── Joint complex bandpass + per-scan source coherence ───────────────────────
#
# `solve_joint_bandpass!` fits the complex visibilities against
#
#     V_ab(ν) | scan  ≈  g_a(ν) · S_{scan,ab,pol} · conj(g_b(ν)),
#
# one frequency-flat complex `S` per (scan, baseline, polarization product), so
# a resolved or polarized source's per-baseline structure is absorbed into `S`
# rather than biasing the station bandpass. `S` given the gains is closed-form
# (`_source_coherence!`), and the gains take Gauss–Newton steps of the MAP
# objective with `S` profiled out (`_map_step`).
#
# The data are `(Scan, AntennaPair, FeedPair, Frequency)` arrays over the
# refinement cells every station's segmentation is a union of. The gains are one
# `(AntennaName, Feed, Frequency, Ti)` array per phase station block, over the block's
# own stations, frequency segments and time segments; a station's block and
# position come from `_block_locations`. Station and feed
# indices come from the `AntennaPair`/`FeedPair` labels (`_cell_nodes`).

# One scan's per-(baseline, pol, segment) coherent residual, written directly
# into `rview`/`wview`, a (AntennaPair, FeedPair, Frequency) slice of the multi-scan
# accumulator, with no intermediate allocation.
#
# Cells are kept at their inverse-variance weights at any SNR: a weak cell is
# unbiased, and the weak cells pairing different feeds are the only rows tying
# one feed's gains to another's. Outliers are flagged upstream of the solve.
function _reduce_scan_segments!(rview, wview, sc, segs)
    for p in axes(rview, FeedPair), bi in axes(rview, AntennaPair)
        for (fs, chans) in enumerate(segs)
            rc, wc = _segment_residual(sc.rl, sc.wl, bi, p, chans)
            keep = isfinite(rc) && isfinite(wc) && wc > 0
            rview[bi, p, fs] = keep ? rc : zero(rc)
            wview[bi, p, fs] = keep ? wc : zero(wc)
        end
    end
    return nothing
end

# Every scan's (AntennaPair, FeedPair, segment) residual, stacked over an added Scan
# axis — not summed across scans, since the per-scan source coherence needs each
# scan's own coherent visibility, in the scan accumulators' element types or
# in the real type `eltype`. The segments are numbered along `Frequency`.
function _reduce_all_scans(scans, segs; eltype = nothing)
    rl = first(scans).rl
    T = something(eltype, real(Base.eltype(first(scans).wl)))
    C = isnothing(eltype) ? Base.eltype(rl) : complex(eltype)
    d = (Scan(eachindex(scans)), dims(rl, AntennaPair), dims(rl, FeedPair), Frequency(eachindex(segs)))
    rseg, wseg = zeros(C, d), zeros(T, d)
    for (r, w, sc) in zip(eachslice(rseg; dims = Scan), eachslice(wseg; dims = Scan), scans)
        _reduce_scan_segments!(r, w, sc, segs)
    end
    return rseg, wseg
end

# Each station's block and its position on that block's `AntennaName` axis; `(0, 0)`
# for a station no block covers.
function _block_locations(blocks, nant)
    loc = fill((0, 0), nant)
    for (k, b) in pairs(blocks), (ai, a) in pairs(b.stations)
        loc[a] = (k, ai)
    end
    return loc
end

# The blocks a joint solve holds its complex gains on: the phase blocks, or the
# amplitude blocks when the model has no phase. Where both exist they match.
_gain_blocks(phase_blocks, amp_blocks) = isempty(phase_blocks) ? amp_blocks : phase_blocks

# One gain block's solved state over `(AntennaName, Feed, Frequency, Ti)` — its
# stations, every feed, its frequency segments and its time segments, both
# numbered: the
# complex gains `g`; their phases `φ` on the branches the solve chose; the slots
# whose phase and whose log-amplitude the solve gave a value (`phase_valid`,
# `amp_valid`), which differ where a prior fills one observable over a slot
# without data and the other has no prior; and the gauge pins (`pinned`).

function _block_gains(block, geom::DataGeometry, C::Type)
    ax = (
        _station_dim(geom.stations[block.stations]), Feed(1:geom.nfeed),
        Frequency(1:block.plan.shape.Frequency), Ti(1:block.plan.shape.Ti),
    )
    return DimStack((;
        g = ones(C, ax), φ = zeros(real(C), ax),
        phase_valid = zeros(Bool, ax), amp_valid = zeros(Bool, ax), pinned = zeros(Bool, ax),
    ))
end

"""
    _station_freq_segments(blocks, nant) -> (fseg, cells)

The frequency cells a joint solve reduces its data onto — the coarsest
channel partition every block's own segmentation is a union of cells of — and
`fseg[a, k]` is the segment station `a` carries over cell `k`, `0` for a station
no block covers.

Stations that share one frequency segmentation give the identity refinement, so
`cells` is that segmentation's own channel groups and every station's cell maps
to itself.
"""
function _station_freq_segments(blocks, nant)
    cell_ids, ncell = common_refinement([b.plan.fseg_id for b in blocks])
    cells = segment_groups(cell_ids, ncell)
    fseg = zeros(Int, Base.OneTo(nant), Base.OneTo(ncell))
    for b in blocks, a in b.stations
        for (k, chans) in pairs(cells)
            # A cell lies wholly inside one segment of every block's
            # segmentation, so its first channel names them all.
            fseg[a, k] = b.plan.fseg_id[first(chans)]
        end
    end
    return fseg, cells
end

# Dense ids of the gain slots, one array per block in its gain array's shape,
# numbered consecutively across the blocks.
function _gain_slot_ids(gains)
    lasts = cumsum(map(st -> length(st.g), gains))
    return map(gains, lasts) do st, l
        reshape((l - length(st.g) + 1):l, size(st.g))
    end
end

# The phase-gauge graph of a joint bandpass solve: one node per gain slot —
# (station, feed, that station's own frequency segment, its own time segment) —
# one edge per (scan, station pair, feed pair, refinement cell) correlation,
# joining its two ends in the segments they are in for that scan and that cell.
# `nodew[n]` sums the data weights of the correlations touching node `n`, the
# score the gauge falls back to when choosing an anchor.
#
# The node set is the gauge condition. An edge names the two gain slots one
# correlation reads, and a shared phase cancels out of `g_a·S·conj(g_b)` only
# when both ends of every internal edge carry it, so the unobservable phases are
# one constant per connected component. No coarser node set states that, and a
# station whose frequency segments are finer than the modes the array leaves free
# would be over-constrained by any pin that fixed its whole track.
#
# `ids` numbers the slots (`_gain_slot_ids`); `compid[n]` is `0` for a slot no
# correlation touches.
function _joint_bandpass_graph(data, layout, ids)
    (; w, ends) = data
    (; loc, tseg, fseg) = layout
    nslots = sum(length, ids; init = 0)
    edges = Tuple{Int, Int}[]
    nodew = zeros(Base.eltype(w), nslots)
    for si in axes(tseg, 2), p in axes(ends, FeedPair), bi in axes(ends, AntennaPair)
        _solvable(ends[bi, p]) || continue
        (a, fa), (b, fb) = ends[bi, p]
        # A station no block covers (segment 0) carries no bandpass parameter, so
        # the correlations touching it constrain nothing.
        ta, tb = tseg[a, si], tseg[b, si]
        (iszero(ta) || iszero(tb)) && continue
        (ka, ia), (kb, ib) = loc[a], loc[b]
        for cell in axes(fseg, 2)
            na = ids[ka][ia, fa, fseg[a, cell], ta]
            nb = ids[kb][ib, fb, fseg[b, cell], tb]
            push!(edges, (na, nb))
            nodew[na] += w[si, bi, p, cell]
            nodew[nb] += w[si, bi, p, cell]
        end
    end
    compid, ncomp, _ = connected_components(nslots, edges)
    return compid, ncomp, nodew
end

"""
    _joint_bandpass_gauge!(gains, data, layout, blocks, gauge) -> Union{Nothing, NamedTuple}

Impose `gauge` on the phase of a joint bandpass solve: one
[`GaugeFreedom`](@ref) per connected component of the joint-bandpass graph,
and the constraints `C·φ = d` that [`gauge_constraints`](@ref) gives over
every gain slot's phase `φ`, numbered by `_gain_slot_ids`. When every row
holds a single slot at zero, as `PinAntenna`'s do, those slots are marked in
the `pinned` layer of `gains`, held at zero phase, and the result is
`nothing`. Any other gauge is imposed exactly in every Gauss–Newton step
(`_map_step`), and this returns the constraints as `(; rows, d)`,
`rows[j]` the `(slot, coefficient)` pairs of row `j`. Either way a constrained
slot's amplitude is solved like any other slot's.

Only the phase is a gauge freedom. Multiplying the gains of a set of nodes by a
shared `c` sends `g_a·S·conj(g_b)` to `|c|²·g_a·S·conj(g_b)` on every
correlation internal to that set: the phase of `c` cancels between the two
conjugated factors. The sets on which it cancels everywhere are exactly the
components above, so the phase carries one unobservable constant per component
and needs one constraint there — no more. A station-uniform segmentation splits
the graph along the array-wide (frequency segment, time segment) cells, so the
free direction is a phase spectrum common to every station, and a pin per cell
zeroes the reference station's whole track; a model in which one station
breaks mid-track keeps the whole track in one component, because the stations
that hold one gain over it bridge the broken station's two segments, and the
relative phase across that break is then measured rather than gauged away.

The per-station priors along frequency are not invariant to that common
spectrum, so the gauge changes the solution and is imposed during the fit
rather than by shifting a fitted result.

The same bridging on the frequency axis makes a pin PARTIAL: where a station
holds one gain across cells the others split, those cells lie in one component,
and its single pin fixes ONE segment of the pinned station's track while the
rest of that track is fitted.

Besides that common phase, each (station, feed, time segment) track has a free
band-constant phase and log-amplitude: `S` is per scan and baseline but
frequency-flat, so it absorbs a band-constant factor on any one station, and
a prior is profiled over that constant (see the header of bandpass_priors.jl). Pinning `|g|` as well
would assert the reference antenna has a flat amplitude bandpass, discarding
structure that is identifiable (mean-removing `log|V_ab| = la_a + la_b + ls_ab`
over the band eliminates `ls` and leaves a signless-Laplacian system, full rank
wherever each connected group of baselines has an odd loop; see
`_split_free_amplitudes!` for a group without one) and biasing every other
station through the inconsistency.
"""
function _joint_bandpass_gauge!(gains, data, layout, blocks, gauge)
    T = real(eltype(first(gains).g))
    ids = _gain_slot_ids(gains)
    compid, ncomp, nodew = _joint_bandpass_graph(data, layout, ids)
    slot = [(k, I) for (k, idk) in pairs(ids) for I in CartesianIndices(idk)]
    freedoms = map(1:ncomp) do c
        cn = findall(==(c), compid)
        GaugeFreedom(;
            nodes = cn, station = [blocks[slot[n][1]].stations[slot[n][2][1]] for n in cn],
            feed = [slot[n][2][2] for n in cn], scan = zeros(Int, length(cn)),
            component = [blocks[slot[n][1]].plan.path for n in cn], observable = fill(:phase, length(cn)),
            direction = ones(T, length(cn)), weight = nodew[cn],
        )
    end
    C, d = _gauge_system(gauge, GaugeFreedoms{T}(freedoms, length(compid)))
    Cs = sparse(C)
    rows = [Tuple{Int, T}[] for _ in axes(Cs, 1)]
    for n in axes(Cs, 2), i in nzrange(Cs, n)
        iszero(nonzeros(Cs)[i]) || push!(rows[rowvals(Cs)[i]], (n, nonzeros(Cs)[i]))
    end
    if all(j -> length(rows[j]) == 1 && iszero(d[j]), eachindex(rows, d))
        for row in rows
            k, I = slot[first(only(row))]
            gains[k].pinned[I] = true
        end
        return nothing
    end
    return (; rows, d)
end

# The source coherence that minimizes the χ² for the given slot gains `g`: for
# each (scan, station pair, feed pair) the weighted least-squares solution of
# `Σ_cell w·|r/w − g_a·S·conj(g_b)|²` over the single complex `S`.
function _source_coherence!(S, g, data, jl)
    (; r, w) = data
    (; edge, slot_pairs) = jl
    num = zeros(ComplexF64, size(S))
    den = zeros(size(S))
    for I in CartesianIndices(edge)
        e = edge[I]
        iszero(e) && continue
        wc = w[I]
        wc > 0 || continue
        na, nb = slot_pairs[e]
        u = g[na] * conj(g[nb])
        J = CartesianIndex(I[1], I[2], I[3])
        num[J] += conj(u) * r[I]
        den[J] += wc * abs2(u)
    end
    for J in eachindex(S, num, den)
        S[J] = den[J] > 0 ? num[J] / den[J] : zero(eltype(S))
    end
    return S
end

# The χ² of the visibilities against `g_a·S·conj(g_b)`.
function _chi2(g, S, data, jl)
    (; r, w) = data
    (; edge, slot_pairs) = jl
    out = 0.0
    for I in CartesianIndices(edge)
        e = edge[I]
        iszero(e) && continue
        wc = w[I]
        wc > 0 || continue
        na, nb = slot_pairs[e]
        out += abs2(r[I] - wc * g[na] * S[I[1], I[2], I[3]] * conj(g[nb])) / wc
    end
    return out
end

# The χ² to second order in each slot's log-gain about `g`. A correlation's model
# `m = g_a·S·conj(g_b)` moves to `m·(1 + δℓ_a + δℓ_b + i(δφ_a − δφ_b))`, so with
# `u = r/(w·m) − 1` and weight `q = w·|m|²`, `Re u` measures `δℓ_a + δℓ_b` and
# `Im u` measures `δφ_a − δφ_b`, and the two observables decouple. Returns each
# correlation pair's weight `q`, each slot's `c = Σ q·u` (with `conj(u)` at a
# second end), whose real and imaginary parts are the log-amplitude and phase
# gradients, and its total weight `W`: `1 + c/W` is the slot's own gain
# estimate given every other gain and `S`, relative to its current one.
function _linearize(g, S, data, jl, nslots)
    (; r, w) = data
    (; edge, slot_pairs) = jl
    q = zeros(length(slot_pairs))
    c = zeros(ComplexF64, nslots)
    W = zeros(nslots)
    for I in CartesianIndices(edge)
        e = edge[I]
        iszero(e) && continue
        wc = w[I]
        wc > 0 || continue
        na, nb = slot_pairs[e]
        m = g[na] * S[I[1], I[2], I[3]] * conj(g[nb])
        qc = wc * abs2(m)
        qc > 0 || continue
        u = r[I] / (wc * m) - 1
        q[e] += qc
        c[na] += qc * u
        c[nb] += qc * conj(u)
        W[na] += qc
        W[nb] += qc
    end
    return (; q, c, W)
end

# The pattern of the χ²'s curvature in one observable's slot values, over the
# slot pairs each correlation `e` joins: a matrix holding it and, per `e`, the
# indices among its stored values of `(a, a)`, `(b, b)`, `(a, b)` and `(b, a)`.
function _curvature_pattern(slot_pairs, nslots)
    Is = [n for (na, nb) in slot_pairs for n in (na, nb, na, nb)]
    Js = [n for (na, nb) in slot_pairs for n in (na, nb, nb, na)]
    H = sparse(Is, Js, ones(length(Is)), nslots, nslots)
    pos = [ntuple(r -> _position(H, Is[4(e - 1) + r], Js[4(e - 1) + r]), 4) for e in eachindex(slot_pairs)]
    return H, pos
end

# Writes into `H` (`_curvature_pattern`) the curvature of the χ² in one
# observable's slot values, `Σ_e q_e·(x_a + s·x_b)²`, with `s = +1` for
# log-amplitude sums and `−1` for phase differences.
function _data_curvature!(H, pos, q, s)
    Hv = fill!(nonzeros(H), 0.0)
    for (e, p) in pairs(pos)
        qe = q[e]
        qe > 0 || continue
        Hv[p[1]] += qe
        Hv[p[2]] += qe
        Hv[p[3]] += s * qe
        Hv[p[4]] += s * qe
    end
    return H
end
