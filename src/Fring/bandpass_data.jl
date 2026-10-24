# ── Joint complex bandpass + per-scan source coherence ───────────────────────
#
# `solve_joint_bandpass!` fits the complex visibilities against
#
#     V_ab(ν) | scan  ≈  g_a(ν) · S_{scan,ab,pol} · conj(g_b(ν)),
#
# one frequency-flat complex `S` per (scan, baseline, polarization product), so
# a resolved or polarized source's per-baseline structure is absorbed into `S`
# rather than biasing the station bandpass. `S` given the gains is closed-form,
# and the gains maximize the posterior with `S` profiled out (bandpass_map.jl).
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
    for (k, b) in pairs(blocks), (ai, a) in pairs(b.antennas)
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
        _antenna_dim(geom.antennas[block.antennas]), Feed(1:geom.nfeed),
        Frequency(1:block.plan.shape.Frequency), Ti(1:block.plan.shape.Ti),
    )
    return DimStack((;
        g = ones(C, ax), φ = zeros(real(C), ax),
        phase_valid = zeros(Bool, ax), amp_valid = zeros(Bool, ax), pinned = zeros(Bool, ax),
    ))
end

"""
    _antenna_freq_segments(blocks, nant) -> (fseg, cells)

The frequency cells a joint solve reduces its data onto — the coarsest
channel partition every block's own segmentation is a union of cells of — and
`fseg[a, k]` is the segment station `a` carries over cell `k`, `0` for a station
no block covers.

Stations that share one frequency segmentation give the identity refinement, so
`cells` is that segmentation's own channel groups and every station's cell maps
to itself.
"""
function _antenna_freq_segments(blocks, nant)
    cell_ids, ncell = common_refinement([b.plan.fseg_id for b in blocks])
    cells = segment_groups(cell_ids, ncell)
    fseg = zeros(Int, Base.OneTo(nant), Base.OneTo(ncell))
    for b in blocks, a in b.antennas
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
    _joint_bandpass_gauge!(gains, data, layout, blocks, gauge; tracks = []) -> Union{Nothing, NamedTuple}

Impose `gauge` on the phase of a joint bandpass solve, over every gain slot's
phase `φ`, numbered by `_gain_slot_ids`. Under `PinAntenna` there is one
[`GaugeFreedom`](@ref) per connected component and the reference's slot in
each is held at zero: the reference's whole track is zero, and the priors act
on every other track relative to it. Under any other gauge the freedoms are
the shifts that neither the data nor the priors of `tracks` (the phase tracks,
`_bandpass_tracks`) see (`_gauge_directions`): the solve holds one
well-measured slot per freedom at zero (`_gauge_pins`), and afterwards moves
along the freedoms onto the constraints `C·φ = d` the gauge gives
(`_meet_gauge!`), which this returns as `(; rows, d, freedoms)`: `rows[j]` the
`(slot, coefficient)` pairs of row `j` and `freedoms[k]` the slots of freedom
`k` with their shift. When the held slots are the constraints themselves, as
under a pin, it returns `nothing`. Held slots are marked in the `pinned` layer
of `gains`; a held slot's amplitude is solved like any other slot's.

Only the phase is a gauge freedom. Multiplying the gains of a set of nodes by a
shared `c` sends `g_a·S·conj(g_b)` to `|c|²·g_a·S·conj(g_b)` on every
correlation internal to that set: the phase of `c` cancels between the two
conjugated factors. The sets on which it cancels everywhere are the connected
components of the joint-bandpass graph, so the data leave one phase per
component free. A station-uniform segmentation splits the graph along the
array-wide (frequency segment, time segment) cells, so the free direction is a
phase spectrum common to every station; a model in which one station breaks
mid-track keeps the whole track in one component, because the stations that
hold one gain over it bridge the broken station's two segments.

A proper prior along frequency is not invariant to that spectrum: it sees all
of it but the part its track's mean takes up — a constant under a level, a
line in frequency under a delay. A pin settles the spectrum by the
reference's data: the reference's bandpass is zero by definition, and its
structure appears, sign flipped, on every other track, where their priors act
on it. A prior on the reference's own track has nothing left to act on, so a
fully held track takes no part in the prior fit. Other gauges leave the
spectrum to the priors: only the shifts every prior is blind to are freedoms,
each fixed by a projection onto its shift being zero (under a zero sum, of
every slot). Along such a freedom the data and the priors are unchanged, so
the gauge changes no gauge-invariant quantity, nor the posterior of anything
the priors see.

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
function _joint_bandpass_gauge!(gains, data, layout, blocks, gauge; tracks = [])
    T = real(eltype(first(gains).g))
    ids = _gain_slot_ids(gains)
    compid, ncomp, nodew = _joint_bandpass_graph(data, layout, ids)
    slot = [(k, I) for (k, idk) in pairs(ids) for I in CartesianIndices(idk)]
    # A slot no correlation touches (`compid` 0) is in no component.
    members = [Int[] for _ in 1:ncomp]
    for (n, c) in pairs(compid)
        iszero(c) || push!(members[c], n)
    end
    # A pin holds the reference's track at zero in every component; any other
    # gauge fixes only the shifts the priors leave free.
    reference = _gauge_for(gauge, first(blocks).plan.path) isa PinAntenna
    directions = reference ? [([c], [1.0]) for c in 1:ncomp] : _gauge_directions(compid, ncomp, tracks)
    freedoms = map(directions) do (comps, shift)
        cn = reduce(vcat, members[comps])
        along = Dict(zip(comps, shift))
        GaugeFreedom(;
            nodes = cn, antenna = [blocks[slot[n][1]].antennas[slot[n][2][1]] for n in cn],
            feed = [slot[n][2][2] for n in cn], scan = zeros(Int, length(cn)),
            component = [blocks[slot[n][1]].plan.path for n in cn], observable = fill(:phase, length(cn)),
            direction = T[along[compid[n]] for n in cn], weight = nodew[cn],
        )
    end
    C, d = _bandpass_gauge_system(gauge, GaugeFreedoms{T}(freedoms, length(compid)))
    Cs = sparse(C)
    rows = [Tuple{Int, T}[] for _ in axes(Cs, 1)]
    for n in axes(Cs, 2), i in nzrange(Cs, n)
        iszero(nonzeros(Cs)[i]) || push!(rows[rowvals(Cs)[i]], (n, nonzeros(Cs)[i]))
    end
    pins = all(j -> length(rows[j]) == 1 && iszero(d[j]), eachindex(rows, d)) ?
        [first(only(row)) for row in rows] : _gauge_pins(freedoms, compid, nodew)
    for n in pins
        k, I = slot[n]
        gains[k].pinned[I] = true
    end
    length(pins) == length(rows) && all(j -> rows[j] == [(pins[j], one(T))], eachindex(rows)) && return nothing
    return (; rows, d, freedoms = [(f.nodes, f.direction) for f in freedoms])
end

# The slots the solve holds at zero to fix the gauge `freedoms`: one per
# freedom, chosen by a pivoted QR of their shifts over the slots any
# correlation touches, each weighted by the square root of its row weight
# `nodew`, so that holding them fixes every freedom on well-measured slots. The
# gauge asked for is reached afterwards by a shift along the freedoms
# (`_meet_gauge!`), so the choice changes no result, only the path to it.
function _gauge_pins(freedoms, compid, nodew)
    pins = Int[]
    for group in _freedom_blocks(freedoms)
        fs = freedoms[group]
        nodes = first(fs).nodes
        N = reduce(hcat, [f.direction for f in fs]) .* sqrt.(nodew[nodes])
        F = qr(Matrix(N'), ColumnNorm())
        append!(pins, nodes[F.p[1:length(fs)]])
    end
    return pins
end

# The indices of `freedoms` grouped by the slots they move: the freedoms of one
# block share them.
function _freedom_blocks(freedoms)
    groups = Dict{Vector{Int}, Vector{Int}}()
    for (j, f) in pairs(freedoms)
        push!(get!(groups, f.nodes, Int[]), j)
    end
    return sort!(collect(values(groups)); by = first)
end

# The phase shifts of the joint-bandpass graph's components that neither the
# data nor the priors of `tracks` see, as `(components, shift)`: one value per
# component, moving every slot of it. The data see none. Track `t`'s prior sees
# every shift but those whose values on its slots lie in its null space
# (`_prior_null_bases`). A component no prior acts on is its own freedom; the
# rest are joined into blocks by the tracks that cross them. In a block, with
# `D` the count of prior-bearing slots in each component and `U` the
# orthonormal null bases carried onto the components, the freedoms are the
# shifts `g` with `(D − UUᵀ)g = 0`, i.e. `g = D⁻¹Uy` for `y` an eigenvector of
# `UᵀD⁻¹U` with eigenvalue one, the largest it has.
function _gauge_directions(compid, ncomp, tracks)
    D = zeros(ncomp)
    columns = Tuple{Vector{Int}, Vector{Float64}}[]
    links = Graphs.SimpleGraph(ncomp)
    for t in tracks, (slots, B) in _prior_null_bases(t)
        inside = findall(!iszero, compid[slots])
        isempty(inside) && continue
        comps = compid[slots[inside]]
        B = B[inside, :]
        foreach(k -> D[k] += 1, comps)
        foreach(i -> Graphs.add_edge!(links, comps[1], comps[i]), eachindex(comps))
        F = svd(B)
        for j in 1:count(>(first(F.S) * length(slots) * eps()), F.S)
            push!(columns, (comps, F.U[:, j]))
        end
    end
    out = Tuple{Vector{Int}, Vector{Float64}}[]
    for block in Graphs.connected_components(links)
        if all(k -> iszero(D[k]), block)
            foreach(k -> push!(out, ([k], [1.0])), block)
            continue
        end
        at = Dict(k => i for (i, k) in pairs(block))
        mine = [c for c in columns if haskey(at, first(c[1]))]
        U = zeros(length(block), length(mine))
        for (j, (comps, q)) in pairs(mine), (k, v) in zip(comps, q)
            U[at[k], j] += v
        end
        Dinv = Diagonal(inv.(D[block]))
        E = eigen(Symmetric(U' * Dinv * U))
        for j in findall(>=(1 - sqrt(eps())), E.values)
            g = Dinv * U * E.vectors[:, j]
            push!(out, (block, g ./ maximum(abs, g)))
        end
    end
    return out
end

# The constraints fixing the bandpass freedoms. A zero sum sets the projection
# of every slot onto a freedom's shift to zero: on a freedom of one component,
# the zero sum of its one value; where a prior spreads freedoms over many
# components, a sum of ones would miss a shift that sums to zero. `ByComponent`
# applies each freedom's own choice; other gauges give their own rows.
_bandpass_gauge_system(g, fs) = _gauge_system(g, fs)

function _bandpass_gauge_system(::ZeroSumPhase, fs::GaugeFreedoms{T}) where {T}
    I, J, V = Int[], Int[], T[]
    for (j, f) in pairs(fs), i in eachindex(f.nodes, f.direction)
        push!(I, j)
        push!(J, f.nodes[i])
        push!(V, f.direction[i] / length(f.nodes))
    end
    return sparse(I, J, V, length(fs), fs.nnodes), zeros(T, length(fs))
end

function _bandpass_gauge_system(g::ByComponent, fs::GaugeFreedoms{T}) where {T}
    I, J, V = Int[], Int[], T[]
    d = zeros(T, length(fs))
    choice = [Calibration._component_choice(g, f) for f in fs]
    for c in unique(choice)
        idx = findall(x -> x === c, choice)
        Cc, d[idx] = _bandpass_gauge_system(c, GaugeFreedoms{T}(fs.freedoms[idx], fs.nnodes))
        Ic, Jc, Vc = findnz(sparse(Cc))
        append!(I, idx[Ic])
        append!(J, Jc)
        append!(V, Vc)
    end
    return sparse(I, J, V, length(fs), fs.nnodes), d
end

# Shift `φ` along the gauge freedoms onto the constraints `gauge.rows`/`gauge.d`.
# Neither the data nor the priors change along a freedom, so the shift moves a
# solve held by its pins onto the gauge asked for without changing anything
# else. `freedoms` holds each freedom's slots and its shift there.
function _meet_gauge!(φ, gauge, freedoms)
    along = [Dict(zip(nodes, shift)) for (nodes, shift) in freedoms]
    C = zeros(length(gauge.rows), length(freedoms))
    r = copy(gauge.d)
    for (j, row) in pairs(gauge.rows), (n, c) in row
        for (k, a) in pairs(along)
            C[j, k] += c * get(a, n, 0.0)
        end
        r[j] -= c * φ[n]
    end
    δ = C \ r
    for (k, (nodes, shift)) in pairs(freedoms), (n, v) in zip(nodes, shift)
        φ[n] += δ[k] * v
    end
    return φ
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
