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
# The data are `(Scan, AntennaPair, FeedPair, cell)` arrays of the scans
# reduced onto frequency cells; `cover[si, c]` names one time and one channel
# inside entry `(si, c)`, at which each component's parameters are read
# (`StationGains`).

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

# The constraints a gauge puts on the station gains, one freedom per connected
# group: a zero sum sets each group's weighted direction to sum to zero;
# `ByComponent` applies each freedom's own choice; other gauges give their own
# rows.
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

# The pattern of the χ²'s curvature in one observable's station-gain values,
# over the pairs of gains each correlation `e` joins: a matrix holding it and,
# per `e`, the indices among its stored values of `(a, a)`, `(b, b)`, `(a, b)`
# and `(b, a)`.
function _curvature_pattern(gain_pairs, ngains)
    Is = [n for (na, nb) in gain_pairs for n in (na, nb, na, nb)]
    Js = [n for (na, nb) in gain_pairs for n in (na, nb, nb, na)]
    H = sparse(Is, Js, ones(length(Is)), ngains, ngains)
    pos = [ntuple(r -> _position(H, Is[4(e - 1) + r], Js[4(e - 1) + r]), 4) for e in eachindex(gain_pairs)]
    return H, pos
end

# Writes into `H` (`_curvature_pattern`) the curvature of the χ² in one
# observable's station-gain values, `Σ_e q_e·(x_a + s·x_b)²`, with `s = +1` for
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
