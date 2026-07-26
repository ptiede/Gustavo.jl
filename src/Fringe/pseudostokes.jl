# ── Pseudo-Stokes-I adhoc rows (linear-feed / VGOS mixed-pol) ─────────────────
#
# For LINEAR feeds the field rotation ψ(t) is a REAL 2×2 rotation of the feed
# basis, not a phase: with an unpolarized source
#
#     V(ab) = (I/2)·R(ψ_a)·R(ψ_b)ᵀ  ⇒  XX = YY ∝ cos Δ,  XY = −YX ∝ sin Δ,
#
# Δ = ψ_a − ψ_b. A single product can NULL outright (cos Δ → 0) and flips sign
# through the null — near it the few-percent polarized flux dominates and the
# product's phase is arbitrary. The structure is baseline-dependent (through
# cos Δ), so NO station-gain model can absorb it; on near-polar sources
# (1803+784, dec +78°) ψ swings fast and the per-AP adhoc solve is handed
# parallel-hand rows that disagree by >100° per baseline — its compromise then
# drifts (the fake ±π station-track arcs on VR2505 WN/KE). Circular feeds
# never see any of this: there rotation is the station PHASE e^{∓iψ}, already
# the form of a station gain.
#
# The fourfit VGOS answer, applied to the adhoc's rbar/wbar ROWS ONLY (the
# visibilities and the solved gains are untouched — consistent with the
# "no field-rotation correction in the fringe fitter" rule): combine all four
# products into one pseudo-Stokes-I phasor with the rotation coefficients,
# recovering full Stokes-I amplitude at every Δ. With rbar_p = wbar_p·m_p and
# m_p ≈ c_p·(I·e^{iφ_ab}), the WLS combination is
#
#     rI = Σ_p c_p·rbar_p,   wI = Σ_p c_p²·wbar_p,
#     c = (cos Δ, −sin Δ, +sin Δ, cos Δ) for (XX, XY, YX, YY).

using Dates: DateTime, Date, datetime2julian
using AstroLib: ct2lst
import ..UVData: _ecef_to_geodetic

# Field-rotation angle per (station, time): ψ = f_par·q + f_el·el + offset +
# feed-a pol angle, with the mount coefficients from the antenna table (alt-az
# ⇒ (1, 0); Naismith ⇒ (1, ±1)) — the same convention as the Comrade/EHT
# feed-rotation model. `jds` are UTC Julian dates; `ra`/`dec` radians.
function _field_rotation_angles(ants, ra::Real, dec::Real, jds::AbstractVector{<:Real})
    nant = length(ants.name)
    ψ = zeros(nant, length(jds))
    for a in 1:nant
        lat, lon, _ = _ecef_to_geodetic(ants.station_xyz[a])
        m = ants.mount[a]
        fq = Float64(UVData.parallactic_mount(m))
        fe = Float64(UVData.elevation_mount(m))
        off = Float64(UVData.offset_mount(m)) + Float64(ants.pol_angles[a][1])
        for (k, jd) in enumerate(jds)
            lst_hr = ct2lst(rad2deg(lon), jd)
            ha = mod(deg2rad(lst_hr * 15.0) - ra + π, 2π) - π
            q = atan(cos(lat) * sin(ha), sin(lat) * cos(dec) - cos(lat) * sin(dec) * cos(ha))
            el = asin(clamp(sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(ha), -1.0, 1.0))
            ψ[a, k] = fq * q + fe * el + off
        end
    end
    return ψ
end

# Collapse the per-AP residuals to pseudo-Stokes-I in place: the combined
# phasor lands in the first parallel-hand product slot, every other product is
# zeroed (the shared-feeds solve then sees ONE consistent high-SNR row per
# baseline per AP). `sign` flips the sin-coefficient convention (mount/feed
# handedness); the default is validated on VR2505.
#
# `cross_hands` (default false) chooses HOW MUCH of the fourfit collapse to
# trust. The parallel-hand rows alone already carry the failure the collapse
# exists to fix: XX, YY ∝ cos Δ, so through the rotation NULL (cos Δ → 0) they
# change SIGN — a raw ±π jump in the adhoc row exactly where a near-zenith
# station (VR2505 KE) sweeps Δ through 90°. Multiplying by c = cos Δ un-flips
# it (cos Δ·(cos Δ·I e^{iφ}) = cos²Δ·I e^{iφ} ≥ 0) and drives the weight to 0
# AT the null, where the per-track smoother bridges (KE barely needs adhoc:
# noadh η ≈ 0.99). That parallel-only sign correction is band-INDEPENDENT and
# safe. Pulling in the cross hands (∓sin Δ·XY) additionally recovers Stokes-I
# amplitude THROUGH the null — but only if the source is unpolarized with no
# leakage; then XY = −YX and their phase IS φ_ab. With real cross-pol the XY/YX
# phase is frequency-dependent leakage/source polarization, and near the null
# (where sin Δ dominates) it swamps the row: on VR2505 0607−157 it HELPS band 1
# (small leakage) but injects a 330° KE arc in band 4 (large/rotated leakage).
# So default to the parallel-only correction; `cross_hands = true` is opt-in for
# data known unpolarized and leakage-free.
function _pseudo_stokes_collapse!(
        rbar, wbar, bl_pairs, pol_products, ψ::AbstractMatrix{<:Real};
        sign::Real = 1.0, cross_hands::Bool = false,
    )
    feeds = [correlation_feed_pair(p) for p in pol_products]
    pI = findfirst(f -> f[1] == f[2], feeds)
    pI === nothing && return rbar, wbar
    nbl, npol, nap = size(rbar)
    @inbounds for ap in 1:nap, bi in 1:nbl
        a, b = bl_pairs[bi]
        a == b && continue
        Δ = ψ[a, ap] - ψ[b, ap]
        c, s = cos(Δ), Float64(sign) * sin(Δ)
        rI = zero(eltype(rbar)); wI = 0.0
        for p in 1:npol
            fa, fb = feeds[p]
            if fa == fb
                cp = c
            elseif cross_hands
                cp = fa < fb ? -s : s
            else
                continue                      # parallel-only sign correction
            end
            rI += cp * rbar[bi, p, ap]
            wI += cp^2 * wbar[bi, p, ap]
        end
        for p in 1:npol
            rbar[bi, p, ap] = zero(eltype(rbar))
            wbar[bi, p, ap] = 0.0
        end
        rbar[bi, pI, ap] = rI
        wbar[bi, pI, ap] = wI
    end
    return rbar, wbar
end

# Resolve the `adhoc_pseudo_stokes` option against the data: `:auto` enables it
# when the feeds are LINEAR (X/Y nominal basis — circular feeds don't need it
# and the synthetic test arrays are circular). Returns `nothing` (disabled) or
# a NamedTuple carrying the JD epoch the per-scan ψ computation needs.
function _pseudo_stokes_config(option, first_leaf, root_meta)
    enabled = option === true || (
        option === :auto &&
            UVData.metadata(first_leaf).antennas.nominal_basis[1][1] isa UVData.XPol
    )
    enabled || return nothing
    rdate = root_meta.array_obs.rdate
    isempty(rdate) && return nothing            # no epoch — cannot compute ψ
    return (; base_jd = datetime2julian(DateTime(Date(rdate))))
end
