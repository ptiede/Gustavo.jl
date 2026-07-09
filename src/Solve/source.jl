# ── Source model (Stage 2, Milestone 5) ──────────────────────────────────────
#
# The forward model predicts `V̂ = gₐ·S·conj(g_b)` from the per-station gains and
# a 2×2 source coherency `S`. For GENERIC fringe fitting we do NOT want to assume
# the source flux/structure or an absolute amplitude scale (Tsys may be absent):
# the source is a NUISANCE that is PROFILED OUT.
#
# The model is nonlinear in the gains, but for FIXED gains it is LINEAR in `S`
# (`V̂ = (gₐ·conj(g_b))·S`), so the ML source is a closed-form weighted
# least-squares estimate — variable projection. Substituting it back gives a
# source-agnostic reduced objective whose gradient wrt the gains is the plugged-in
# `∂ℓ/∂g` (envelope theorem: `∂ℓ/∂S = 0` at `Ŝ`), so Enzyme differentiates through
# the profile correctly. Profiling the source also makes the overall amplitude
# scale free (absorbed by `Ŝ`) — relative amplitude, no Tsys needed.

abstract type AbstractSourceModel end

"""
    FixedCoherency(S)

A fixed, known per-baseline source coherency `S :: (nbl, 2, 2)` (e.g. from a
Comrade sky model). Not profiled.
"""
struct FixedCoherency{A <: AbstractArray} <: AbstractSourceModel
    S::A
end

"""
    PointSource(flux = 1.0)

A fixed unpolarized point source: `S = flux·I` on every baseline. The
amplitude-calibrated default when the flux is known.
"""
struct PointSource{T <: Real} <: AbstractSourceModel
    flux::T
end
PointSource(; flux::Real = 1.0) = PointSource(flux)

"""
    ProfiledPointSource()

An unpolarized point source of UNKNOWN complex flux, profiled per leaf (per
scan × spectral window) in closed form from the current gains and data —
variable projection. This is the generic fringe-fit default: it assumes no source
structure and no absolute amplitude scale (the profiled flux absorbs the overall
gain-magnitude/phase common mode, so the solve stabilizes phase and amplitude
across scans and IFs *relatively*, without Tsys).
"""
struct ProfiledPointSource <: AbstractSourceModel end

# The 2×2-per-baseline coherency this source presents for a leaf, given the
# windowed gains `g` and the leaf context `ctx`. Fixed sources ignore `g`; the
# profiled source solves the ML flux from `g` + data (differentiable in `g`).
leaf_source(s::FixedCoherency, g, ctx) = s.S

function leaf_source(s::PointSource, g, ctx)
    return point_source_coherency(length(ctx.bl_a); flux = s.flux, T = complex(float(eltype(g))))
end

function leaf_source(::ProfiledPointSource, g, ctx)
    ŝ = _profile_point_flux(g, ctx)
    nbl = length(ctx.bl_a)
    T = typeof(ŝ)
    S = Array{T}(undef, nbl, 2, 2)
    z = zero(T)
    @inbounds for bi in 1:nbl
        S[bi, 1, 1] = ŝ
        S[bi, 2, 2] = ŝ
        S[bi, 1, 2] = z
        S[bi, 2, 1] = z
    end
    return S
end

# Closed-form ML complex flux over the leaf's PARALLEL-hand samples (the cross
# hands carry no unpolarized-source signal): with `A = gₐ·conj(g_b)`,
# `ŝ = Σ w·conj(A)·V / Σ w·|A|²`. Branch only on data (`w`, finiteness); the
# denominator is > 0 for any leaf with parallel-hand data and finite gains.
function _profile_point_flux(g::AbstractArray{<:Complex}, ctx)
    Vobs = ctx.vis
    W = ctx.weights
    nchan, nti, nbl, npol = size(Vobs)
    T = complex(float(eltype(g)))
    num = zero(T)
    den = zero(real(T))
    @inbounds for p in 1:npol
        fa = ctx.feed_a[p]
        fb = ctx.feed_b[p]
        fa == fb || continue                      # parallel hands only
        for bi in 1:nbl
            a = ctx.bl_a[bi]
            b = ctx.bl_b[bi]
            for t in 1:nti, c in 1:nchan
                w = W[c, t, bi, p]
                vo = Vobs[c, t, bi, p]
                (w > 0 && isfinite(vo)) || continue
                A = g[c, t, a, fa] * conj(g[c, t, b, fb])
                num += w * conj(A) * vo
                den += w * abs2(A)
            end
        end
    end
    return num / den
end
