# ── Phase-cal (injected tone) instrumental calibration ─────────────────────────
#
# VGOS-style systems inject a phase-locked tone comb at each station front end;
# the correlator extracts the tone phasors per (station, polarization, band,
# epoch) and writes them to the FITS-IDI PHASE-CAL table. Those tones measure the
# station's INSTRUMENTAL phase response — per-band delays and phase offsets that
# otherwise decohere the multi-band fringe — so dividing them out of the
# visibilities before fringe fitting aligns the bands (EU-VGOS / fourfit
# "multitone" scheme, Alef et al. 2024, A&A: per band fit a tone delay τ_pc from
# the tone-phase slope, then the instrumental phase φ_pc as the mean tone
# residual phase).
#
# Everything here is format-neutral: `load_fitsidi_phasecal` (the FITS-IDI
# reader) is a stub implemented in `GustavoFITSFilesExt`. The fitted correction
# is packed as an ordinary `CalibrationSolution` — a phase-only
# `StationGainModel` with per-scan, per-spectral-window `Delay` + `ConstantTerm`
# per feed — so applying, saving, and composing it reuses the Calibration
# machinery unchanged. What phase-cal does NOT fix: the ionospheric dispersive
# delay (∝ 1/ν; it enters after the injection point) and anything sky-side —
# those stay with the fringe fitter.

"""
    PhaseCalTable

Format-neutral phase-cal tone table (one row per station × epoch):

- `station[row]`  — station code (matches `AntennaTable` names).
- `time[row]`     — epoch centre, HOURS since the file reference date (the
  UVSet `Ti` convention).
- `interval[row]` — accumulation interval (hours).
- `cable[row]`    — cable-cal delay (s); `NaN` when absent.
- `freq[tone, band, feed, row]` — tone sky frequency (Hz); `NaN` = absent.
- `tone[tone, band, feed, row]` — measured tone phasor; `NaN` = absent.

Produced by [`load_fitsidi_phasecal`](@ref); consumed by
[`phasecal_solution`](@ref) and [`tone_channel_mask`](@ref).
"""
struct PhaseCalTable
    station::Vector{String}
    time::Vector{Float64}
    interval::Vector{Float64}
    cable::Vector{Float64}
    freq::Array{Float64, 4}
    tone::Array{ComplexF64, 4}
end

"""
    load_fitsidi_phasecal(path) -> PhaseCalTable

Read a FITS-IDI PHASE-CAL table (AIPS Memo 114) into a [`PhaseCalTable`](@ref).
Provided by `GustavoFITSFilesExt` (load FITSFiles).
"""
function load_fitsidi_phasecal end

# The phase-cal correction model: per-feed, per-scan, per-spectral-window delay
# and constant phase — exactly what the multitone fit measures. Phase-only
# (tone amplitudes track injection power, not the signal-path gain).
function _phasecal_model()
    return StationGainModel(
        phase = (
            TiedComponent(GainComponent(Delay(), PerScan(), PerSpectralWindow()), PerFeed()),
            TiedComponent(GainComponent(ConstantTerm(), PerScan(), PerSpectralWindow()), PerFeed()),
        ),
        logamp = (),
    )
end

# Robust multitone fit for the tones of ONE (station, feed, scan, spw) block:
# `ν` (Hz) ascending with phasors `z`. Returns `(τ, φ0, nused)` where the block's
# instrumental phase is `φ(f) = φ0 + 2πτ(f − f0)` (φ0 referenced DIRECTLY to the
# global f0 — the `Delay` term's coordinate — so no per-band reference juggling),
# or `nothing` when fewer than `min_tones` usable tones survive.
#
# τ from the median of the wrapped adjacent-increment estimates (immune to a
# single bad tone; unambiguous for |τ| < 1/(2·tone spacing), ±100 ns for the
# 5 MHz VGOS comb — the same inherent ambiguity as fourfit's tone fringe), then
# outlier tones rejected by their residual phase and τ refined by a weighted LSQ
# on the residuals.
function _fit_tone_block(ν::Vector{Float64}, z::Vector{ComplexF64}, f0::Float64; min_tones::Int = 2, max_resid::Float64 = 1.0)
    n = length(ν)
    n >= min_tones || return nothing
    νb = sum(ν) / n

    τ = 0.0
    if n >= 2
        incr = Float64[]
        for j in 1:(n - 1)
            dν = ν[j + 1] - ν[j]
            dν > 0 || continue
            push!(incr, angle(z[j + 1] * conj(z[j])) / (2π * dν))
        end
        isempty(incr) && return nothing
        τ = median(incr)
    end

    # Residual phasors at the median delay; reject tones whose residual phase is
    # far from the pack (a corrupted tone leaves ~uniform phase).
    keep = trues(n)
    r = [z[i] * cis(-2π * τ * (ν[i] - νb)) for i in 1:n]
    rbar = sum(r)
    if abs(rbar) > 0
        for i in 1:n
            keep[i] = abs(angle(r[i] * conj(rbar))) <= max_resid
        end
    end
    count(keep) >= min_tones || return nothing

    # Weighted LSQ refinement of τ on the kept residual phases (small after the
    # median fit, so no unwrap needed), weights = |z| (tone detection strength).
    rbar = sum(r[i] for i in 1:n if keep[i])
    num = 0.0
    den = 0.0
    for i in 1:n
        keep[i] || continue
        w = abs(z[i])
        x = ν[i] - νb
        num += w * x * angle(r[i] * conj(rbar))
        den += w * x^2
    end
    den > 0 && (τ += num / (2π * den))

    # Instrumental phase at the GLOBAL reference frequency.
    s = zero(ComplexF64)
    for i in 1:n
        keep[i] || continue
        s += z[i] * cis(-2π * τ * (ν[i] - f0))
    end
    abs(s) > 0 || return nothing
    return (τ = τ, φ0 = angle(s), nused = count(keep))
end

"""
    phasecal_solution(pcal::PhaseCalTable, uvset::UVSet;
                      sign = -1, min_tones = 2, max_resid = 1.0) -> CalibrationSolution

Fit the fourfit-style multitone instrumental correction from `pcal` on the
geometry of `uvset`: per (station, feed, scan, spectral window), a robust tone
delay + constant phase (see `_fit_tone_block`), packed as a phase-only
`CalibrationSolution` (`Delay` + `ConstantTerm`, `PerScan` × `PerSpectralWindow`,
`PerFeed`). Apply it with `apply_calibration(uvset, sol)` or put
`ApplySolution(pcal)` in the pipeline's transform chain (applied in-stream, no
materialization of the full set).

- `sign` — orientation of the correction: the gain stored is
  `cis(sign·(φ_pc + 2πτ_pc(f − f0)))` and `apply_calibration` DIVIDES it out.
  The default `sign = +1` (divide by the measured tone phasor itself) is the
  orientation validated on VGOS DiFX data (VR2505: it raises the multi-band
  fringe amplitude ~30% on a real scan; `sign = -1` lowers it by the same
  amount). Flip it if your correlator uses the opposite convention — the wrong
  sign ADDS the instrumental decoherence instead of removing it, which the
  fringe SNR makes obvious.

Note the fourfit-shared limitation: the tone delay is ambiguous modulo
1/(tone spacing) (±100 ns for the 5 MHz VGOS comb). A wrapped tone delay still
reproduces every tone phase exactly (the wrap is absorbed into the constant),
but the interpolation BETWEEN tones winds; that residual per-channel structure
is exactly what the fringe solve's phase-bandpass stage absorbs — keep
`phase_bandpass = true` when instrumental delays may exceed the ambiguity.
- Tones are matched to spectral windows BY FREQUENCY (FITS band order need not
  match the geometry's spw order). Blocks with no usable tones get identity
  gains, and stations absent from the table are left uncorrected.

The `info` reports coverage: `nblocks` fitted / `nmissing` empty, per-station
fit counts, and the tone-delay range.
"""
function phasecal_solution(
        pcal::PhaseCalTable, uvset::UVSet;
        sign::Integer = 1, min_tones::Integer = 2, max_resid::Real = 1.0,
    )
    abs(Int(sign)) == 1 || error("phasecal_solution: sign must be +1 or -1")
    geom = build_geometry(uvset)
    first_leaf = first(values(UVData.branches(uvset)))
    ants = UVData.metadata(first_leaf).antennas
    nant = length(ants)
    ant_names = String.(collect(ants.name))
    ant_of = Dict(ant_names[i] => i for i in eachindex(ant_names))

    model = _phasecal_model()
    layout = plan_parameters(model, nant, geom)
    θ = zeros(layout.nθ)
    delay_plan = layout.plans[1]
    const_plan = layout.plans[2]

    # Scan time spans (hours) per time-segment id of the model.
    ntseg = maximum(delay_plan.tseg_id; init = 0)
    tmin = fill(Inf, ntseg)
    tmax = fill(-Inf, ntseg)
    for (ti, ts) in enumerate(delay_plan.tseg_id)
        tmin[ts] = min(tmin[ts], geom.times[ti])
        tmax[ts] = max(tmax[ts], geom.times[ti])
    end

    # Frequency span per spw segment (for tone → spw matching), padded by one
    # channel spacing so edge tones land inside.
    nfseg = maximum(delay_plan.fseg_id; init = 0)
    flo = fill(Inf, nfseg)
    fhi = fill(-Inf, nfseg)
    for (c, fs) in enumerate(delay_plan.fseg_id)
        flo[fs] = min(flo[fs], geom.channel_freqs[c])
        fhi[fs] = max(fhi[fs], geom.channel_freqs[c])
    end
    dch = length(geom.channel_freqs) > 1 ? median(diff(sort(geom.channel_freqs))) : 0.0
    flo .-= dch
    fhi .+= dch

    # Rows per station (table row indices), sorted by time.
    rows_of = Dict{String, Vector{Int}}()
    for (r, st) in enumerate(pcal.station)
        push!(get!(rows_of, st, Int[]), r)
    end
    for v in values(rows_of)
        sort!(v; by = r -> pcal.time[r])
    end

    ntone, nband, nfeed, _ = size(pcal.tone)
    f0 = geom.f0
    sgn = Int(sign)
    nfit = 0
    nmissing = 0
    fit_per_ant = zeros(Int, nant)
    τlo = Inf
    τhi = -Inf

    νbuf = Float64[]
    zbuf = ComplexF64[]
    for (st, rows) in rows_of
        ant = get(ant_of, st, 0)
        ant == 0 && continue
        for ts in 1:ntseg
            # Rows overlapping this scan (± one accumulation interval); fall back
            # to the nearest row — the instrument drifts slowly between scans.
            tc = (tmin[ts] + tmax[ts]) / 2
            sel = [
                r for r in rows if pcal.time[r] >= tmin[ts] - pcal.interval[r] - 1.0e-9 &&
                    pcal.time[r] <= tmax[ts] + pcal.interval[r] + 1.0e-9
            ]
            isempty(sel) && (sel = [rows[argmin([abs(pcal.time[r] - tc) for r in rows])]])
            for feed in 1:min(nfeed, 2), fs in 1:nfseg
                # Coherent tone average over the selected rows, keeping tones in
                # this spw's frequency span.
                empty!(νbuf)
                empty!(zbuf)
                for b in 1:nband, tn in 1:ntone
                    ν = pcal.freq[tn, b, feed, sel[1]]
                    (isfinite(ν) && flo[fs] <= ν <= fhi[fs]) || continue
                    acc = zero(ComplexF64)
                    for r in sel
                        z = pcal.tone[tn, b, feed, r]
                        isfinite(z) || continue
                        acc += z
                    end
                    abs(acc) > 0 || continue
                    push!(νbuf, ν)
                    push!(zbuf, acc)
                end
                ord = sortperm(νbuf)
                fit = _fit_tone_block(νbuf[ord], zbuf[ord], f0; min_tones = Int(min_tones), max_resid = Float64(max_resid))
                if fit === nothing
                    nmissing += 1
                    continue
                end
                θ[delay_plan.off1[ant, feed, ts, fs]] = sgn * fit.τ
                θ[const_plan.off1[ant, feed, ts, fs]] = sgn * fit.φ0
                nfit += 1
                fit_per_ant[ant] += 1
                τlo = min(τlo, fit.τ)
                τhi = max(τhi, fit.τ)
            end
        end
    end

    info = (;
        nant = nant,
        nscan = ntseg,
        nspw = nfseg,
        nblocks = nfit,
        nmissing = nmissing,
        fit_per_ant = fit_per_ant,
        tone_delay_range_ns = (τlo * 1.0e9, τhi * 1.0e9),
        sign = sgn,
        ant_names = ant_names,
    )
    return CalibrationSolution(model, layout, geom, θ, info)
end

"""
    tone_channel_mask(pcal::PhaseCalTable, uvset::UVSet; pad::Integer = 0) -> BitVector

`true` for every GLOBAL channel (geometry order) that contains an injected
phase-cal tone (± `pad` neighbouring channels). Tone combs are phase-locked at
every station, so they can cross-correlate and leave spurious spikes in those
channels; put `FlagChannels(mask)` in the pipeline's transform chain to
zero-weight them during the solve (≈ ntones/nchan of the data, ~4% for the
VGOS 5 MHz comb at 0.2 MHz channels).
"""
function tone_channel_mask(pcal::PhaseCalTable, uvset::UVSet; pad::Integer = 0)
    geom = build_geometry(uvset)
    freqs = geom.channel_freqs
    nchan = length(freqs)
    mask = falses(nchan)
    nchan == 0 && return mask
    dch = nchan > 1 ? median(diff(sort(freqs))) : 1.0
    tones = sort!(unique(filter(isfinite, vec(pcal.freq))))
    isempty(tones) && return mask
    for (c, f) in enumerate(freqs)
        j = searchsortedfirst(tones, f)
        near = min(
            j <= length(tones) ? abs(tones[j] - f) : Inf,
            j > 1 ? abs(tones[j - 1] - f) : Inf,
        )
        mask[c] = near <= (0.5 + pad) * dch
    end
    return mask
end
