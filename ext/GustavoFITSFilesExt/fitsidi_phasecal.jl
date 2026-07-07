# FITS-IDI PHASE-CAL table reader (AIPS Memo 114 §3.7): the correlator-extracted
# phase-cal tone phasors per (station, polarization, band, epoch), plus the
# cable-cal delay. AIPS specifics (NOSTA→name mapping, TIME in days since RDATE,
# PC_* column layout `ntone × nband` flattened per row, _1/_2 polarization
# suffixes matching POLTYA/POLTYB feed order) live here; the multitone fit in
# `src/Fringe/phasecal.jl` is format-neutral.

import Gustavo.Fringe
using Gustavo.Fringe: PhaseCalTable

# One PC_* column group (`_1` or `_2`) reshaped to (ntone, nband, nrow); a
# missing column (NO_POL = 1) yields all-NaN.
function _idi_pc_pol!(freq, tone, d, suffix, ntone, nband, nrow)
    fcol = Symbol("PC_FREQ_", suffix)
    rcol = Symbol("PC_REAL_", suffix)
    icol = Symbol("PC_IMAG_", suffix)
    props = propertynames(d)
    if !(fcol in props)
        fill!(freq, NaN)
        fill!(tone, ComplexF64(NaN, NaN))
        return nothing
    end
    f = getproperty(d, fcol)
    re = getproperty(d, rcol)
    im = getproperty(d, icol)
    for r in 1:nrow
        fr = f[r, :]
        rr = re[r, :]
        ir = im[r, :]
        for b in 1:nband, tn in 1:ntone
            k = (b - 1) * ntone + tn
            freq[tn, b, r] = Float64(fr[k])
            tone[tn, b, r] = complex(Float64(rr[k]), Float64(ir[k]))
        end
    end
    return nothing
end

function Fringe.load_fitsidi_phasecal(path::AbstractString)
    fid = FITSFiles.fits(path)
    pc = _idi_find_hdu(fid, "PHASE-CAL")
    pc === nothing && error("load_fitsidi_phasecal: no PHASE-CAL HDU in $(path)")
    ag = _idi_find_hdu(fid, "ARRAY_GEOMETRY")
    ag === nothing && error("load_fitsidi_phasecal: no ARRAY_GEOMETRY HDU in $(path)")

    # NOSTA → cleaned station name (must match what `load_fitsidi` stores).
    nosta = round.(Int, collect(getproperty(ag.data, :NOSTA)))
    anames = _idi_clean.(collect(getproperty(ag.data, :ANNAME)))
    name_of = Dict(nosta[i] => anames[i] for i in eachindex(nosta))

    ntone = Int(something(card_value(pc.cards, "NO_TONES"), 1))
    nband = Int(something(card_value(pc.cards, "NO_BAND"), 1))
    npol = Int(something(card_value(pc.cards, "NO_POL"), 1))

    d = pc.data
    ant = round.(Int, collect(getproperty(d, :ANTENNA_NO)))
    nrow = length(ant)
    station = [get(name_of, a, string("ant", a)) for a in ant]
    time = Float64.(collect(getproperty(d, :TIME))) .* 24.0          # days → hours since RDATE
    interval = Float64.(collect(getproperty(d, :TIME_INTERVAL))) .* 24.0
    cable = :CABLE_CAL in propertynames(d) ?
        Float64.(collect(getproperty(d, :CABLE_CAL))) : fill(NaN, nrow)

    freq = Array{Float64, 4}(undef, ntone, nband, 2, nrow)
    tone = Array{ComplexF64, 4}(undef, ntone, nband, 2, nrow)
    _idi_pc_pol!(view(freq, :, :, 1, :), view(tone, :, :, 1, :), d, 1, ntone, nband, nrow)
    if npol >= 2
        _idi_pc_pol!(view(freq, :, :, 2, :), view(tone, :, :, 2, :), d, 2, ntone, nband, nrow)
    else
        freq[:, :, 2, :] .= NaN
        tone[:, :, 2, :] .= ComplexF64(NaN, NaN)
    end
    return PhaseCalTable(station, time, interval, cable, freq, tone)
end
