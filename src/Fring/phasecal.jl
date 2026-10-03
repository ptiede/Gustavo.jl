# ── Phase-cal (injected tone) instrumental calibration ───────────────────────
#
# VGOS-style systems inject a phase-locked tone comb at each station front end;
# the correlator extracts the tone phasors per (station, polarization, band,
# epoch) into the FITS-IDI phase-CAL table. The tones measure the station's
# instrumental phase response — per-band delays and phase offsets that
# otherwise decohere the multi-band fringe — so dividing them out before fringe
# fitting aligns the bands. Per band, a tone delay τ_pc is fit from the
# tone-phase slope and the instrumental phase φ_pc is the mean tone residual
# phase (EU-VGOS / fourfit "multitone", Alef et al. 2024, A&A).
#
# `load_fitsidi_phasecal` is a stub implemented in `GustavoFITSFilesExt`.

"""
    PhaseCalTable

Format-neutral phase-cal tone table (one row per station × epoch):

- `station[row]`  — station code (matches `AntennaTable` names).
- `time[row]`     — epoch centre, seconds since `UVData.JD_UNIX_EPOCH`.
- `interval[row]` — accumulation interval (seconds).
- `cable[row]`    — cable-cal delay (s); `NaN` when absent.
- `freq[tone, band, feed, row]` — tone sky frequency (Hz); `NaN` = absent.
- `tone[tone, band, feed, row]` — measured tone phasor, in the same phase
  convention as the visibilities; `NaN` = absent.

Produced by [`load_fitsidi_phasecal`](@ref).
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

Read a FITS-IDI phase-CAL table (AIPS Memo 114) into a [`PhaseCalTable`](@ref).
The tone phasors are conjugated on read, as the visibilities are, so the table
is in the same phase convention as the visibilities.
Provided by `GustavoFITSFilesExt` (load FITSFiles).
"""
function load_fitsidi_phasecal end
