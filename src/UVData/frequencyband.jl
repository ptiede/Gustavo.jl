"""
    FrequencySetup(; name, ref_freq, channel_freqs, ch_widths, total_bandwidths,
                     sidebands, extras = NamedTuple())

Frequency-axis description for a UVData dataset. Basis-agnostic — no AIPS-
specific identifiers beyond the optional `extras`.

- `name`              : Identifier for this spectral setup (typically `String`).
- `ref_freq`          : Reference frequency (Hz).
- `channel_freqs`     : Absolute IF center frequencies (Hz).
- `ch_widths`         : Per-IF channel width (Hz).
- `total_bandwidths`  : Per-IF total bandwidth (Hz).
- `sidebands`         : Per-IF sideband (+1 = upper, -1 = lower).
- `extras`            : `NamedTuple` of optional FQ-table columns preserved
  verbatim for round-trip (e.g. `BANDCODE`, raw `FRQSEL`).
"""
Base.@kwdef struct FrequencySetup{N, TFreq, TCfreqs, TCw, TBw, TSb, TExtras <: NamedTuple}
    name::N
    ref_freq::TFreq
    channel_freqs::TCfreqs
    ch_widths::TCw
    total_bandwidths::TBw
    sidebands::TSb
    extras::TExtras = NamedTuple()
end

function Base.show(io::IO, fs::FrequencySetup)
    flo = round(minimum(fs.channel_freqs) / 1.0e9; digits = 3)
    fhi = round(maximum(fs.channel_freqs) / 1.0e9; digits = 3)
    return print(io, "FrequencySetup($(fs.name), $(length(fs.channel_freqs)) IFs, $(flo)–$(fhi) GHz)")
end
