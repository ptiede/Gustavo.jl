"""
    AbstractFrequencySetup

Per-leaf frequency-axis description. Concrete subtypes implement
`channel_freqs`, `ref_freq`, `ch_widths`, `total_bandwidths`, `sidebands`,
and `setup_name`; defaults are provided for `nchannels`, `band_center_frequency`,
and `centered_channel_freqs`. `Base.length` and `Base.iterate` walk the
channel frequencies, so `for ν in fs end` and broadcast over a setup work.
"""
abstract type AbstractFrequencySetup end

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
Base.@kwdef struct FrequencySetup{N, TFreq, TCfreqs, TCw, TBw, TSb, TExtras <: NamedTuple} <: AbstractFrequencySetup
    name::N
    ref_freq::TFreq
    channel_freqs::TCfreqs
    ch_widths::TCw
    total_bandwidths::TBw
    sidebands::TSb
    extras::TExtras = NamedTuple()
end

# Interface implementations.
channel_freqs(fs::FrequencySetup) = fs.channel_freqs
ref_freq(fs::FrequencySetup) = fs.ref_freq
ch_widths(fs::FrequencySetup) = fs.ch_widths
total_bandwidths(fs::FrequencySetup) = fs.total_bandwidths
sidebands(fs::FrequencySetup) = fs.sidebands
setup_name(fs::FrequencySetup) = fs.name

# Defaults that lean on the interface methods so subtypes inherit them.
nchannels(fs::AbstractFrequencySetup) = length(channel_freqs(fs))
function band_center_frequency(fs::AbstractFrequencySetup)
    cf = channel_freqs(fs)
    return (first(cf) + last(cf)) / 2
end
centered_channel_freqs(fs::AbstractFrequencySetup) =
    channel_freqs(fs) .- band_center_frequency(fs)

Base.length(fs::AbstractFrequencySetup) = nchannels(fs)
Base.iterate(fs::AbstractFrequencySetup, state...) = iterate(channel_freqs(fs), state...)
Base.eltype(::Type{<:AbstractFrequencySetup}) = Any

# Structural equality and hashing — two independently-constructed setups
# with field-equal contents compare equal and hash the same so they can
# dedup in a Dict (used by the FITS writer to assign FREQID per setup).
Base.:(==)(a::FrequencySetup, b::FrequencySetup) =
    a.name == b.name && a.ref_freq == b.ref_freq &&
    a.channel_freqs == b.channel_freqs && a.ch_widths == b.ch_widths &&
    a.total_bandwidths == b.total_bandwidths && a.sidebands == b.sidebands &&
    a.extras == b.extras
Base.hash(a::FrequencySetup, h::UInt) = hash(
    (
        a.name, a.ref_freq, a.channel_freqs, a.ch_widths,
        a.total_bandwidths, a.sidebands, a.extras,
    ),
    hash(:FrequencySetup, h),
)

function Base.show(io::IO, fs::FrequencySetup)
    flo = round(minimum(fs.channel_freqs) / 1.0e9; digits = 3)
    fhi = round(maximum(fs.channel_freqs) / 1.0e9; digits = 3)
    return print(io, "FrequencySetup($(fs.name), $(length(fs.channel_freqs)) IFs, $(flo)–$(fhi) GHz)")
end
