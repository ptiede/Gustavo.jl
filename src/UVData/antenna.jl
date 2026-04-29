const PolTypes = Union{RPol, LPol, XPol, YPol}

"""
    Mount(parallactic, elevation, offset=0)

Defines the telescope mount type. `parallactic` and `elevation` are the
rotation-rate coefficients applied during parallactic-angle evolution;
`offset` is the feed angular offset (radians for natural use, but any
numeric type works).

Convenience constructors:
- `MountAltAz(offset=0)`: alt-az mount (AIPS `MNTSTA = 0`)
- `MountEquatorial(offset=0)`: equatorial mount (AIPS `MNTSTA = 1`)
- `MountNaismithR(offset=0)`: right Naismith mount (AIPS `MNTSTA = 4`)
- `MountNaismithL(offset=0)`: left Naismith mount (AIPS `MNTSTA = 5`)
"""
struct Mount{A, B, C}
    parallactic::A
    elevation::B
    offset::C
    function Mount(parallactic::A, elevation::B, offset::C = 0.0) where {A, B, C}
        return new{A, B, C}(parallactic, elevation, offset)
    end
end

parallactic_mount(m::Mount) = getfield(m, :parallactic)
elevation_mount(m::Mount) = getfield(m, :elevation)
offset_mount(m::Mount) = getfield(m, :offset)

MountAltAz(offset = 0) = Mount(1, 0, offset)
MountEquatorial(offset = 0) = Mount(0, 0, offset)
MountNaismithR(offset = 0) = Mount(1, 1, offset)
MountNaismithL(offset = 0) = Mount(1, -1, offset)

"""
    Antenna(; name, station_xyz, mount, nominal_basis, response, pol_angles = (0.0, 0.0))

Single-antenna record. The fields are basis-agnostic: the `nominal_basis`
captures the labeled feed type (R/L/X/Y) but real instrumental leakage and
non-orthogonality are carried in `response` (a 2×2 Jones matrix).

- `name`           : Telescope name.
- `station_xyz`    : Antenna position in meters relative to the array center.
- `mount`          : Telescope mount (see `Mount`).
- `nominal_basis`  : Labeled feed types `(feed_a, feed_b)` ∈ {RPol,LPol,XPol,YPol}².
  This is *only* the label written by the antenna table — the actual
  instrumental basis differs from the nominal one whenever there is leakage.
- `pol_angles`     : Feed-orientation angles `(pola, polb)`. Element type is
  parametric so callers can pick `Float32`/`Float64`/etc.; defaults to
  `(0.0, 0.0)` (`Float64`).
- `response`       : Polarization-response Jones matrix relative to the
  nominal basis.
"""
Base.@kwdef struct Antenna{TName, TXYZ, TMnt <: Mount, TPolAng, R}
    name::TName
    station_xyz::TXYZ
    mount::TMnt
    nominal_basis::NTuple{2, PolTypes}
    response::R
    pol_angles::NTuple{2, TPolAng} = (0.0, 0.0)
end

"""
    AntennaTable

Array-of-structs antenna table.

- `antennas`   : `StructArray{Antenna}` of per-antenna records. Property
  access on the table forwards to the underlying StructArray, so
  `tab.name`, `tab.station_xyz`, `tab.mount`, … return per-antenna vectors.
- `array_xyz`  : geocentric array center in meters (read with `array_xyz(tab)`).
- `array_name` : array name string (read with `array_name(tab)`).
- `extras`     : `NamedTuple` of optional per-antenna columns (e.g.
  `POLCALA`, `POLCALB`, `DIAMETER`, `BEAMFWHM`); read with `extras(tab)`.

The table-level fields are accessed via the helper functions, *not* via
property access — `tab.array_name` would resolve to a per-antenna vector
inside the StructArray (and currently throws), which is the intended
trade-off so all property access goes through the StructArray.
"""
struct AntennaTable{TAnt <: StructArray, TXyz, TName, TExtras <: NamedTuple}
    antennas::TAnt
    array_xyz::TXyz
    array_name::TName
    extras::TExtras
    function AntennaTable(antennas, array_xyz, array_name, extras::NamedTuple)
        sa = antennas isa StructArray ? antennas : StructArray(antennas)
        return new{typeof(sa), typeof(array_xyz), typeof(array_name), typeof(extras)}(
            sa, array_xyz, array_name, extras,
        )
    end
end

AntennaTable(antennas, array_xyz, array_name) =
    AntennaTable(antennas, array_xyz, array_name, NamedTuple())

# Property access forwards to the inner StructArray (so `tab.name`, `tab.mount`,
# `tab.nominal_basis`, … all return per-antenna vectors). Table-level fields
# are accessed via `array_xyz(tab)` / `array_name(tab)` / `extras(tab)`.
Base.propertynames(ant::AntennaTable) = propertynames(getfield(ant, :antennas))
Base.getproperty(ant::AntennaTable, name::Symbol) =
    getproperty(getfield(ant, :antennas), name)

Base.length(t::AntennaTable) = length(getfield(t, :antennas))
Base.getindex(t::AntennaTable, i) = getfield(t, :antennas)[i]
Base.iterate(t::AntennaTable, args...) = iterate(getfield(t, :antennas), args...)

array_xyz(ant::AntennaTable) = getfield(ant, :array_xyz)
array_name(ant::AntennaTable) = getfield(ant, :array_name)
extras(ant::AntennaTable) = getfield(ant, :extras)

function Base.show(io::IO, t::AntennaTable)
    return print(io, "AntennaTable($(array_name(t)), $(length(t)) antennas)")
end

# Structural equality and hashing. Two independently-constructed structs
# with field-equal contents must compare `==` so they dedup in `Dict` and
# the `merge_uvsets` strict-equality check works on independent loads.
Base.:(==)(a::Mount, b::Mount) =
    a.parallactic == b.parallactic && a.elevation == b.elevation &&
    a.offset == b.offset
Base.hash(a::Mount, h::UInt) =
    hash((a.parallactic, a.elevation, a.offset), hash(:Mount, h))

Base.:(==)(a::Antenna, b::Antenna) =
    a.name == b.name && a.station_xyz == b.station_xyz &&
    a.mount == b.mount && a.nominal_basis == b.nominal_basis &&
    a.response == b.response && a.pol_angles == b.pol_angles
Base.hash(a::Antenna, h::UInt) = hash(
    (a.name, a.station_xyz, a.mount, a.nominal_basis, a.response, a.pol_angles),
    hash(:Antenna, h),
)

Base.:(==)(a::AntennaTable, b::AntennaTable) =
    getfield(a, :antennas) == getfield(b, :antennas) &&
    getfield(a, :array_xyz) == getfield(b, :array_xyz) &&
    getfield(a, :array_name) == getfield(b, :array_name) &&
    getfield(a, :extras) == getfield(b, :extras)
Base.hash(a::AntennaTable, h::UInt) = hash(
    (
        getfield(a, :antennas), getfield(a, :array_xyz),
        getfield(a, :array_name), getfield(a, :extras),
    ),
    hash(:AntennaTable, h),
)
