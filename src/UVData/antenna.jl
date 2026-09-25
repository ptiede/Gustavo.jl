const PolTypes = Union{RPol, LPol, XPol, YPol}

"""
    Antenna(; name, station_xyz, mount, nominal_basis, pol_angles = (0.0, 0.0))

Single-antenna record.

- `name`           : Telescope name.
- `station_xyz`    : Geocentric antenna position in meters (ITRS).
- `mount`          : Telescope mount, an `XRadio.AbstractMount`.
- `nominal_basis`  : Labeled feed types `(feed_a, feed_b)` ∈ {RPol,LPol,XPol,YPol}².
- `pol_angles`     : Feed-orientation angles `(pola, polb)` in radians. Element
  type is parametric so callers can pick `Float32`/`Float64`/etc.; defaults to
  `(0.0, 0.0)` (`Float64`).
"""
Base.@kwdef struct Antenna{TName, TXYZ, TMnt <: AbstractMount, TPolAng}
    name::TName
    station_xyz::TXYZ
    mount::TMnt
    nominal_basis::NTuple{2, PolTypes}
    pol_angles::NTuple{2, TPolAng} = (0.0, 0.0)
end

"""
    AntennaTable

Array-of-structs antenna table.

- `antennas`   : `StructArray{Antenna}` of per-antenna records. Property
  access on the table forwards to the underlying StructArray, so
  `tab.name`, `tab.station_xyz`, `tab.mount`, … return per-antenna vectors.
- `array_name` : array name string (read with `array_name(tab)`).
- `extras`     : `NamedTuple` of optional per-antenna columns (e.g.
  `POLCALA`, `POLCALB`, `DIAMETER`, `BEAMFWHM`); read with `extras(tab)`.

The table-level fields are accessed via the helper functions, *not* via
property access — `tab.array_name` would resolve to a per-antenna vector
inside the StructArray (and currently throws), which is the intended
trade-off so all property access goes through the StructArray.
"""
struct AntennaTable{TAnt <: StructArray, TName, TExtras <: NamedTuple}
    antennas::TAnt
    array_name::TName
    extras::TExtras
    function AntennaTable(antennas, array_name, extras::NamedTuple)
        sa = antennas isa StructArray ? antennas : StructArray(antennas)
        return new{typeof(sa), typeof(array_name), typeof(extras)}(sa, array_name, extras)
    end
end

AntennaTable(antennas, array_name) = AntennaTable(antennas, array_name, NamedTuple())

# Property access forwards to the inner StructArray (so `tab.name`, `tab.mount`,
# `tab.nominal_basis`, … all return per-antenna vectors). Table-level fields
# are accessed via `array_name(tab)` / `extras(tab)`.
Base.propertynames(ant::AntennaTable) = propertynames(getfield(ant, :antennas))
Base.getproperty(ant::AntennaTable, name::Symbol) =
    getproperty(getfield(ant, :antennas), name)

Base.length(t::AntennaTable) = length(getfield(t, :antennas))
Base.getindex(t::AntennaTable, i) = getfield(t, :antennas)[i]
Base.iterate(t::AntennaTable, args...) = iterate(getfield(t, :antennas), args...)

array_name(ant::AntennaTable) = getfield(ant, :array_name)
extras(ant::AntennaTable) = getfield(ant, :extras)

function Base.show(io::IO, t::AntennaTable)
    return print(io, "AntennaTable($(array_name(t)), $(length(t)) antennas)")
end

# Structural equality and hashing. Two independently-constructed structs
# with field-equal contents must compare `==` so they dedup in `Dict` and
# the `merge_uvsets` strict-equality check works on independent loads.
Base.:(==)(a::Antenna, b::Antenna) =
    a.name == b.name && a.station_xyz == b.station_xyz &&
    a.mount == b.mount && a.nominal_basis == b.nominal_basis &&
    a.pol_angles == b.pol_angles
Base.isequal(a::Antenna, b::Antenna) =
    isequal(a.name, b.name) && isequal(a.station_xyz, b.station_xyz) &&
    isequal(a.mount, b.mount) && isequal(a.nominal_basis, b.nominal_basis) &&
    isequal(a.pol_angles, b.pol_angles)
Base.hash(a::Antenna, h::UInt) = hash(
    (a.name, a.station_xyz, a.mount, a.nominal_basis, a.pol_angles),
    hash(:Antenna, h),
)

Base.:(==)(a::AntennaTable, b::AntennaTable) =
    getfield(a, :antennas) == getfield(b, :antennas) &&
    getfield(a, :array_name) == getfield(b, :array_name) &&
    getfield(a, :extras) == getfield(b, :extras)
Base.hash(a::AntennaTable, h::UInt) = hash(
    (getfield(a, :antennas), getfield(a, :array_name), getfield(a, :extras)),
    hash(:AntennaTable, h),
)

