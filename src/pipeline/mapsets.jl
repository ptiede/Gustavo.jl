"""
    mapsets(f, src; exec = ExecutionConfig()) -> OrderedDict

Apply `f` to each unit of data in `src`, read into memory, and return `f`'s
results under the units' keys, in `src`'s order: `k => f(read(u))` for each
`k => u` of `src`, with the units scheduled across tasks and progress
reported. `src` is an ordered keyed collection whose values are Measurement
Sets or processing sets: a `ProcessingSet` (its Measurement Sets, keyed by
name), a `groupby` result such as `groupby(ps, ByScan())` (its processing sets,
keyed by label), or an `OrderedDict` of either. The fringe and adhoc steps need
every spectral window of a scan, so a body that fits them takes scan groups.

Each unit is read with arrays of its own, copied where the source is already in
memory, so `f` may modify it in place (`calibrate!`) without changing the source or
another unit. A unit is held only while `f` runs, unless `f` returns it: return
solutions or reduced data, and write full-size data out from inside `f`.
Results that are Measurement Sets or processing sets form one processing set
again with `XRadio.ProcessingSet`: `ProcessingSet(out, metadata(ps))` for
Measurement Sets, `ProcessingSet(out)` for processing sets.

Units run on `exec`'s `outer_executor`, largest first, and a unit's
Measurement Sets are read on its `inner_executor`. With more than one task,
`f` must be safe to run concurrently. `exec.progress` receives
`(:mapsets, done, total)` as units finish.

```julia
gauge = PinAntenna(1)
averaged = mapsets(groupby(ps, ByScan())) do g
    fr = fit(BaselineFringeFit(; gauge), g)
    calibrate!(fr, g; flag_bad = false, apply_flags = false)
    return average(g, ByScan())
end
bp = fit(Bandpass(; gauge), ProcessingSet(averaged))
```
"""
function mapsets(f, src; exec::ExecutionConfig = ExecutionConfig())
    units = collect(values(src))
    sizes = [_unit_bytes(u) for u in units]
    work(unit) = f(_read_unit(unit, inner_executor(exec)))
    results = _map_groups(work, units, sizes, exec; stage = :mapsets)
    return OrderedDict(zip(keys(src), results))
end

_unit_bytes(ms::XRadio.MeasurementSet) = _group_bytes((ms,))
_unit_bytes(group::XRadio.ProcessingSet) = _group_bytes(values(group))

_read_unit(ms::XRadio.MeasurementSet, executor) = materialize(ms)

_read_unit(group::XRadio.ProcessingSet, executor) = _read_group(group, executor, materialize)

"""
    write!(out::XRadio.ProcessingSet, unit::XRadio.ProcessingSet) -> out

Write the data variables of each Measurement Set of `unit` into the Measurement
Set of the same name in `out`, a store opened with
`open(ProcessingSet, path; mode = "r+")`. A member holds some of the stored
times and all of every other axis, as a [`mapsets`](@ref) unit of a `groupby`
does; data variables are the layers [`UVData.GUSTAVO_VISIBILITY_SCHEMA`](@ref) does
not call coordinates.

`write!` creates no arrays and writes no metadata, and refuses a member whose
times cover part of a stored chunk, so units with distinct times may be
written from concurrent tasks. Write the template with chunks no longer in time
than the shortest unit, such as `chunks = (; time = 1)`; the store's
consolidated metadata stays valid.

```julia
write(path, ps; data = false, chunks = (; time = 1), schemas = [UVData.GUSTAVO_VISIBILITY_SCHEMA])
out = open(ProcessingSet, path; mode = "r+")
mapsets(groupby(ps, ByScan())) do g
    calibrate!(sol, g)
    write!(out, g)
    return nothing
end
```
"""
function write!(out::XRadio.ProcessingSet, unit::XRadio.ProcessingSet)
    for (name, ms) in pairs(unit)
        haskey(out, name) || throw(
            ArgumentError("the store holds no Measurement Set `$name`; `write!` creates none")
        )
        _write_member!(out[name], ms, name)
    end
    return out
end

const _COORDINATE_LAYERS = Set(c.name for c in GUSTAVO_VISIBILITY_SCHEMA.coords)

function _write_member!(dest, ms, name)
    stored = lookup(dest, Ti)
    position = Dict(t => i for (i, t) in pairs(stored))
    rows = map(lookup(ms, Ti)) do t
        get(() -> throw(ArgumentError("time $t of `$name` is not a time of the store")), position, t)
    end
    for key in keys(ms)
        key in _COORDINATE_LAYERS && continue
        haskey(dest, key) || throw(
            ArgumentError("the store's `$name` has no layer `$key`; `write!` creates none")
        )
        layer = dest[key]
        z = parent(layer)
        z isa Zarr.ZArray && z.writeable || throw(
            ArgumentError("`$key` of `$name` is not a writable stored array; open the store with `mode = \"r+\"`")
        )
        for d in dims(layer)
            d isa Ti && continue
            lookup(layer, d) == lookup(ms[key], d) || throw(
                ArgumentError("`$key` of `$name` must hold all of the stored $(DimensionalData.name(d))")
            )
        end
        _check_whole_chunks(z, DimensionalData.dimnum(layer, Ti), rows, key, name)
        layer[Ti(rows)] .= ms[key]
    end
    return dest
end

# Two writes into one chunk lose data: each reads the chunk, changes its own
# part and writes the whole chunk back.
function _check_whole_chunks(z, axis, rows, key, name)
    wanted = Set(rows)
    for chunk in DiskArrays.eachchunk(z).chunks[axis]
        covered = count(in(wanted), chunk)
        0 < covered < length(chunk) && throw(
            ArgumentError(
                "the times of `$name` cover part of a stored chunk of `$key` " *
                    "($(length(chunk)) times); write the store with chunks no longer " *
                    "in time than a unit, e.g. `chunks = (; time = 1)`"
            )
        )
    end
    return nothing
end
