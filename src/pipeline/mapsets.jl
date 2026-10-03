"""
    mapsets(f, src; exec = ExecutionConfig()) -> OrderedDict

Apply `f` to each unit of data in `src`, read into memory, and return `f`'s
results under the units' keys, in `src`'s order: `k => f(Gustavo.materialize(u))`
for each `k => u` of `src`, with the units scheduled across tasks and progress
reported. `src` is an ordered keyed collection whose values are Measurement
Sets or processing sets: a `ProcessingSet` (its Measurement Sets, keyed by
name), a `groupby` result such as `groupby(ps, ByScan())` (its processing sets,
keyed by label), or an `OrderedDict` of either. The fringe and adhoc steps need
every spectral window of a scan, so a body that fits them takes scan groups.

Each unit is read with arrays of its own ([`Gustavo.materialize`](@ref UVData.materialize)),
so `f` may modify it in place (`calibrate!`) without changing the source or
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
