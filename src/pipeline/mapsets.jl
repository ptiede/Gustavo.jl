"""
    mapsets(f, ps::ProcessingSet; exec = ExecutionConfig()) -> Vector
    mapsets(f, groups::AbstractDict{<:Any, <:ProcessingSet}; exec = ExecutionConfig()) -> Vector

Apply `f` to each unit of data read into memory, and return `f`'s results in
unit order: `map(u -> f(Gustavo.materialize(u)), values(src))`, with the units
scheduled across tasks and progress reported. The units of `ps` are its
Measurement Sets; the units of `groups`, a `groupby` result such as
`groupby(ps, ByScan())`, are its processing sets. The fringe and adhoc steps
need every spectral window of a scan, so a body that fits them takes scan
groups.

Each unit is read with arrays of its own ([`Gustavo.materialize`](@ref UVData.materialize)), so `f` may
modify it in place (`calibrate!`) without changing the source or another unit. A unit is held
only while `f` runs, unless `f` returns it: return solutions or reduced data,
and write full-size data out from inside `f`.

Units run on `exec`'s `outer_executor`, largest first, and a unit's
Measurement Sets are read on its `inner_executor`. With more than one task,
`f` must be safe to run concurrently. `exec.progress` receives
`(:mapsets, done, total)` as units finish.

```julia
gauge = PinAntenna(1)
sols = mapsets(groupby(ps, ByScan())) do g
    fr = fit(BaselineFringeFit(; gauge), g)
    calibrate!(fr, g; flag_bad = false, apply_flags = false)
    return (; fr, ad = fit(AdhocPhase(; gauge), g))
end
```
"""
function mapsets(f, ps::XRadio.ProcessingSet; exec::ExecutionConfig = ExecutionConfig())
    units = collect(values(ps))
    sizes = [_group_bytes((ms,)) for ms in units]
    return _map_units(f, units, sizes, exec)
end

function mapsets(
        f, groups::AbstractDict{<:Any, <:XRadio.ProcessingSet};
        exec::ExecutionConfig = ExecutionConfig(),
    )
    units = collect(values(groups))
    sizes = [_group_bytes(values(g)) for g in units]
    return _map_units(f, units, sizes, exec)
end

function _map_units(f, units, sizes, exec::ExecutionConfig)
    work(unit) = f(_read_unit(unit, inner_executor(exec)))
    return _map_groups(work, units, sizes, exec; stage = :mapsets)
end

_read_unit(ms::XRadio.MeasurementSet, executor) = materialize(ms)

_read_unit(group::XRadio.ProcessingSet, executor) = _read_group(group, executor, materialize)
