# ── Saving a solution as a Zarr store ────────────────────────────────────────
#
# Layout of a store written by `save_solution`:
#
#   /                 attrs: format, pipeline, component labels, step names
#   geometry/         times, scan_of_time, channel_freqs, spw_of_chan and, when
#                     stated, channel_widths; t0, f0 and the name vectors as
#                     attributes
#   components/<label>/  one per SolvedComponent: `data` over the params dims plus
#                     their coordinates; step, path and component as attributes
#   steps/<step>/     that step's diagnostics
#   info/             run-wide diagnostics
#
# A collection of solutions stores its keys in the root's attributes and each
# solution, laid out as above, in a group named by its position (`1/`, `2/`, …).
#
# Plain numeric arrays are Zarr arrays with xarray's `_ARRAY_DIMENSIONS`, as is
# a labeled array of enums, by its integer codes with the enum type named in
# its attributes; everything else is JSON in the attributes, tagged with its
# Julia type so it is rebuilt exactly on load.

import JSON
import Zarr
using DimensionalData: AbstractDimStack, DimStack, basetypeof, layers
using DimensionalData.Lookups: Categorical, Points, span, sampling, order

const _STORE_FORMAT = 2

# Typed even when empty: JSON writes an empty `Vector{Union{}}` as `{}`.
_strings(xs) = String[String(x) for x in xs]

"""
    storage_constructor(x)
    storage_arguments(x)

How [`save_solution`](@ref) writes a value it has no array form for (a gain
term, segmentation, prior, a diagnostic struct): as the type
`storage_constructor(x)` and the arguments `storage_arguments(x)` (a `Tuple` or
`NamedTuple`), rebuilt on load as `storage_constructor(x)(storage_arguments(x)...)`.
The defaults are `typeof(x)` and every field in order, which suits a struct
whose positional constructor takes its fields. A type that does not must extend
both; `save_solution` rebuilds each value as it writes it and throws when the
result is written differently from the original.
"""
storage_constructor(x) = typeof(x)
storage_arguments(x::T) where {T} = NamedTuple{fieldnames(T)}(ntuple(i -> getfield(x, i), fieldcount(T)))

storage_constructor(d::Distribution) = Base.typename(typeof(d)).wrapper
storage_arguments(d::Distribution) = map(_plain_param, Distributions.params(d))
_plain_param(p::AbstractMatrix) = Matrix(p)
_plain_param(p) = p

# ── Type names ───────────────────────────────────────────────────────────────

_type_string(T::DataType) = string(
    join(fullname(parentmodule(T)), '.'), '.', nameof(T),
    isempty(T.parameters) ? "" : "{" * join(map(_param_string, T.parameters), ", ") * "}",
)
# A `UnionAll` is named by its leading fixed parameters (`Dim{:param}`).
function _type_string(T::UnionAll)
    body = Base.unwrap_unionall(T)
    w = Base.typename(body).wrapper
    ps = collect(body.parameters)
    k = something(findfirst(p -> p isa TypeVar, ps), length(ps) + 1) - 1
    name = string(join(fullname(parentmodule(w)), '.'), '.', nameof(w))
    named = k == 0 ? w : w{ps[1:k]...}
    named == T || throw(ArgumentError("cannot name the type $T in a store"))
    return k == 0 ? name : name * "{" * join(map(_param_string, ps[1:k]), ", ") * "}"
end
_type_string(T::Union) = "Union{" * join(map(_type_string, Base.uniontypes(T)), ", ") * "}"
_type_string(T) = throw(ArgumentError("cannot name the type $T in a store"))

_param_string(p::Type) = _type_string(p)
_param_string(p::Union{Symbol, Integer}) = repr(p)
_param_string(p::Tuple) = "(" * join(map(_param_string, p), ", ") * (length(p) == 1 ? ",)" : ")")
_param_string(p) = throw(ArgumentError("cannot name the type parameter $(repr(p)) in a store"))

_resolve_type(s::AbstractString) = _resolve_type_expr(Meta.parse(s))

function _resolve_type_expr(ex)
    if ex isa Expr && ex.head === :.
        return getproperty(_resolve_type_expr(ex.args[1]), ex.args[2].value)
    elseif ex isa Expr && ex.head === :curly
        params = map(_resolve_param, ex.args[2:end])
        ex.args[1] === :Union && return Union{params...}
        return _resolve_type_expr(ex.args[1]){params...}
    elseif ex isa Symbol
        return _root_module(ex)
    end
    throw(ArgumentError("unreadable type name in a store: $ex"))
end

_resolve_param(ex::QuoteNode) = ex.value
_resolve_param(ex::Union{Integer, Bool}) = ex
_resolve_param(ex::Expr) = ex.head === :tuple ? Tuple(map(_resolve_param, ex.args)) : _resolve_type_expr(ex)
_resolve_param(ex) = _resolve_type_expr(ex)

function _root_module(name::Symbol)
    name === :Main && return Main
    name === :Core && return Core
    name === :Base && return Base
    for m in values(Base.loaded_modules)
        nameof(m) === name && return m
    end
    throw(ArgumentError("the store names module `$name`, which is not loaded"))
end

# ── JSON values ──────────────────────────────────────────────────────────────

_kind(kind, T, rest...) = Dict{String, Any}("_kind" => kind, "_type" => _type_string(T), rest...)

_encode(x::Union{Bool, Int64, String, Nothing}) = x
_encode(x::AbstractString) = String(x)
_encode(x::Float64) = isfinite(x) ? x : _kind("value", Float64, "value" => string(x))
_encode(x::Symbol) = _kind("value", Symbol, "value" => String(x))
_encode(x::Real) = _kind("value", typeof(x), "value" => isfinite(x) ? x : string(x))
_encode(x::Enum) = _kind("value", typeof(x), "value" => Integer(x))
_encode(x::Tuple) = Dict{String, Any}("_kind" => "tuple", "items" => Any[_encode(v) for v in x])
_encode(x::NamedTuple) = Dict{String, Any}(
    "_kind" => "namedtuple", "names" => _strings(keys(x)), "items" => Any[_encode(v) for v in x],
)
_encode(x::Array) = _kind("array", eltype(x), "size" => collect(size(x)), "data" => Any[_encode(v) for v in vec(x)])
function _encode(x)
    args = storage_arguments(x)
    d = _kind("struct", storage_constructor(x), "fields" => Any[_encode(v) for v in args])
    args isa NamedTuple && (d["names"] = _strings(keys(args)))
    return d
end

_decode(x) = x
_decode(x::AbstractVector) = map(_decode, x)
function _decode(d::AbstractDict)
    kind = d["_kind"]
    kind == "tuple" && return Tuple(map(_decode, d["items"]))
    kind == "namedtuple" && return NamedTuple{Tuple(Symbol.(d["names"]))}(Tuple(map(_decode, d["items"])))
    T = _resolve_type(d["_type"])
    if kind == "value"
        v = d["value"]
        T === Symbol && return Symbol(v)
        return v isa AbstractString ? parse(T, v) : T(v)
    elseif kind == "array"
        return reshape(T[_decode(v) for v in d["data"]], Tuple(d["size"])...)
    elseif kind == "struct"
        return T(map(_decode, d["fields"])...)
    end
    throw(ArgumentError("unreadable value in a store: kind $(repr(kind))"))
end

# `x` as JSON, checked to rebuild from its JSON text to a value written identically.
function _encode_checked(x, path)
    e = try
        _encode(x)
    catch err
        throw(ArgumentError("cannot save $path: $(sprint(showerror, err))"))
    end
    back = try
        _encode(_decode(JSON.parse(JSON.json(e))))
    catch err
        throw(ArgumentError("cannot save $path: a $(typeof(x)) does not rebuild from its stored form ($(sprint(showerror, err))); extend `storage_constructor`/`storage_arguments` for it"))
    end
    back == e || throw(
        ArgumentError("cannot save $path: a $(typeof(x)) rebuilds to a different value; extend `storage_constructor`/`storage_arguments` for it")
    )
    return e
end

# ── Arrays and labeled arrays ────────────────────────────────────────────────

_is_plain_array(A) = A isa AbstractArray && isconcretetype(eltype(A)) &&
    eltype(A) <: Union{Bool, Base.BitInteger, Base.IEEEFloat, Complex{<:Base.IEEEFloat}}

# Julia's column-major axes are xarray's dims reversed.
function _write_array!(g, name, A::AbstractArray, dimnames)
    Base.require_one_based_indexing(A)
    z = Zarr.zcreate(
        eltype(A), g, name, size(A)...;
        attrs = Dict{String, Any}("_ARRAY_DIMENSIONS" => reverse(_strings(dimnames))),
        chunks = max.(size(A), 1),
    )
    isempty(A) || (z[axes(A)...] = Array(A))
    return z
end

_read_array(z) = z[ntuple(_ -> Colon(), ndims(z))...]

# The spec of one axis of a labeled array; its coordinate arrays are pushed to
# `arrays` as `(name, array, dimnames)`.
function _dim_spec!(arrays, d, path)
    name = String(DimensionalData.name(d))
    l = lookup(d)
    spec = Dict{String, Any}(
        "name" => name, "_type" => _type_string(basetypeof(d)), "order" => _type_string(typeof(order(l))),
    )
    vals = parent(l)
    if l isa Categorical
        spec["lookup"] = "labels"
        spec["values"] = _encode_checked(collect(vals), "$path.$name")
    elseif l isa Sampled && span(l) isa Explicit && sampling(l) == Intervals(Center())
        spec["lookup"] = "intervals"
        push!(arrays, (name, collect(Float64, vals), (name,)))
        push!(arrays, (name * "_bounds", Matrix{Float64}(DimensionalData.val(span(l))), ("bnds", name)))
    elseif l isa Sampled && sampling(l) isa Points && vals isa AbstractRange
        spec["lookup"] = "range"
        spec["values"] = _encode_checked(vals, "$path.$name")
    elseif l isa Sampled && sampling(l) isa Points && _is_plain_array(vals)
        spec["lookup"] = "points"
        push!(arrays, (name, vals, (name,)))
    else
        throw(ArgumentError("cannot save $path: its $name axis has an unsupported $(typeof(l)) lookup"))
    end
    return spec
end

function _read_dim(g, spec)
    D = _resolve_type(spec["_type"])
    name = spec["name"]
    kind = spec["lookup"]
    ord = _resolve_type(spec["order"])()
    if kind == "labels"
        return D(Categorical(_decode(spec["values"]); order = ord))
    elseif kind == "range"
        return D(Sampled(_decode(spec["values"]); order = ord, sampling = Points()))
    elseif kind == "points"
        return D(Sampled(_read_array(g.arrays[name]); order = ord, sampling = Points()))
    elseif kind == "intervals"
        return D(
            Sampled(
                _read_array(g.arrays[name]); order = ord,
                span = Explicit(_read_array(g.arrays[name * "_bounds"])), sampling = Intervals(Center()),
            )
        )
    end
    throw(ArgumentError("unreadable axis $name in a store: lookup $(repr(kind))"))
end

# A group holding the labeled array `A`, with `attrs` besides its own.
function _write_dimarray!(parent_group, key, A::AbstractDimArray, path; attrs = Dict{String, Any}())
    data = Array(parent(A))
    arrays = Tuple{String, AbstractArray, Any}[]
    specs = Any[_dim_spec!(arrays, d, path) for d in DimensionalData.dims(A)]
    own = Dict{String, Any}("kind" => "dimarray", "dims" => specs)
    if eltype(data) <: Enum
        own["enum"] = _type_string(eltype(data))
        push!(arrays, ("data", Integer.(data), [s["name"] for s in specs]))
    elseif _is_plain_array(data)
        push!(arrays, ("data", data, [s["name"] for s in specs]))
    else
        own["data"] = _encode_checked(data, path)
    end
    g = Zarr.zgroup(parent_group, key; attrs = merge(attrs, own))
    for (name, a, dimnames) in arrays
        _write_array!(g, name, a, dimnames)
    end
    return g
end

function _read_dimarray(g)
    attrs = g.attrs
    dims_ = Tuple(_read_dim(g, s) for s in attrs["dims"])
    data = haskey(attrs, "data") ? _decode(attrs["data"]) : _read_array(g.arrays["data"])
    haskey(attrs, "enum") && (data = _resolve_type(attrs["enum"]).(data))
    return DimArray(data, dims_)
end

# ── Diagnostics ──────────────────────────────────────────────────────────────

_is_node(v) = v isa NamedTuple || v isa AbstractDimArray || v isa AbstractDimStack || _is_plain_array(v)

function _write_node!(g, key, v::NamedTuple, path)
    values = Dict{String, Any}()
    for (k, x) in pairs(v)
        _is_node(x) || (values[String(k)] = _encode_checked(x, "$path.$k"))
    end
    sub = Zarr.zgroup(
        g, key; attrs = Dict{String, Any}(
            "kind" => "namedtuple", "keys" => _strings(keys(v)), "values" => values,
        )
    )
    for (k, x) in pairs(v)
        _is_node(x) && _write_node!(sub, String(k), x, "$path.$k")
    end
    return sub
end

_write_node!(g, key, v::AbstractDimArray, path) = _write_dimarray!(g, key, v, path)

function _write_node!(g, key, v::AbstractDimStack, path)
    names = collect(keys(layers(v)))
    sub = Zarr.zgroup(g, key; attrs = Dict{String, Any}("kind" => "dimstack", "layers" => _strings(names)))
    for k in names
        _write_dimarray!(sub, String(k), v[k], "$path.$k")
    end
    return sub
end

_write_node!(g, key, v::AbstractArray, path) = _write_array!(g, key, v, ["$(key)_dim_$i" for i in 1:ndims(v)])

function _read_node(g::Zarr.ZGroup)
    kind = g.attrs["kind"]
    kind == "dimarray" && return _read_dimarray(g)
    if kind == "dimstack"
        names = Tuple(Symbol.(g.attrs["layers"]))
        return DimStack(NamedTuple{names}(Tuple(_read_dimarray(g.groups[String(k)]) for k in names)))
    elseif kind == "namedtuple"
        values = g.attrs["values"]
        ks = g.attrs["keys"]
        vals = map(ks) do k
            haskey(values, k) ? _decode(values[k]) :
                haskey(g.groups, k) ? _read_node(g.groups[k]) : _read_array(g.arrays[k])
        end
        return NamedTuple{Tuple(Symbol.(ks))}(Tuple(vals))
    end
    throw(ArgumentError("unreadable group $(g.path) in a store: kind $(repr(kind))"))
end

# ── Geometry ─────────────────────────────────────────────────────────────────

function _write_geometry!(root, geom::DataGeometry)
    g = Zarr.zgroup(
        root, "geometry"; attrs = Dict{String, Any}(
            "t0" => geom.t0, "f0" => geom.f0, "scan_names" => geom.scan_names,
            "spw_names" => geom.spw_names, "stations" => geom.stations, "nfeed" => geom.nfeed,
        )
    )
    _write_array!(g, "times", geom.times, ("time",))
    _write_array!(g, "scan_of_time", geom.scan_of_time, ("time",))
    _write_array!(g, "channel_freqs", geom.channel_freqs, ("frequency",))
    _write_array!(g, "spw_of_chan", geom.spw_of_chan, ("frequency",))
    isempty(geom.channel_widths) || _write_array!(g, "channel_widths", geom.channel_widths, ("frequency",))
    return g
end

function _read_geometry(g)
    a = g.attrs
    return DataGeometry(;
        times = _read_array(g.arrays["times"]), scan_of_time = _read_array(g.arrays["scan_of_time"]),
        channel_freqs = _read_array(g.arrays["channel_freqs"]), spw_of_chan = _read_array(g.arrays["spw_of_chan"]),
        channel_widths = haskey(g.arrays, "channel_widths") ? _read_array(g.arrays["channel_widths"]) : Float64[],
        t0 = a["t0"], f0 = a["f0"], scan_names = _strings(a["scan_names"]),
        spw_names = _strings(a["spw_names"]), stations = _strings(a["stations"]), nfeed = a["nfeed"],
    )
end

# ── save / load ──────────────────────────────────────────────────────────────

"""
    save_solution(path, sol::CalibrationSolution) -> path
    save_solution(path, sols::AbstractDict{<:Any, <:CalibrationSolution}) -> path

Write `sol` to a new Zarr store at `path`: its geometry, each component's
parameters as a labeled array (with the component's term, segmentation, feed
tying and prior as attributes), each step's diagnostics, the run-wide `info`
and the provenance text. [`load_solution`](@ref) rebuilds it.

The second form writes a collection of solutions, such as the per-unit fits
[`mapsets`](@ref Gustavo.mapsets) returns, one group per entry in order, each key stored with
its type.

Numeric arrays are stored as Zarr arrays with xarray dimension names; other
values are stored as JSON attributes tagged with their Julia type (see
[`storage_constructor`](@ref)). Throws when `path` exists, and when a value
cannot be stored so that it rebuilds exactly; nothing is left at `path` then.
"""
save_solution(path::AbstractString, sol::CalibrationSolution) =
    _save_store(() -> _write_solution!(Zarr.zgroup(path; attrs = _solution_attrs(sol, true)), sol), path)

function save_solution(path::AbstractString, sols::AbstractDict{<:Any, <:CalibrationSolution})
    return _save_store(path) do
        ks = Any[_encode_checked(k, "the key $(repr(k))") for k in keys(sols)]
        root = Zarr.zgroup(
            path; attrs = Dict{String, Any}(
                "gustavo_solution_format" => _STORE_FORMAT, "kind" => "collection", "keys" => ks,
            )
        )
        for (i, sol) in enumerate(values(sols))
            _write_solution!(Zarr.zgroup(root, string(i); attrs = _solution_attrs(sol, false)), sol)
        end
    end
end

function _save_store(write, path)
    ispath(path) && throw(ArgumentError("save_solution: $path exists; remove it or choose another path"))
    try
        write()
    catch
        rm(path; recursive = true, force = true)
        rethrow()
    end
    return path
end

function _solution_attrs(sol::CalibrationSolution, toplevel::Bool)
    attrs = Dict{String, Any}(
        "pipeline" => sol.provenance.pipeline,
        "components" => _strings(map(_label, sol.components)), "steps" => _strings(keys(sol.steps)),
    )
    toplevel && (attrs["gustavo_solution_format"] = _STORE_FORMAT)
    return attrs
end

function _write_solution!(root, sol::CalibrationSolution)
    _write_geometry!(root, sol.geom)
    comps = Zarr.zgroup(root, "components")
    for c in sol.components
        label = _label(c)
        _write_dimarray!(
            comps, label, c.params, label; attrs = Dict{String, Any}(
                "step" => String(c.step), "path" => _strings(c.path),
                "component" => _encode_checked(c.component, label),
            )
        )
    end
    steps = Zarr.zgroup(root, "steps")
    for (name, diag) in sol.steps
        _write_node!(steps, String(name), diag, String(name))
    end
    _write_node!(root, "info", sol.info, "info")
    return nothing
end

"""
    load_solution(path) -> CalibrationSolution or OrderedDict

Read what [`save_solution`](@ref) wrote to the Zarr store at `path`: a
`CalibrationSolution`, or, for a collection, an `OrderedDict` with the saved
keys in the saved order. Stored values are rebuilt by calling the constructors
the store names, so load only stores from a trusted source.
"""
function load_solution(path::AbstractString)
    root = Zarr.zopen(path)
    fmt = get(root.attrs, "gustavo_solution_format", nothing)
    fmt == _STORE_FORMAT || throw(
        ArgumentError(
            isnothing(fmt) ? "load_solution: $path is not a Gustavo solution store" :
                "load_solution: $path has store format $fmt; this Gustavo reads format $_STORE_FORMAT"
        )
    )
    get(root.attrs, "kind", nothing) == "collection" || return _read_solution(root)
    return OrderedDict(
        _decode(k) => _read_solution(root.groups[string(i)]) for (i, k) in enumerate(root.attrs["keys"])
    )
end

function _read_solution(root)
    comps = root.groups["components"]
    components = map(root.attrs["components"]) do label
        g = comps.groups[label]
        SolvedComponent(
            Symbol(g.attrs["step"]), Tuple(Symbol.(g.attrs["path"])),
            _decode(g.attrs["component"]), _read_dimarray(g),
        )
    end
    steps = OrderedDict{Symbol, NamedTuple}(
        Symbol(s) => _read_node(root.groups["steps"].groups[s]) for s in root.attrs["steps"]
    )
    return CalibrationSolution(
        _read_geometry(root.groups["geometry"]), components, steps, _read_node(root.groups["info"]);
        pipeline = root.attrs["pipeline"],
    )
end
