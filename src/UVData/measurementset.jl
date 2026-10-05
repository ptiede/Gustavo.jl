# The per-partition accessors, answered from an MSv4 `MeasurementSet`.
#
# Each is rebuilt from the store on every call.

"""
    feed_pairs(ms::XRadio.MeasurementSet) -> Matrix{Tuple{Int, Int}}

The `(feed_a, feed_b)` pair each stored product relates on each baseline,
indexed `[product, baseline]` in `ms`'s own order. A product names one receptor
of each antenna, resolved against that antenna's `polarization_type`: the
letter's position there, or, for an antenna whose receptors are all of the
other basis (`X`/`Y` for an `R`/`L` letter, or the reverse), the letter's
position in its own basis (`R`, `X` first; `L`, `Y` second). Any other letter
throws. The same stored product can so relate different feed pairs on
different baselines.
"""
function feed_pairs(ms::XRadio.MeasurementSet)
    types = XRadio.polarization_types(ms)
    names = XRadio.antennas(ms)
    receptors = [String.(collect(types[:, a])) for a in axes(types, 2)]
    products = XRadio.polarizations(ms)
    for p in products
        length(p) == 2 || throw(ArgumentError("a product label has two receptors, got \"$p\""))
    end
    return [
        (_receptor_feed(p[1], receptors[a], names[a]), _receptor_feed(p[2], receptors[b], names[b]))
            for p in products, (a, b) in _antenna_pairs(ms)
    ]
end

const _BASES = (("R", "L"), ("X", "Y"))

function _receptor_feed(letter::Char, receptors, antenna)
    r = string(letter)
    i = findfirst(==(r), receptors)
    i === nothing || return i
    own = findfirst(basis -> r in basis, _BASES)
    other = own === nothing ? nothing : _BASES[3 - own]
    (other !== nothing && all(in(other), receptors)) && return findfirst(==(r), _BASES[own])
    throw(
        ArgumentError(
            "product letter `$r` names no receptor of antenna `$antenna`, whose receptors are $(join(receptors, ", "))"
        )
    )
end

# The shared feed-pair order of a solver cube, and for each baseline the stored
# product holding each pair: `perm[k, bi]` is the product of baseline `bi` that
# relates `order[k]`. Every baseline must relate each pair exactly once.
function _feed_permutation(pairs::AbstractMatrix{Tuple{Int, Int}})
    order = sort!(unique(vec(pairs)))
    perm = similar(pairs, Int, (eachindex(order), axes(pairs, 2)))
    for bi in axes(pairs, 2)
        col = view(pairs, :, bi)
        (allunique(col) && length(col) == length(order)) || throw(
            ArgumentError(
                "baseline $bi relates feed pairs $(collect(col)); every baseline must relate each of $order once"
            )
        )
        for k in eachindex(order)
            perm[k, bi] = findfirst(==(order[k]), col)
        end
    end
    return order, perm
end

# The baselines of `ms` in `baseline_id` order, as indices into `XRadio.antennas(ms)`.
function _antenna_pairs(ms::XRadio.MeasurementSet)
    slot = Dict(n => i for (i, n) in pairs(XRadio.antennas(ms)))
    return [(slot[a], slot[b]) for (a, b) in XRadio.baselines(ms)]
end

"""
    materialize(ms::MeasurementSet) -> MeasurementSet
    materialize(ps::ProcessingSet) -> ProcessingSet

`ms` or `ps` read into memory with arrays of its own, which no other
Measurement Set shares, so in-place corrections such as `calibrate!` never
reach the source. `read` of in-memory data shares its arrays; this copies
those it would share. What `mapsets` and `calibrate` hand to in-place corrections.
"""
function materialize(ms::XRadio.MeasurementSet)
    out = read(ms)
    for k in keys(out)
        parent(out[k]) === parent(ms[k]) && (out[k] = copy(out[k]))
    end
    return out
end

materialize(ps::XRadio.ProcessingSet) = XRadio.ProcessingSet(
    OrderedDict{Symbol, XRadio.MeasurementSet}(k => materialize(ms) for (k, ms) in pairs(ps)),
    copy(DimensionalData.metadata(ps)),
)
