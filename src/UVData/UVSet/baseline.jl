# Per-baseline `DimStack` views over leaves and `UVSet`s.
#
# The DimStack carries the same layer names as a leaf (`:vis`, `:weights`,
# `:uvw`, `:flag`) with the `Baseline` axis dropped. `vis`/`weights`/`flag`
# have dims `(Frequency, Ti, Pol)`; `:uvw` has dims `(Ti, UVW)`.
# DimStacks support per-layer slicing and DimensionalData selectors out of
# the box (e.g. `bl[:vis][Pol = pol_at("RR")]`), so most downstream tasks
# (radplots, per-channel diagnostics, time-averaging) become one-liners.

"""
    baseline(leaf::AbstractDimTree, bl) -> DimStack

Build a `DimStack` view of one baseline of a single leaf. `bl` may be:

- an antenna-index pair `(a::Int, b::Int)`,
- a name pair `("AA", "AX")`, or
- the dash-joined label `"AA-AX"`.

The returned stack carries layers `:vis`, `:weights`, `:flag` (each
`(Frequency, Ti, Pol)`) and `:uvw` (`(Ti, UVW)`). Throws `KeyError` if
the baseline is absent from this leaf.

```julia
bl   = baseline(leaf, ("AA", "AX"))
amp  = abs.(bl[:vis][Pol = pol_at("RR")])     # (Frequency, Ti)
uvw  = bl[:uvw]                                # (Ti, UVW)
```
"""
function baseline(leaf::DimensionalData.AbstractDimTree, bl)
    bls = baselines(leaf)
    bi = baseline_index(bls, bl)
    bi == 0 && throw(KeyError(bl))
    return _baseline_stack(leaf, bi)
end

# Build the DimStack from a leaf and a resolved baseline slot.
function _baseline_stack(leaf::DimensionalData.AbstractDimTree, bi::Integer)
    vis_l = leaf[:vis]
    w_l = leaf[:weights]
    uvw_l = leaf[:uvw]
    flag_l = leaf[:flag]

    # vis/weights/flag layout is (Frequency, Ti, Baseline, Pol) — slice
    # the Baseline slot using DimensionalData selector so the resulting
    # arrays keep their (Frequency, Ti, Pol) lookups.
    vis_bl = view(vis_l, Baseline(bi))
    w_bl = view(w_l, Baseline(bi))
    flag_bl = view(flag_l, Baseline(bi))
    uvw_bl = view(uvw_l, Baseline(bi))   # (Ti, UVW)

    bls = baselines(leaf)
    a, b = bls.pairs[bi]
    info = DimensionalData.metadata(leaf)
    md = (
        scan_name = info.scan_name,
        source_name = info.source_name,
        spw_name = info.spw_name,
        ant1 = bls.ant1_names[bi],
        ant2 = bls.ant2_names[bi],
        ant1_index = a,
        ant2_index = b,
        label = bls.labels[bi],
        freq_setup = info.freq_setup,
        ra = info.ra,
        dec = info.dec,
    )
    return DimStack((; vis = vis_bl, weights = w_bl, flag = flag_bl, uvw = uvw_bl); metadata = md)
end

"""
    baseline(uvset::UVSet, bl; require_unique_spw = true) -> DimStack

Cross-leaf view of one baseline: walk every leaf containing `bl` and
concatenate along the `Ti` axis. The returned `DimStack` matches the
single-leaf shape (layers `:vis`, `:weights`, `:flag` with dims
`(Frequency, Ti, Pol)`; `:uvw` with `(Ti, UVW)`) but its `Ti` axis spans
the entire observation for that baseline.

By default the function errors when the participating leaves do not all
share the same frequency setup. Pass `require_unique_spw = false` to
opt out of this check (caller becomes responsible for the resulting
mixed-setup stack).

```julia
bl  = baseline(uvset, ("AA", "AX"))
amp = abs.(bl[:vis][Pol = pol_at("RR")])      # (Frequency, all-Ti)
uv  = bl[:uvw]                                # (all-Ti, UVW)
```
"""
function baseline(uvset::UVSet, bl; require_unique_spw::Bool = true)
    parts = baselines_per_scan(uvset, bl)
    isempty(parts) && error(
        "baseline: no leaf contains baseline $(bl); call ",
        "`participating_antennas(leaf)` per leaf to inspect availability.",
    )
    if require_unique_spw
        spws = unique(getfield(DimensionalData.metadata(s), :spw_name) for s in values(parts))
        if length(spws) > 1
            spw_list = join(spws, ", ")
            error(
                "baseline: requested baseline appears in multiple SPWs " *
                    "($spw_list); call `select_partition` first or pass " *
                    "`require_unique_spw=false`.",
            )
        end
    end

    stacks = collect(values(parts))
    length(stacks) == 1 && return stacks[1]
    return _concat_baseline_stacks(stacks)
end

"""
    baselines_per_scan(uvset::UVSet, bl) -> OrderedDict{Symbol, DimStack}

Walk every leaf containing baseline `bl` and return a per-leaf `DimStack`
keyed by partition key (preserving scan structure). Useful when the
caller wants to keep per-scan boundaries — e.g. for diagnostics that
process scans independently.
"""
function baselines_per_scan(uvset::UVSet, bl)
    out = OrderedDict{Symbol, DimStack}()
    for (k, leaf) in DimensionalData.branches(uvset)
        bls = baselines(leaf)
        bi = baseline_index(bls, bl)
        bi == 0 && continue
        out[k] = _baseline_stack(leaf, bi)
    end
    return out
end

# Concatenate per-leaf DimStacks along the Ti axis. We assume Frequency
# and Pol axes are identical across leaves (verified by the SPW check
# upstream); UVW just gets stacked along Ti like the others.
function _concat_baseline_stacks(stacks::AbstractVector{<:DimStack})
    vis_cat = cat((s[:vis] for s in stacks)...; dims = Ti)
    w_cat = cat((s[:weights] for s in stacks)...; dims = Ti)
    flag_cat = cat((s[:flag] for s in stacks)...; dims = Ti)
    uvw_cat = cat((s[:uvw] for s in stacks)...; dims = Ti)
    md_first = DimensionalData.metadata(stacks[1])
    md = (;
        md_first...,
        scan_name = "concat",
        # Drop scan-specific bookkeeping when concatenating across scans;
        # ant1/ant2/label/freq_setup/ra/dec stay since they're invariant.
    )
    return DimStack((; vis = vis_cat, weights = w_cat, flag = flag_cat, uvw = uvw_cat); metadata = md)
end
