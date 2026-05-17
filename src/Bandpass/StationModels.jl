abstract type AbstractBandpassModel end

struct PerChannelBandpassModel <: AbstractBandpassModel end
struct FlatBandpassModel <: AbstractBandpassModel end
struct DelayBandpassModel <: AbstractBandpassModel end

struct PolynomialBandpassModel <: AbstractBandpassModel
    degree::Int
end

abstract type AbstractTimeSegmentation end
abstract type AbstractFrequencySegmentation end

struct GlobalTimeSegmentation <: AbstractTimeSegmentation end
struct PerScanTimeSegmentation <: AbstractTimeSegmentation end

struct GlobalFrequencySegmentation <: AbstractFrequencySegmentation end

struct BlockFrequencySegmentation <: AbstractFrequencySegmentation
    block_size::Int
    function BlockFrequencySegmentation(block_size::Integer)
        block_size >= 1 || error("block_size must be at least 1")
        return new(Int(block_size))
    end
end

# Each segmented component carries BOTH time and frequency segmentation —
# all three positional fields are required, no defaults. The previous
# `BandpassSegmentation` wrapper and `default_bandpass_segmentation()` are
# gone; specs no longer attach a segmentation, components do.
struct SegmentedBandpassModel{
        M <: AbstractBandpassModel,
        T <: AbstractTimeSegmentation,
        F <: AbstractFrequencySegmentation,
    } <: AbstractBandpassModel
    model::M
    time::T
    frequency::F
end

struct CompositeBandpassModel{C <: Tuple} <: AbstractBandpassModel
    components::C
end

function CompositeBandpassModel(components::Vararg{SegmentedBandpassModel})
    isempty(components) && error("CompositeBandpassModel requires at least one component")
    return CompositeBandpassModel{typeof(components)}(components)
end

# A BandpassSpec wraps either a single `SegmentedBandpassModel` or a
# `CompositeBandpassModel` — atomic models must be wrapped explicitly,
# which forces every spec to have explicit time/frequency segmentation
# on each component.
struct BandpassSpec{M <: AbstractBandpassModel}
    model::M
    function BandpassSpec{M}(model::M) where {M <: AbstractBandpassModel}
        if !(model isa SegmentedBandpassModel || model isa CompositeBandpassModel)
            error(
                "BandpassSpec requires a SegmentedBandpassModel or CompositeBandpassModel; " *
                "got $M. Wrap atomic models with " *
                "SegmentedBandpassModel(model, time_segmentation, frequency_segmentation)."
            )
        end
        return new{M}(model)
    end
end

BandpassSpec(model::AbstractBandpassModel) = BandpassSpec{typeof(model)}(model)

struct FeedBandpassModel{P <: BandpassSpec, A <: BandpassSpec}
    phase::P
    amplitude::A
end

function FeedBandpassModel(; phase::BandpassSpec, amplitude::BandpassSpec)
    return validate_feed_bandpass_model(FeedBandpassModel(phase, amplitude))
end

struct StationBandpassModel{F <: FeedBandpassModel, G <: FeedBandpassModel}
    reference_feed::Int
    reference::F
    relative::G
end

function StationBandpassModel(;
        reference_feed::Integer,
        reference::FeedBandpassModel,
        relative::FeedBandpassModel,
    )
    rf = validate_reference_feed(reference_feed)
    return validate_station_bandpass_model(StationBandpassModel(rf, reference, relative))
end

parameter_count(::PerChannelBandpassModel) = nothing
parameter_count(::FlatBandpassModel) = 0
parameter_count(::DelayBandpassModel) = 1
parameter_count(model::PolynomialBandpassModel) = model.degree
parameter_count(model::SegmentedBandpassModel) = parameter_count(model.model)
parameter_count(model::CompositeBandpassModel) =
    sum(something(parameter_count(component.model), 0) for component in model.components)

function bandpass(::FlatBandpassModel, params, f)
    isempty(params) || error("FlatBandpassModel expects zero parameters")
    return zero(float(f))
end

function bandpass(::DelayBandpassModel, params, f)
    length(params) == 1 || error("DelayBandpassModel expects one parameter")
    return params[1] * f
end

function bandpass(model::PolynomialBandpassModel, params, f)
    length(params) == model.degree || error("PolynomialBandpassModel expects $(model.degree) parameters")
    value = zero(promote_type(typeof(f), eltype(params)))
    for degree in 1:model.degree
        value += params[degree] * f^degree
    end
    return value
end

bandpass(::PerChannelBandpassModel, params, f) = error("PerChannelBandpassModel requires channel-indexed evaluation, not scalar frequency evaluation")
bandpass(::SegmentedBandpassModel, params, f) = error("SegmentedBandpassModel requires segmented/channel-indexed evaluation, not scalar frequency evaluation")
bandpass(::CompositeBandpassModel, params, f) = error("CompositeBandpassModel requires segmented/channel-indexed evaluation, not scalar frequency evaluation")

is_valid_phase_model(::AbstractBandpassModel) = true
is_valid_amplitude_model(::PerChannelBandpassModel) = true
is_valid_amplitude_model(::FlatBandpassModel) = true
is_valid_amplitude_model(::PolynomialBandpassModel) = true
is_valid_amplitude_model(model::SegmentedBandpassModel) = is_valid_amplitude_model(model.model)
is_valid_amplitude_model(model::CompositeBandpassModel) = all(is_valid_amplitude_model(c.model) for c in model.components)
is_valid_amplitude_model(::AbstractBandpassModel) = false

function validate_phase_model(model::AbstractBandpassModel)
    if model isa SegmentedBandpassModel
        validate_phase_model(model.model)
        return model
    elseif model isa CompositeBandpassModel
        isempty(model.components) && error("CompositeBandpassModel must contain at least one component")
        foreach(validate_phase_model, model.components)
        return model
    end
    is_valid_phase_model(model) || error("Unsupported phase model type: $(typeof(model))")
    return model
end

function validate_amplitude_model(model::AbstractBandpassModel)
    if model isa SegmentedBandpassModel
        validate_amplitude_model(model.model)
        return model
    elseif model isa CompositeBandpassModel
        isempty(model.components) && error("CompositeBandpassModel must contain at least one component")
        foreach(validate_amplitude_model, model.components)
        return model
    end
    is_valid_amplitude_model(model) || error("Unsupported amplitude model type for current solver: $(typeof(model))")
    return model
end

function validate_reference_feed(reference_feed::Integer)
    reference_feed in (1, 2) || error("reference_feed must be 1 or 2")
    return Int(reference_feed)
end

# Walk components helper — every spec resolves to a tuple of
# `SegmentedBandpassModel`s, each carrying its own time and frequency.
spec_components(spec::BandpassSpec) = _components(spec.model)
_components(m::SegmentedBandpassModel) = (m,)
_components(m::CompositeBandpassModel) = m.components

# Per-component time queries
is_per_scan(::AbstractTimeSegmentation) = false
is_per_scan(::PerScanTimeSegmentation) = true
component_is_per_scan(c::SegmentedBandpassModel) = is_per_scan(c.time)

# Spec-level: per-scan if ANY component is per-scan
phase_is_per_scan(spec::BandpassSpec) = any(component_is_per_scan, spec_components(spec))
amplitude_is_per_scan(spec::BandpassSpec) = any(component_is_per_scan, spec_components(spec))

# FeedBandpassModel level
phase_is_per_scan(model::FeedBandpassModel) = phase_is_per_scan(model.phase)
amplitude_is_per_scan(model::FeedBandpassModel) = amplitude_is_per_scan(model.amplitude)

# StationBandpassModel level
phase_is_per_scan(model::StationBandpassModel) = phase_is_per_scan(model.reference) || phase_is_per_scan(model.relative)
amplitude_is_per_scan(model::StationBandpassModel) = amplitude_is_per_scan(model.reference) || amplitude_is_per_scan(model.relative)

function phase_is_per_scan(model::StationBandpassModel, feed::Integer)
    feed == model.reference_feed && return phase_is_per_scan(model.reference)
    feed == partner_feed_index(model.reference_feed) && return phase_is_per_scan(model.reference) || phase_is_per_scan(model.relative)
    error("feed must be 1 or 2")
end

function amplitude_is_per_scan(model::StationBandpassModel, feed::Integer)
    feed == model.reference_feed && return amplitude_is_per_scan(model.reference)
    feed == partner_feed_index(model.reference_feed) && return amplitude_is_per_scan(model.reference) || amplitude_is_per_scan(model.relative)
    error("feed must be 1 or 2")
end

# α refactor (commit 2): mixed time segmentations within a single spec
# are now permitted. The solver routes global-time components into the
# template solve and per-scan-time components into the per-scan solve
# (which fits a deviation against the template).
function validate_spec_has_components(spec::BandpassSpec)
    isempty(spec_components(spec)) && error("BandpassSpec must have at least one component")
    return spec
end

function validate_feed_bandpass_model(model::FeedBandpassModel)
    validate_phase_model(model.phase.model)
    validate_amplitude_model(model.amplitude.model)
    validate_spec_has_components(model.phase)
    validate_spec_has_components(model.amplitude)
    return model
end

function validate_station_bandpass_model(model::StationBandpassModel)
    validate_reference_feed(model.reference_feed)
    validate_feed_bandpass_model(model.reference)
    validate_feed_bandpass_model(model.relative)
    return model
end

# Returns the time segmentation of a spec when all components share one,
# else `nothing`. Mixed-time specs were forbidden in commit 1 and are
# now allowed; the solver branches per-component on its own.
function spec_time_segmentation(spec::BandpassSpec)
    components = spec_components(spec)
    isempty(components) && error("BandpassSpec has no components")
    first_time = typeof(components[1].time)
    for c in Iterators.drop(components, 1)
        typeof(c.time) === first_time || return nothing
    end
    return components[1].time
end

bandpass_model_label(::PerChannelBandpassModel) = "per_channel"
bandpass_model_label(::FlatBandpassModel) = "flat"
bandpass_model_label(::DelayBandpassModel) = "delay"
bandpass_model_label(model::PolynomialBandpassModel) = string("poly", model.degree)

frequency_segmentation_label(::GlobalFrequencySegmentation) = "global"
frequency_segmentation_label(seg::BlockFrequencySegmentation) = string("block", seg.block_size)

time_segmentation_label(::GlobalTimeSegmentation) = "global"
time_segmentation_label(::PerScanTimeSegmentation) = "per_scan"

function bandpass_model_label(model::SegmentedBandpassModel)
    inner = bandpass_model_label(model.model)
    freq_label = frequency_segmentation_label(model.frequency)
    return freq_label == "global" ? inner : string(inner, "@", freq_label)
end
bandpass_model_label(model::CompositeBandpassModel) = join(bandpass_model_label.(model.components), "+")

# Spec-level summary used by station_model_summary printing.
spec_label(spec::BandpassSpec) = bandpass_model_label(spec.model)
function spec_time_label(spec::BandpassSpec)
    seg = spec_time_segmentation(spec)
    seg === nothing || return time_segmentation_label(seg)
    # Mixed-time spec — list the components' time labels in order.
    return join((time_segmentation_label(c.time) for c in spec_components(spec)), "+")
end

reference_feed_label(reference_feed::Integer) = string(reference_feed)
partner_feed_index(reference_feed::Integer) = 3 - reference_feed

segmentation_block_size(seg::BlockFrequencySegmentation) = seg.block_size
