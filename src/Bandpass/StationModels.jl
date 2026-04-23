abstract type AbstractBandpassModel end

struct PerChannelBandpassModel <: AbstractBandpassModel end
struct FlatBandpassModel <: AbstractBandpassModel end
struct DelayBandpassModel <: AbstractBandpassModel end

struct PolynomialBandpassModel <: AbstractBandpassModel
    degree::Int
end

abstract type AbstractTimeSegmentation end
abstract type AbstractFrequencySegmentation end

struct SegmentedBandpassModel{M<:AbstractBandpassModel,F<:AbstractFrequencySegmentation} <: AbstractBandpassModel
    model::M
    segmentation::F
end

struct CompositeBandpassModel{C<:Tuple} <: AbstractBandpassModel
    components::C
end

struct GlobalTimeSegmentation <: AbstractTimeSegmentation end
struct PerScanTimeSegmentation <: AbstractTimeSegmentation end

struct GlobalFrequencySegmentation <: AbstractFrequencySegmentation end

struct BlockFrequencySegmentation <: AbstractFrequencySegmentation
    block_size::Int
end

BlockFrequencySegmentation(; block_size=1) =
    BlockFrequencySegmentation(block_size)

struct BandpassSegmentation{T<:AbstractTimeSegmentation,F<:AbstractFrequencySegmentation}
    time::T
    frequency::F
end


struct FeedBandpassModel{P<:AbstractBandpassModel,A<:AbstractBandpassModel}
    phase::P
    amplitude::A
end

struct StationBandpassModel{F<:FeedBandpassModel,G<:FeedBandpassModel}
    reference_feed::Int
    reference::F
    relative::G
    segmentation::BandpassSegmentation
end

parameter_count(::PerChannelBandpassModel) = nothing
parameter_count(::FlatBandpassModel) = 0
parameter_count(::DelayBandpassModel) = 1
parameter_count(model::PolynomialBandpassModel) = model.degree
parameter_count(model::SegmentedBandpassModel) = parameter_count(model.model)
parameter_count(model::CompositeBandpassModel) = sum(something(parameter_count(component.model), 0) for component in model.components)

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
is_valid_amplitude_model(model::CompositeBandpassModel) = all(is_valid_amplitude_model(component.model) for component in model.components)
is_valid_amplitude_model(::AbstractBandpassModel) = false

function validate_phase_model(model::AbstractBandpassModel)
    if model isa SegmentedBandpassModel
        validate_phase_model(model.model)
        validate_frequency_segmentation(model.segmentation)
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
        validate_frequency_segmentation(model.segmentation)
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

function validate_feed_bandpass_model(model::FeedBandpassModel)
    validate_phase_model(model.phase)
    validate_amplitude_model(model.amplitude)
    return model
end

is_per_scan(::AbstractTimeSegmentation) = false
is_per_scan(::PerScanTimeSegmentation) = true

bandpass_model_label(::PerChannelBandpassModel) = "per_channel"
bandpass_model_label(::FlatBandpassModel) = "flat"
bandpass_model_label(::DelayBandpassModel) = "delay"
bandpass_model_label(model::PolynomialBandpassModel) = string("poly", model.degree)
frequency_segmentation_label(::GlobalFrequencySegmentation) = "global"
frequency_segmentation_label(segmentation::BlockFrequencySegmentation) = string("block", segmentation.block_size)
bandpass_model_label(model::SegmentedBandpassModel) = string(bandpass_model_label(model.model), "@", frequency_segmentation_label(model.segmentation))
bandpass_model_label(model::CompositeBandpassModel) = join(bandpass_model_label.(model.components), "+")
effective_bandpass_model_label(model::SegmentedBandpassModel, default_segmentation::AbstractFrequencySegmentation) = bandpass_model_label(model)
effective_bandpass_model_label(model::CompositeBandpassModel, default_segmentation::AbstractFrequencySegmentation) = bandpass_model_label(model)
function effective_bandpass_model_label(model::AbstractBandpassModel, default_segmentation::AbstractFrequencySegmentation)
    label = bandpass_model_label(model)
    return frequency_segmentation_label(default_segmentation) == "global" ? label : string(label, "@", frequency_segmentation_label(default_segmentation))
end
reference_feed_label(reference_feed::Integer) = string(reference_feed)
partner_feed_index(reference_feed::Integer) = 3 - reference_feed

time_segmentation_label(::GlobalTimeSegmentation) = "global"
time_segmentation_label(::PerScanTimeSegmentation) = "per_scan"

segmentation_block_size(segmentation::BlockFrequencySegmentation) = segmentation.block_size

function validate_time_segmentation(segmentation::AbstractTimeSegmentation)
    return segmentation
end

validate_frequency_segmentation(::GlobalFrequencySegmentation) = GlobalFrequencySegmentation()

function validate_frequency_segmentation(segmentation::BlockFrequencySegmentation)
    segmentation.block_size >= 1 || error("block_size must be at least 1")
    return segmentation
end

function validate_segmentation(segmentation::BandpassSegmentation)
    validate_time_segmentation(segmentation.time)
    validate_frequency_segmentation(segmentation.frequency)
    return segmentation
end

function FeedBandpassModel(; phase=PerChannelBandpassModel(), amplitude=PerChannelBandpassModel())
    return validate_feed_bandpass_model(FeedBandpassModel(validate_phase_model(phase), validate_amplitude_model(amplitude)))
end

SegmentedBandpassModel(model::AbstractBandpassModel) =
    SegmentedBandpassModel(model, GlobalFrequencySegmentation())

CompositeBandpassModel(components::Vararg{SegmentedBandpassModel}) =
    CompositeBandpassModel{typeof(components)}(components)

model_components(model::SegmentedBandpassModel, default_segmentation::AbstractFrequencySegmentation) = (model,)
model_components(model::CompositeBandpassModel, default_segmentation::AbstractFrequencySegmentation) = model.components
model_components(model::AbstractBandpassModel, default_segmentation::AbstractFrequencySegmentation) =
    (SegmentedBandpassModel(model, default_segmentation),)

function validate_station_bandpass_model(model::StationBandpassModel)
    validate_reference_feed(model.reference_feed)
    validate_feed_bandpass_model(model.reference)
    validate_feed_bandpass_model(model.relative)
    validate_segmentation(model.segmentation)
    return model
end

function StationBandpassModel(; reference_feed=1,
    reference=FeedBandpassModel(), relative=FeedBandpassModel(),
    segmentation=BandpassSegmentation(GlobalTimeSegmentation(), GlobalFrequencySegmentation()))
    segmentation_spec = validate_segmentation(segmentation)
    return validate_station_bandpass_model(StationBandpassModel(
        validate_reference_feed(reference_feed),
        validate_feed_bandpass_model(reference),
        validate_feed_bandpass_model(relative),
        segmentation_spec))
end
