# Whole-set functions on an MSv4 `ProcessingSet`, answered by walking its
# Measurement Sets through the per-partition accessors.

"""
    freq_setup(ps::XRadio.ProcessingSet) -> FrequencySetup

Single-SPW shorthand: returns the unique `FrequencySetup` if every partition
shares one, otherwise throws `ArgumentError`. Multi-SPW callers should read
each partition's `freq_setup` individually.
"""
function freq_setup(data::XRadio.ProcessingSet)
    setups = unique(freq_setup(ms) for ms in values(data))
    n = length(setups)
    n == 1 && return setups[1]
    n == 0 && throw(ArgumentError("the set has no partitions; no frequency setup"))
    throw(
        ArgumentError(
            "the set has $(n) distinct frequency setups; " *
                "use freq_setup(partition) on each partition",
        )
    )
end
