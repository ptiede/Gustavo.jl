# HDF5 caltable I/O for Gustavo calibration solutions. Implements the
# `save_solution_hdf5`/`load_solution_hdf5` stubs declared in `Gustavo.Calibration`.
#
# The file is self-describing and language-neutral: external tools (Python/h5py,
# CASA, …) read the evaluated complex gains + labeled axes + diagnostics directly,
# while Julia round-trips the exact solution losslessly through an embedded
# `Serialization` blob (which external readers simply ignore).
module GustavoHDF5Ext

using HDF5
using Serialization: serialize, deserialize
import DimensionalData
import Gustavo.Calibration: CalibrationSolution, save_solution_hdf5, load_solution_hdf5,
    nchannels, ntimes, _composed_gains, _serializable_transforms

# Write `info` (the solution's or one step's diagnostics NamedTuple) into HDF5
# group `g`, generically: vectors and numbers go straight in; a NamedTuple or
# `DimStack` (e.g. a step's own `timing`) recurses into a subgroup under its
# own key. A third-party step's custom diagnostics round-trip with no changes
# needed here — anything else (a bare struct, e.g. an estimator's config) is
# silently skipped, same as before.
function _write_info!(g, info)
    for k in keys(info)
        v = info[k]
        if v isa AbstractVector
            g[String(k)] = collect(v)
        elseif v isa Number
            g[String(k)] = v
        elseif v isa NamedTuple || v isa DimensionalData.AbstractDimStack
            _write_info!(create_group(g, String(k)), v)
        end
    end
    return nothing
end

function save_solution_hdf5(
        path::AbstractString, sol::CalibrationSolution;
        gains::Bool = true, time_block::Integer = 1024,
    )
    geom = sol.geom
    nchan, ntime, nant = nchannels(geom), ntimes(geom), sol.steps[1].layout.nant
    h5open(path, "w") do f
        attrs = attributes(f)
        attrs["format"] = "GustavoCalibrationSolution"
        attrs["version"] = 2
        attrs["gains_layout"] = "(channel, time, antenna, feed)"
        attrs["gain_convention"] = "V_corr = V / (g_a * conj(g_b)); weight *= abs2(g_a * g_b)"
        attrs["nant"] = nant
        attrs["nchan"] = nchan
        attrs["ntime"] = ntime

        ax = create_group(f, "axes")
        ax["channel_freq_hz"] = collect(geom.channel_freqs)
        ax["time"] = collect(geom.times)
        ax["scan_of_time"] = collect(geom.scan_of_time)
        ax["spw_of_chan"] = collect(geom.spw_of_chan)
        ax["f0_hz"] = geom.f0
        ax["t0"] = geom.t0
        isempty(geom.scan_names) || (ax["scan_names"] = collect(geom.scan_names))
        isempty(geom.spw_names) || (ax["spw_names"] = collect(geom.spw_names))

        ig = create_group(f, "info")
        _write_info!(ig, sol.info)
        sg = create_group(ig, "steps")
        for s in sol.steps
            _write_info!(create_group(sg, String(s.name)), s.info)
        end

        if gains
            blk = min(max(Int(time_block), 1), max(ntime, 1))
            dims = (nchan, ntime, nant, 2)
            ch = (nchan, min(blk, max(ntime, 1)), nant, 2)
            gr = create_dataset(f, "gain/real", Float32, dims; chunk = ch, shuffle = true, deflate = 4)
            gi = create_dataset(f, "gain/imag", Float32, dims; chunk = ch, shuffle = true, deflate = 4)
            t = 1
            while t <= ntime
                t2 = min(t + blk - 1, ntime)
                g = _composed_gains(sol, 1:nchan, t:t2)   # (nchan, block, nant, 2)
                gr[:, t:t2, :, :] = Float32.(real.(g))
                gi[:, t:t2, :, :] = Float32.(imag.(g))
                t = t2 + 1
            end
        end

        # Lossless Julia round-trip: the model carries Julia types HDF5 can't
        # represent natively, so embed the Serialization bytes (external readers
        # ignore this dataset and use `gain/*` + `axes/*`). The blob must carry
        # `transforms`/`postcal` too: they are part of the correction
        # `calibrate(sol, uvset)` reproduces, so a solution reloaded without them
        # would apply strictly less than it was fit with.
        buf = IOBuffer()
        serialize(
            buf, (;
                version = 5, sol.steps, sol.geom, sol.info,
                transforms = _serializable_transforms(sol.transforms),
                postcal = _serializable_transforms(sol.postcal),
            ),
        )
        jg = create_group(f, "julia")
        jg["blob"] = take!(buf)
    end
    return path
end

function load_solution_hdf5(path::AbstractString)
    w = h5open(path, "r") do f
        haskey(f, "julia") && haskey(f["julia"], "blob") ||
            error("load_solution_hdf5: $path has no julia/blob (not written by Gustavo, or gains-only export)")
        deserialize(IOBuffer(read(f["julia"]["blob"])))
    end
    w.version == 5 || error(
        "load_solution_hdf5: unsupported julia/blob version $(w.version) — saved by an " *
            "incompatible Gustavo (the solution shape changed); re-solve to produce a " *
            "current file.",
    )
    return CalibrationSolution(w.steps, w.geom, w.info; transforms = w.transforms, postcal = w.postcal)
end

end # module
