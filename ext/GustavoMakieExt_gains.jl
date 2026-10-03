# Implements `plot_gain_phases`, declared in `Gustavo.Calibration`.

import Gustavo.Calibration
using Gustavo.UVData: AntennaName, Feed
using DimensionalData: AbstractDimArray, At, dims, hasdim, lookup, name, otherdims

function _gain_phase_xdim(g::AbstractDimArray)
    (hasdim(g, AntennaName) && hasdim(g, Feed) && ndims(g) == 3) || throw(
        ArgumentError(
            "plot_gain_phases: expected gains over AntennaName, Feed and one more " *
                "dimension; got $(map(name, dims(g))). Select the rest away, e.g. gains(sol; Ti = 1)."
        )
    )
    return only(otherdims(g, (AntennaName, Feed)))
end

function Calibration.plot_gain_phases(parent, g::AbstractDimArray)
    xdim = _gain_phase_xdim(g)
    x = collect(lookup(xdim))
    for (row, ant) in enumerate(lookup(g, AntennaName))
        axrow = Axis[]
        for (col, feed) in enumerate(lookup(g, Feed))
            ax = Axis(
                parent[row, col];
                xlabel = string(name(xdim)), ylabel = string(ant),
                title = (row == 1 ? string("feed ", feed, " phase (rad)") : ""),
            )
            scatter!(ax, x, collect(angle.(g[AntennaName(At(ant)), Feed(At(feed))])); markersize = 5)
            push!(axrow, ax)
        end
        for ax in axrow[2:end]
            linkxaxes!(axrow[1], ax)
            linkyaxes!(axrow[1], ax)
        end
    end
    return parent
end

function Calibration.plot_gain_phases(g::AbstractDimArray)
    _gain_phase_xdim(g)
    fig = Figure(size = (480 * size(g, Feed) + 40, 220 * size(g, AntennaName) + 40))
    Calibration.plot_gain_phases(fig, g)
    return fig
end
