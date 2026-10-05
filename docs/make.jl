using Gustavo
using Documenter

DocMeta.setdocmeta!(Gustavo, :DocTestSetup, :(using Gustavo); recursive = true)

makedocs(;
    modules = [Gustavo],
    authors = "Paul Tiede <ptiede91@gmail.com> and contributors",
    sitename = "Gustavo.jl",
    format = Documenter.HTML(;
        canonical = "https://ptiede.github.io/Gustavo.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Conventions" => "conventions.md",
        "Specifying gain models" => "models.md",
        "Fringe fitting" => "fringe_fitting.md",
        "Fitting under priors" => "priors.md",
        "Authoring a new gain term" => "authoring_terms.md",
        "Authoring a solve step" => "authoring_steps.md",
        "API reference" => [
            "Gustavo" => "api/gustavo.md",
            "Calibration" => "api/calibration.md",
            "Fring" => "api/fring.md",
        ],
        "Internals" => "internals.md",
    ],
)

deploydocs(;
    repo = "github.com/ptiede/Gustavo.jl",
    devbranch = "main",
)
