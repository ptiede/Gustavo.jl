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
    ],
)

deploydocs(;
    repo = "github.com/ptiede/Gustavo.jl",
    devbranch = "main",
)
