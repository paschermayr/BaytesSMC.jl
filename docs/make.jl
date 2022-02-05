using BaytesSMC
using Documenter

DocMeta.setdocmeta!(BaytesSMC, :DocTestSetup, :(using BaytesSMC); recursive=true)

makedocs(;
    modules=[BaytesSMC],
    authors="Patrick Aschermayr <p.aschermayr@gmail.com>",
    repo="https://github.com/paschermayr/BaytesSMC.jl/blob/{commit}{path}#{line}",
    sitename="BaytesSMC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://paschermayr.github.io/BaytesSMC.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Introduction" => "intro.md",
    ],
)

deploydocs(;
    repo="github.com/paschermayr/BaytesSMC.jl",
    devbranch="master",
)
