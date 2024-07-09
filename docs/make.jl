using Documenter
using XPlot
using Jevo

makedocs(
    sitename = "Jevo",
    format = Documenter.HTML(),
    modules = [Jevo],
    pages = [
        "Home" => "index.md",
        "Overview" => "overview.md",
        "Operators" => "operators.md",
        "Phylogeny" => "phylogeny.md",
        "Miscellaneous" => "miscellaneous.md",
        "API" => "api.md",
    ],
    warnonly=true,


)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/jarbus/Jevo.jl.git",
)
