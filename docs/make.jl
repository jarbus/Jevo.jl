using Documenter
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
        "Examples" => "examples.md",
        "API" => "api.md",
    ],


)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
