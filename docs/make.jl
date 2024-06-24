using Documenter
using Jevo

makedocs(
    sitename = "Jevo",
    format = Documenter.HTML(),
    modules = [Jevo],
    checkdocs=:exports,
    pages = [
        "Home" => "index.md",
        "Overview" => "overview.md",
        "Operators" => "operators.md",
        "Examples" => "examples.md",
    ],


)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
