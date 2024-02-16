test:
    julia --project=. -e 'using Pkg; Pkg.test()'
rerun:
    find . -name "*.jl" | entr julia test/runtests.jl
