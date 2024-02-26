export Assertor, PopSizeAssertor
struct Assertor <: AbstractAssertor
    condition::Function
    retriever::AbstractRetriever
    operator::Function
    updater::Function
    time::Bool
end
Assertor(condition::Function,
         retriever::AbstractRetriever,
         operator::Function) =
    Assertor(condition, retriever, operator, noop, false)

PopSizeAssertor(size::Int,
                pop_ids::Vector{String}=String[]) =
    Assertor(always,
             PopulationRetriever(pop_ids),
             map(map((s, p)->@assert length(p.individuals) == size)))
