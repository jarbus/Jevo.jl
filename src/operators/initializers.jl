export InitializeAllPopulations
# ====================
# Initializers are used produce objects from creators and put them 
# into the state.
# ======================

"""Samples 1 population from each population creator and puts them into state.populations"""
struct InitializeAllPopulations <: AbstractOperator
    condition::Function        
    retriever::AbstractRetriever
    operator::Function      
    updater::AbstractUpdater
    rng::Nothing
    data::Vector{AbstractData}
    time::Bool
end
function InitializeAllPopulations(;time::Bool=false)
    condition = first_gen
    retriever = PopulationCreatorRetriever()
    operator = create 
    updater = PopulationAdder()
    rng = nothing
    InitializeAllPopulations(condition, retriever, operator, updater, rng, AbstractData[], time)
end
