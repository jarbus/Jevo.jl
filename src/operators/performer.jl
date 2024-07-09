export Performer

"""
            Performer <: AbstractOperator

Runs all matches in state.matches, and adds interactions to the individuals in the matches. This operator is intended to be used after a MatchMaker has created matches.
"""
@define_op "Performer"
Performer(;kwargs...) = create_op("Performer", 
            retriever=(state::AbstractState, _) -> state.matches,
            operator=(_, matches)->(@assert !isempty(matches) "No matches to perform"; matches),
            updater=ComputeInteractions!(); kwargs...)
