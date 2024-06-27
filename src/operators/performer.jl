export Performer

@define_op "Performer"
Performer(;kwargs...) = create_op("Performer", 
            retriever=(state::AbstractState, _) -> state.matches,
            operator=(_, matches)->(@assert !isempty(matches) "No matches to perform"; matches),
            updater=ComputeInteractions!(); kwargs...)
