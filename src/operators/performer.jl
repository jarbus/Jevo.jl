export Performer

@define_op "Performer"
Performer(;batch_size=10, kwargs...) = create_op("Performer", 
            retriever=(state::AbstractState) -> state.matches,
            operator=(_, matches)->(@assert !isempty(matches) "No matches to perform"; matches),
            updater=ComputeInteractions!(batch_size); kwargs...)
