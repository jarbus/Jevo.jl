export Performer

@define_op "Performer"
Performer(;kwargs...) = create_op("Performer", 
                                 retriever=(state::AbstractState) -> state.matches,
                                 updater=ComputeInteractions(); kwargs...)
