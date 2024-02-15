struct Mutator <: AbstractMutator 
    condition::Function 
    retriever::AbstractRetriever # returns iterable of individuals to mutate
    operator::Function  # returns iterable of mutated individuals,
                        # does not update the state
    updater::AbstractUpdater   # adds mutated individuals to the respective 
                        # populations
end
