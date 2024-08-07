export InitializeAllPopulations
"""Samples 1 population from each population creator and puts them into state.populations"""
@define_op "InitializeAllPopulations"
InitializeAllPopulations(;kwargs...) = create_op("InitializeAllPopulations",
          condition=first_gen,
          retriever=PopulationCreatorRetriever(),
          operator=create,
          updater=PopulationAdder!(),kwargs...)
