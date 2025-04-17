#= @testset "checkpoint" begin =#
#==#
#=     checkpointname = "test-checkpoint.jls" =#
#=     state = State("", rng, AbstractCreator[], [Checkpointer(checkpointname, interval=5)], counters=default_counters()) =#
#=     run!(state, 10) =#
#=     @test generation(state) == 11 =#
#=     state = restore_from_checkpoint(checkpointname) =#
#=     @test generation(state) == 6 =#
#= end =#

@testset "neural restore" begin
  n_inds = 4
  k=2
  env_creator = Creator(MaxLogits, (;n=4))
  mrs = (0.1f0, 0.01f0)

  counters = default_counters()
  gene_counter = find(:type, AbstractGene, counters)

  gc = Creator(Delta, Creator(JevoChain, (rng, gene_counter, [
        (Jevo.Dense, (dims=(4, 1), σ=identity))
    ])))

PrintInteractions(;kwargs...) = create_op("Reporter";
    retriever=get_individuals,
    operator=(s,is)-> (m=StatisticalMeasurement("Sum", [sum(int.score for int in ind.interactions) for ind in is], generation(s)); @info m;), kwargs...)

  developer = Creator(Model)
  pop_creator = Creator(Population, ("p", n_inds, PassThrough(gc), PassThrough(developer), counters))
  state = State("", rng, [pop_creator, env_creator], 
                [ 
                  LoadCheckpoint(),
                  InitializeAllPopulations(),
                  InitializePhylogeny(),
                  InitializeDeltaCache(),
                  SoloMatchMaker(),
                  Performer(), 
                  PrintInteractions(),
                  ScalarFitnessEvaluator(),
                  TruncationSelector(k),
                  CloneUniformReproducer(n_inds),

                  UpdatePhylogeny(),
                  UpdateParentsAcrossAllWorkers(),
                  ClearCurrentGenWeights(),

                  NBackMutator(n_back=1000, mrs=mrs),
                  UpdateDeltaCache(),
                  ClearInteractionsAndRecords(),
              ], counters=counters, checkpoint_interval = 5)
  run!(state, 10)
  run!(state, 10)
end
