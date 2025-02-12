@testset "numbers game unit and integration" begin
  # creators
  n_dims = 2
  n_inds = 2
  n_species = 2
  n_pops = 2
  counters = default_counters()
  ng_gc = ng_genotype_creator = Creator(VectorGenotype, (n=n_dims,rng=rng))
  ng_developer = Creator(VectorPhenotype)
  # genotypes
  genotype = ng_genotype_creator()
  @test length(genotype.numbers) == n_dims
  # phenotypes
  phenotype = develop(ng_developer, genotype)
  @test phenotype.numbers == genotype.numbers
  # Test that Individual has randomly generated data
  @test Individual(counters, ng_genotype_creator, ng_developer).genotype.numbers |> sum != 0
  # Population
  @test Population("ng", n_inds, ng_genotype_creator, ng_developer, counters).individuals |> length == n_inds
  single_pop_creator = Creator(Population, ("ng", n_inds, PassThrough(ng_genotype_creator), PassThrough(ng_developer), counters))
  # Composite population
  comp_pop = CompositePopulation("species", [("p$i", n_inds, ng_gc, ng_developer) for i in 1:n_species], counters)
  comp_pop_creator = Creator(CompositePopulation, ("species", [("p$i", n_inds, ng_gc, ng_developer)
                                                               for i in 1:n_species], counters))
  comp_pop = comp_pop_creator()
  @test comp_pop.populations |> length == n_species
  @test all(length(p.individuals) == n_inds for p in comp_pop.populations)
  # Create composite composite population, 2 deep binary tree
  comp_comp_pop = CompositePopulation("ecosystem",
                      [CompositePopulation("composite$i",
                          [("p$i", n_inds, ng_gc, ng_developer) for i in 1:n_species],
                      counters)
                  for i in 1:n_pops])
  # Create CompositePopulation creator
  """
      ecosystem
      ├── composite1
      │   ├── pop1: ind1a1, ind1a2
      │   └── pop2: ind1b1, ind1b2
      └── composite2
          ├── pop3: ind2a1, ind2a2
          └── pop4: ind2b1, ind2b2
  """
  pid = 0
  comp_comp_pop_creator = 
      Creator(CompositePopulation, 
          ("ecosystem", 
           [Creator(CompositePopulation,
                  ("composite$i",
                   [("p$(pid+=1)", n_inds, ng_gc, ng_developer) for p in 1:n_species],
                  counters))
          for i in 1:n_pops]))
  comp_comp_pop = comp_comp_pop_creator()
  pop_initializer = InitializeAllPopulations()
  state = State("", rng, [comp_comp_pop_creator], [pop_initializer], counters=counters)
  @testset "PopulationCreatorRetriever" begin
      pops = [pop_creator() for pop_creator in pop_initializer.retriever(state)]
      @test length(pops) == 1
      @test pops[1].id == "ecosystem"
      run!(state, 1)
  end
  @testset "Population{Retriever,Updater}" begin
      # PopulationRetriever
      @test state.populations |> length == 1
      @test state.populations[1].id == "ecosystem"
      @test state.populations[1].populations |> length == n_pops
      pops = PopulationRetriever(["ecosystem"])(state)
      @test length(pops) == 1 # requesting ecosystem returns single population
      @test length(pops[1]) == 4
      @test typeof(pops) == Vector{Vector{Population}}
      pops = PopulationRetriever()(state) # return all pops by default
      @test length(pops) == 4
      @test length(pops[1]) == 1
      try
          PopulationRetriever(["nopop"])(state)
      catch
          @test true
      end
      pops = PopulationRetriever(["composite1"])(state)
      @test length(pops) == 1
      @test length(pops[1]) == n_species
      @test typeof(pops) == Vector{Vector{Population}}
      pops = PopulationRetriever(["p2"])(state)
      @test length(pops) == 1
      @test length.(pops) == [1] # p2 contains a single pop, p2
      pops = PopulationRetriever(["p1", "p2"])(state)
      @test length(pops) == n_pops
      @test length(pops[1]) == 1
      @test length(pops[1][1].individuals) == n_inds
  #     # find population
      @test find_population("p1", state).id == "p1"
      @test find_population("p4", state).id == "p4"
      try
          find_population("composite1", state) && @test false
      catch e
          @test true
      end
      try
          find_population("nopop", state) && @test false
      catch e
          @test true
      end
      # PopulationUpdater
      pops = PopulationRetriever(["p1", "p2"])(state)
      @test all([length(p) == 1 for p in pops])
      @test all([length(p[1].individuals) == 2 for p in pops])
  end

  env_creator = Creator(CompareOnOne)
  @testset "env" begin
      env = env_creator()
  end

  # Matchmaker only
  ava = AllVsAllMatchMaker(["composite1", "composite2"])
  solo = SoloMatchMaker(["composite1", "composite2"])
  performer = Performer()
  evaluator = ScalarFitnessEvaluator(["composite1","composite2"])
  k = 1
  selector = TruncationSelector(k)
  selector_p1 = TruncationSelector(k, ["p1"])
  selector_p2_p3 = TruncationSelector(k, ["p2", "p3"])
  reproducer = CloneUniformReproducer(n_inds)
  reproducer_p2_p3 = CloneUniformReproducer(n_inds, ["p2", "p3"])
  mutator = Mutator()
  assertor = PopSizeAssertor(n_inds)
  ind_resetter = ClearInteractionsAndRecords()
  reporter = Reporter(GenotypeSum)

  @testset "Clone" begin
    state = State("",rng, [comp_comp_pop_creator, env_creator], [pop_initializer, ava], counters=default_counters())
    run!(state, 1)
    # Clone an individual
    ind = Jevo.find_population("p1", state).individuals[1]
    clone = Jevo.clone(state, ind)
    @test clone.id != ind.id
    @test clone.generation == 2
  end

  @testset "MatchMaker" begin
      @testset "AllVsAll" begin
          # two pops
          state = State("", rng, [comp_comp_pop_creator, env_creator], [pop_initializer, ava], counters=default_counters())
          @test Jevo.get_creators(AbstractEnvironment, state) |> length == 1
          run!(state, 1)
          n_pairs = n_pops * n_species
          n_unique_pairs = n_pairs * (n_inds^2 - n_inds) / 2
          @test length(state.matches) == n_unique_pairs * n_inds^2
          # single pop
          state = State("", rng, [single_pop_creator, env_creator], [pop_initializer, AllVsAllMatchMaker()], counters=default_counters())
          run!(state, 1)
          @test length(state.matches) == n_inds^2
      end

      @testset "BestVsBest" begin
          state = State("", rng, [comp_pop_creator, env_creator], [pop_initializer,RandomEvaluator(), BestVsBestMatchMaker(env_creator=env_creator)], counters=default_counters())
          run!(state, 1)
          # get best individual for p1 and p2
          inds1 = Jevo.find_population("p1", state).individuals
          inds2 = Jevo.find_population("p2", state).individuals
          best_ind_p1 = inds1[argmax([ind.records[1].fitness for ind in inds1])]
          best_ind_p2 = inds2[argmax([ind.records[1].fitness for ind in inds2])]
          @test length(state.matches) == 1
          @test state.matches[1].individuals == [best_ind_p1, best_ind_p2]

          state = State("", rng, [single_pop_creator, env_creator], [pop_initializer, RandomEvaluator(), BestVsBestMatchMaker(env_creator=env_creator)], counters=default_counters())
          run!(state, 1)
          @test length(state.matches) == 1
          @test 1 == state.matches[1].individuals |> unique |> length
      end
      @testset "Solo" begin
            state = State("", rng,[comp_comp_pop_creator, env_creator], [pop_initializer, solo], counters=default_counters())
          run!(state, 1)
          @test length(state.matches) == n_inds * n_species * n_pops
      end
  end


  @testset "Performer" begin
      state = State("", rng,[comp_comp_pop_creator, env_creator],
                    [pop_initializer, ava, performer], counters=default_counters())
      # Make sure no inds have interactions
      @test all(ind -> isempty(ind.interactions),
              Jevo.get_individuals(state.populations))
      run!(state, 1)
      expected_n_interactions = n_species * n_inds
      @test all(length(ind.interactions) == expected_n_interactions 
                for ind in Jevo.get_individuals(state.populations))
  end
  @testset "ScalarFitnessEvaluator" begin
      state = State("", rng,[comp_comp_pop_creator, env_creator],
                    [pop_initializer, ava, performer, evaluator], counters=default_counters())
      run!(state, 1)
      @test all(ind -> length(ind.records) == 1,
              Jevo.get_individuals(state.populations))
      @test generation(state) == 2
  end
  @testset "Selector" begin
      state = State("", rng,[comp_comp_pop_creator, env_creator],
                    [pop_initializer, ava, performer, evaluator, selector_p1, selector_p2_p3], counters=default_counters())
      run!(state, 1)
      @test length(Jevo.find_population("p1", state).individuals) == k
      @test length(Jevo.find_population("p2", state).individuals) == k
      @test length(Jevo.find_population("p3", state).individuals) == k
  end
  @testset "Reproducer" begin
      state = State("", rng,[comp_comp_pop_creator, env_creator],
                     [pop_initializer, ava, performer, evaluator, 
                    selector_p1, selector_p2_p3, reproducer_p2_p3],counters=default_counters())
      run!(state, 1)
      @test length(Jevo.find_population("p1", state).individuals) == k
      @test length(Jevo.find_population("p2", state).individuals) == n_inds
      @test length(Jevo.find_population("p3", state).individuals) == n_inds
      @test Jevo.find_population("p3", state).individuals[1].generation == 0
      @test Jevo.find_population("p3", state).individuals[2].generation == 1
  end
  @testset "Mutator" begin
      # comp_pop_creator has n_species pops with n_inds inds
      state = State("", rng,[comp_comp_pop_creator, env_creator],
                    [pop_initializer, ava, performer, evaluator, selector_p1, selector_p2_p3, reproducer_p2_p3, mutator], counters=default_counters())
      run!(state, 1)
      pops = PopulationRetriever(["p2", "p3"])(state)
      @test [length(p[1].individuals) for p in pops] == [n_inds, n_inds]
  end
  @testset "Assertor" begin
      state = State("", rng,[comp_comp_pop_creator, env_creator],
                    [pop_initializer, ava, performer, evaluator, assertor, selector, reproducer, assertor, mutator, assertor], counters=default_counters())
      run!(state, 1)
      @test true
      state = State("", rng,[comp_comp_pop_creator, env_creator],
                    [pop_initializer, ava, performer, evaluator, assertor, selector, assertor], counters=default_counters())
      try
          run!(state, 1)
          @test false
      catch
          @test true
      end
  end
  @testset "Reporter" begin
      rm("statistics.h5", force=true)
      rm("run.log", force=true)
      with_logger(JevoLogger()) do
          state = State("", rng,[comp_comp_pop_creator, env_creator],
                        [pop_initializer, reporter], counters=default_counters())
          run!(state, 1)
      end
      rm("statistics.h5", force=true)
      rm("run.log", force=true)
  end
  @testset "run multigen" begin
      println("running multigen")
      min_sums = []
      min_sum_computer = create_op("Reporter",
                            operator=(s,_) -> measure(GenotypeSum, s, false,false,false).min)
      with_logger(JevoLogger()) do
          state = State("", rng,[comp_comp_pop_creator, env_creator],
                        [pop_initializer, ava, performer, evaluator, assertor, selector, min_sum_computer, reproducer, assertor, mutator, reporter, ind_resetter], counters=default_counters())
          run!(state, 10)
      end
      @test all(diff(min_sums) .>= 0)  # confirm elite minimum sum monotonically increases
  end
end


