println("==================================")
ENV["JULIA_STACKTRACE_ABBREVIATED"] = true
ENV["JULIA_STACKTRACE_MINIMAL"] = true
ENV["JULIA_TEST_MODE"] = "true"
using AbbreviatedStackTraces
using Jevo
using HDF5
using Test
using StableRNGs
using Logging

@testset "Jevo.jl" begin

rng = StableRNG(1)

@testset "counter" begin
    """Creates a counter, counts 10 times, and checks that the final value is correct."""
    c = Counter(AbstractGene)
    for i in 1:10
        inc!(c)
    end
    @test value(c) == 11
    @test find(:type, AbstractGene, [c]) == c
    try
        find(:type, AbstractIndividual, [c])
    catch
        @test true
    end

end

@testset "state" begin
    state = State()
    Jevo.operate!(state)
    @test Jevo.get_counter(AbstractGeneration, state) |> value == 2
end

@testset "HDF5Logger" begin
    rm("statistics.h5", force=true)
    with_logger(Jevo.HDF5Logger("statistics.h5")) do
        m = Measurement(GenotypeSum, 1, 1)
        sm = StatisticalMeasurement(GenotypeSum, [1,2,3], 1)
        @h5 m
        @h5 sm
    end
    h5open("statistics.h5", "r") do io
        @test haskey(io, "1/GenotypeSum/value")
        @test haskey(io, "1/GenotypeSum/min")
        @test haskey(io, "1/GenotypeSum/max")
        @test haskey(io, "1/GenotypeSum/mean")
        @test haskey(io, "1/GenotypeSum/std")
        @test haskey(io, "1/GenotypeSum/n_samples")
    end
    rm("statistics.h5", force=true)
end
@testset "JevoLogger" begin
    rm("statistics.h5", force=true)
    with_logger(Jevo.JevoLogger()) do
        sm = StatisticalMeasurement(GenotypeSum, [1,2,3], 1)
        # log to hdf5 only
        log(sm, true, false, false)
        # log to text only
        log(sm, false, true, false)
    end
    rm("statistics.h5", force=true)
end

@testset "general tests using numbers game" begin
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
    # Composite population
    comp_pop = CompositePopulation("species", [("p$i", n_inds, ng_gc, ng_developer) for i in 1:n_species], counters)
    # create a composite population creator
    comp_pop_creator = Creator(CompositePopulation, ("species", [("p$i", n_inds, ng_gc, ng_developer)
                                                                 for i in 1:n_species], counters))
    comp_pop = comp_pop_creator()
    @test comp_pop.populations |> length == n_pops
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
    state = State([comp_comp_pop_creator], [pop_initializer])
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
            find_population("composite1", state)
        catch
            @test true
        end
        try
            find_population("nopop", state)
        catch
            @test true
        end
        # PopulationUpdater
        pops = PopulationRetriever(["p1", "p2"])(state)
        @test all([length(p) == 1 for p in pops])
        @test all([length(p[1].individuals) == 2 for p in pops])
        # bad practice, we are making individuals with the same id
        updater = PopulationUpdater(["p1"])(state, [deepcopy(pops[1][1].individuals)])
        pops = PopulationRetriever(["p1", "p2"])(state)
        @test [length(p) for p in pops] == [1,1]
        @test [length(p[1].individuals) for p in pops] == [4,2]
    end

    env_creator = Creator(CompareOnOne)
    @testset "env" begin
        env = env_creator()
    end

    # Matchmaker only
    ava = AllVsAllMatchMaker(["composite1", "composite2"])
    performer = Performer()
    evaluator = ScalarFitnessEvaluator(["composite1","composite2"])
    k = 1
    selector = TruncationSelector(k)
    selector_p1 = TruncationSelector(k, ["p1"])
    selector_p2_p3 = TruncationSelector(k, ["p2", "p3"])
    reproducer = UniformReproducer(n_inds)
    reproducer_p2_p3 = UniformReproducer(n_inds, ["p2", "p3"])
    mutator = Mutator()
    assertor = PopSizeAssertor(n_inds)
    ind_resetter = ClearInteractionsAndRecords()
    reporter = Reporter(GenotypeSum)
    # TODO add tests for no ids

    @testset "MatchMaker" begin
        state = State([comp_comp_pop_creator, env_creator], [pop_initializer, ava])
        @test Jevo.get_creators(AbstractEnvironment, state) |> length == 1
        run!(state, 1)
        n_pairs = n_pops * n_species
        n_unique_pairs = n_pairs * (n_inds^2 - n_inds) / 2
        @test length(state.matches) == n_unique_pairs * n_inds^2
    end
    @testset "Performer" begin
        state = State([comp_comp_pop_creator, env_creator],
                       [pop_initializer, ava, performer])
        # Make sure no inds have interactions
        @test all(ind -> isempty(ind.interactions),
                Jevo.get_individuals(state.populations))
        run!(state, 1)
        expected_n_interactions = n_species * n_inds
        @test all(length(ind.interactions) == expected_n_interactions 
                  for ind in Jevo.get_individuals(state.populations))
    end
    @testset "ScalarFitnessEvaluator" begin
        state = State([comp_comp_pop_creator, env_creator],
                       [pop_initializer, ava, performer, evaluator])
        run!(state, 1)
        @test all(ind -> length(ind.records) == 1,
                Jevo.get_individuals(state.populations))
        @test generation(state) == 2
    end
    @testset "Selector" begin
        state = State([comp_comp_pop_creator, env_creator],
                       [pop_initializer, ava, performer, evaluator, selector_p1, selector_p2_p3])
        run!(state, 1)
        @test length(Jevo.find_population("p1", state).individuals) == k
        @test length(Jevo.find_population("p2", state).individuals) == k
        @test length(Jevo.find_population("p3", state).individuals) == k
    end
    @testset "Reproducer" begin
        state = State([comp_comp_pop_creator, env_creator],
                       [pop_initializer, ava, performer, evaluator, selector_p1, selector_p2_p3, reproducer_p2_p3])
        run!(state, 1)
        @test length(Jevo.find_population("p1", state).individuals) == k
        @test length(Jevo.find_population("p2", state).individuals) == n_inds
        @test length(Jevo.find_population("p3", state).individuals) == n_inds
    end
    @testset "Mutator" begin
        # comp_pop_creator has n_species pops with n_inds inds
        state = State([comp_comp_pop_creator, env_creator],
                      [pop_initializer, ava, performer, evaluator, selector_p1, selector_p2_p3, reproducer_p2_p3, mutator])
        run!(state, 1)
        pops = PopulationRetriever(["p2", "p3"])(state)
        @test [length(p[1].individuals) for p in pops] == [n_inds, n_inds]
    end
    @testset "Assertor" begin
        state = State([comp_comp_pop_creator, env_creator],
                      [pop_initializer, ava, performer, evaluator, assertor, selector, reproducer, assertor, mutator, assertor])
        run!(state, 1)
        @test true
        state = State([comp_comp_pop_creator, env_creator],
                      [pop_initializer, ava, performer, evaluator, assertor, selector, assertor])
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
            state = State([comp_comp_pop_creator, env_creator],
                          [pop_initializer, reporter])
            run!(state, 1)
        end
        rm("statistics.h5", force=true)
        rm("run.log", force=true)
    end
    @testset "run multigen" begin
        println("running multigen")
        with_logger(JevoLogger()) do
            state = State([comp_comp_pop_creator, env_creator],
                          [pop_initializer, ava, performer, evaluator, assertor, selector, reproducer, assertor, mutator, reporter, ind_resetter])
            run!(state, 10)
        end
    end
    @testset "checkpointer" begin
    end
end

@testset "neural-net" begin
end

end
