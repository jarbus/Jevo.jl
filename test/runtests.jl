println("==================================")
ENV["JULIA_STACKTRACE_ABBREVIATED"] = true
ENV["JULIA_STACKTRACE_MINIMAL"] = true
ENV["JULIA_TEST_MODE"] = "true"
using AbbreviatedStackTraces
using Jevo
using Test
using StableRNGs

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

@testset "general tests using numbers game" begin
    # creators
    n_dims = 2
    n_inds = 2
    n_species = 2
    n_pops = 2
    counters = default_counters()
    ng_gc = ng_genotype_creator = Creator(VectorGenotype, (n=n_dims,rng=rng))
    ng_developer = Creator(VectorPhenotype)
    @testset "creation" begin
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
        env_creator = Creator(CompareOnOne)
        @testset "env" begin
            env = env_creator()
        end

        @testset "MatchMaker and Performer" begin
            # Matchmaker only
            ava = AllVsAllMatchMaker(["composite1", "composite2"])
            mm_state = State([comp_comp_pop_creator, env_creator], [pop_initializer, ava])
            @test Jevo.get_creators(AbstractEnvironment, mm_state) |> length == 1
            run!(mm_state, 1)
            n_pairs = n_pops * n_species
            n_unique_pairs = n_pairs * (n_inds^2 - n_inds) / 2
            @test length(mm_state.matches) == n_unique_pairs * n_inds^2
            # Performer
            # Runs matches and updates individuals in each pop
            performer = Performer()
            mm_p_state = State([comp_comp_pop_creator, env_creator],
                               [pop_initializer, ava, performer])
            # Make sure no inds have interactions
            @test all(ind -> isempty(ind.interactions),
                    Jevo.get_individuals(mm_p_state.populations))
            run!(mm_p_state, 1)
            # make sure each ind has some interactions
            @test all(ind -> !isempty(ind.interactions),
                    Jevo.get_individuals(mm_p_state.populations))
            # Test that each individual has the expected number of interactions for AVA
            # for composite1 vs composite2
            expected_n_interactions = n_species * n_inds
            @test all(length(ind.interactions) == expected_n_interactions 
                      for ind in Jevo.get_individuals(mm_p_state.populations))
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
        @testset "Mutator" begin
           # comp_pop_creator has n_species pops with n_inds inds
           state = State([comp_pop_creator], [InitializeAllPopulations(), Mutator()])
           run!(state, 1)
           pops = PopulationRetriever(["p1", "p2"])(state)
           @test [length(p[1].individuals) for p in pops] == [4,4]
        end
    end

    @testset "operators" begin
        @testset "matchmaker" begin
        end

        @testset "evaluators" begin
        end

        @testset "scorers" begin
        end

        @testset "replacer" begin
        end

        @testset "selectors" begin
        end

        @testset "mutators" begin
        end

        @testset "reporter" begin
        end

        @testset "assertor" begin
        end
    end
end

@testset "neural-net" begin
end
end
