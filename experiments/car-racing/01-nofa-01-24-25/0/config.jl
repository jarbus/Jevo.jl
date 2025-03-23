using Distributed, CUDA, Transformers
enable_gpu(CUDA.functional()) 
using Jevo, StableRNGs, Flux, Transformers.TextEncoders, Logging, StatsBase
Jevo.set_device()

record_interval = 10
initial_pop_size = 256
ongoing_pop_size = 256
nback = 1000
n_species = 1
k = 1
n_generations = 40
mrs = Float32.((1e-2,))

env_args = (
    n_envs=1,
    n_steps=10000,
    n_stack=4,
    record_prefix="",
)

if isdir("video")
    rm("video"; force=true, recursive=true)
end

rng = StableRNG(parse(Int, basename(@__DIR__)))

counters = default_counters()
gene_counter = find(:type, AbstractGene, counters)
geno_creator = Creator(Delta, Creator(JevoChain, (rng, gene_counter, [
    (Jevo.Conv, (kernel=(4,4), channels=3*env_args.n_stack=>32, stride=(2,2), σ=relu)),
    (Jevo.Conv, (kernel=(4,4), channels=32=>64, stride=(2,2), σ=relu, )),
    (Jevo.Conv, (kernel=(4,4), channels=64=>128, stride=(2,2), σ=relu, )),
    (Jevo.Conv, (kernel=(4,4), channels=128=>256, stride=(2,2), σ=relu, )),
    Flux.flatten,
    (Jevo.Dense, (dims=(1024,256),σ=relu, )),
    (Jevo.Dense, (dims=(256,3),σ=identity))
])))

record_condition = (s)->(generation(s) % record_interval == 0)
phen_creator = Creator(Model)
pop_creator = Creator(Population, ("p", initial_pop_size, PassThrough(geno_creator), PassThrough(phen_creator), counters))
n_workers = 32

env_creator = Creator(CarRacingV3, env_args)
checkpointname = "check.jls"

struct FirstWave <: AbstractMetric end
struct SecondWave <: AbstractMetric end
Visualizer(;kwargs...) = create_op("Reporter",
    retriever=Jevo.PopulationRetriever(),
    operator=(s,ps)-> ( ind = ps[1][1].individuals[1]; @info(string(ind.id)* " "*visualize(ind, ps[1][1]))); kwargs...)
PrintInteractions(metric;kwargs...) = create_op("Reporter";
    retriever=get_individuals,
    operator=(s,is)-> (m=StatisticalMeasurement(metric, [sum(int.score for int in ind.interactions)/6 for ind in is], generation(s)); @info(m); @h5(m);), kwargs...)

ClearRecords(;kwargs...) = create_op("Operator",
          retriever=get_individuals,
          updater=map((_,ind)->empty!(ind.records)); kwargs...)

state = if isfile(checkpointname)
          restore_from_checkpoint(checkpointname)
    else
        rm("statistics.h5", force=true)
        State("", rng, [pop_creator, env_creator], 
            [ InitializeAllPopulations(),
                    CreateMissingWorkers(n_workers, slurm=false, c=4, n_gpus=1),
                    InitializePhylogeny(),
                    InitializeDeltaCache(),

                    [SoloMatchMaker(["p"]) for i in 1:2]..., 
                    Performer(time=true),
                    ScalarFitnessEvaluator(),
                    PrintInteractions(FirstWave),
                    TruncationSelector(Int(32),), # 32
                    ClearRecords(),

                    [SoloMatchMaker(["p"]) for i in 1:4]..., 
                    Performer(time=true),
                    ScalarFitnessEvaluator(),
                    TruncationSelector(8,),
                    PrintInteractions(SecondWave),
                    Visualizer(condition=record_condition),

                    SoloMatchMaker(["p"], env_creator=Creator(CarRacingV3, (env_args..., record_prefix="1")), condition=record_condition),
                    Performer(time=true, condition=record_condition),

                    CloneUniformReproducer(ongoing_pop_size),
                    UpdatePhylogeny(),
                    UpdateParentsAcrossAllWorkers(),
                    ClearCurrentGenWeights(),
                    NBackMutator(n_back=nback, mrs=mrs, time=true),
                    UpdateDeltaCache(),
                    ClearInteractionsAndRecords(),
            ], counters=counters)
      end

global_logger(JevoLogger())
run!(state, n_generations)
