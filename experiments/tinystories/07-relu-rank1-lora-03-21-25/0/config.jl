using Distributed, CUDA, Transformers
enable_gpu(CUDA.functional()) 
using Jevo, StableRNGs, Flux, Transformers.TextEncoders, Logging

report_interval = 50
initial_pop_size = 512
ongoing_pop_size = 512
nback = 10000
n_species = 1
k = 16
n_generations = 300
n_tokens = 2048
new_head_prob = 0.001
new_block_prob = 0.001
mrs = (0.01f0,)

startsym, endsym, unksym, labels = "<s>", "</s>", "<unk>", string.(0:n_tokens-1)
vocab = [unksym, startsym, endsym, labels...]
vocab_size = length(vocab)

trf_args = (
  n_blocks = 3,
  n_heads = 4,
  head_dim = 4,
  hidden_dim = 32,
  ff_dim = 128,
  qkv_rank=1,
  embed_rank=4,
  ff_rank=1,
  ff_σ=relu,
  o_rank=1,
  vocab_size = vocab_size,
 )

env_args = (
  n_tokens = length(labels), # ignore {unk,start,end}sym
  n_sequences = 1024,
  max_seq_len = -1,
  batch_size=128,
)

rng = StableRNG(parse(Int, basename(@__DIR__)))
n_workers = 8
n_gpus = parse(Int, readchomp(`grep -oP '(?<=--gres=gpu:)\d+' batch.sh`))
checkpointname = "check.jls"

counters = default_counters()
textenc = TransformerTextEncoder(split, vocab; startsym, endsym, unksym, padsym=unksym)
phen_creator = Creator(TextModel, (;textenc=textenc))
gene_counter = find(:type, AbstractGene, counters)
geno_creator = Creator(Delta, Creator(TextTransformer, (rng, gene_counter, trf_args)))
pop_creator = Creator(Population, ("p", initial_pop_size, PassThrough(geno_creator), PassThrough(phen_creator), counters))
env_creator = Creator(TinyStoriesDataSet, env_args)

struct InteractionDist <: AbstractMetric end
Visualizer(;kwargs...) = create_op("Reporter",
    retriever=Jevo.PopulationRetriever(),
    operator=(s,ps)-> ( ind = ps[1][1].individuals[1]; @info(string(ind.id)* " "*visualize(ind, ps[1][1]))); kwargs...)
GC(;kwargs...) = create_op("Reporter", operator=(args...)->Base.GC.gc(); kwargs...)
PrintInteractions(;kwargs...) = create_op("Reporter",
    retriever=get_individuals,
operator=(s,is)-> (@info StatisticalMeasurement(InteractionDist, [round(i.interactions[1].score, digits=3) for i in is], generation(s)); ))

state = if isfile(checkpointname)
          restore_from_checkpoint(checkpointname)
    else
        rm("statistics.h5", force=true)
        State("", rng, [pop_creator, env_creator], 
            [ InitializeAllPopulations(),
                    CreateMissingWorkers(n_workers, slurm=true, c=4, n_gpus=n_gpus),
                    InitializePhylogeny(),
                    InitializeDeltaCache(),
                    SoloMatchMaker(["p"]), 
                    Performer(time=true),
                    GC(),
                    ScalarFitnessEvaluator(),
                    PrintInteractions(),
                    TruncationSelector(k),
                    CloneUniformReproducer(ongoing_pop_size),
                    UpdatePhylogeny(),
                    UpdateParentsAcrossAllWorkers(),
                    Visualizer(condition=s->generation(s) % report_interval == 0),
                    RecordPerformance(env_creator, condition=s->generation(s) % report_interval ==0),
                    ClearCurrentGenWeights(),
                    NBackMutator(n_back=nback, mrs=mrs, no_layer_norm=true),
                    UpdateDeltaCache(),
                    ClearInteractionsAndRecords(),
            ], counters=counters)
      end

global_logger(JevoLogger())
run!(state, n_generations)
