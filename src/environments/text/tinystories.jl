using Transformers.Datasets: batched
using Flux.Losses
export TinyStoriesDataSet

Base.@kwdef struct TinyStoriesDataSet <: AbstractEnvironment
    n_tokens::Int
    n_sequences::Int
    max_seq_len::Int
    batch_size::Int
end

# ==== PERFORMANCE CRITICAL END
function sample_batch(env::TinyStoriesDataSet)
    # Each string is enclosed in a tuple for the batch
    # If we were using encoder-decoder, we would have a tuple of two strings
    # read in ./datasets/tinystories/first10kstories.txt
    stories = readlines(joinpath(@__DIR__, "datasets/tinystories/first10kstories.txt"))
    seqs = [(stories[i],) for i in 1:env.n_sequences]
    batched(seqs)[1]
end


function shift_decode_loss(logits, trg, trg_mask::M) where M <: Transformers.NeuralAttentionlib.LengthMask
    # ignore start
    label = trg[:, 2:end, :]
    # ignore end sequence, idx doesn't correspond to <s>
    -logitcrossentropy(@view(logits[:, 1:end-1, :]), label, trg_mask-1)
end

function loss(input, textmodel)
    logits = textmodel(input)
    ce_loss = shift_decode_loss(logits, input.token, input.attention_mask)
    ce_loss
end

function compute_loss_over_batches(env, input, textmodel)
    ce_loss = 0f0
    n_batches = 0
    for start_idx in 1:env.batch_size:env.n_sequences
        n_batches += 1
        end_idx = min(start_idx+env.batch_size-1, env.n_sequences)
        split_input = (token = input.token[:,:,start_idx:end_idx],
                    attention_mask = Transformers.NeuralAttentionlib.LengthMask(input.attention_mask.len[start_idx:end_idx]))
        ce_loss += loss(split_input, textmodel)
    end
    ce_loss / n_batches
end

function step!(env::TinyStoriesDataSet, ids::Vector{Int}, models::Vector{TextModel{TE,M}}) where {TE, M}
    @assert length(models) == 1 "Only one model is supported for now"
    model = models[1]
    input_batch = get_preprocessed_batch(env, model)
    ce_loss = compute_loss_over_batches(env, input_batch, model)
    [Interaction(ids[1], [], ce_loss)]
end

function evaluate(env_creator::Creator{TinyStoriesDataSet}, individual::Individual, generation::Int)
  wid = workers()[1]
  interactions = remotecall_fetch(Jevo.play, wid, env_creator, [individual])
  @assert length(interactions) == 1
  negativeloss = interactions[1].score
  negativeloss_measurement = Measurement(NegativeLoss, negativeloss, generation)
  @info negativeloss_measurement
  @h5 negativeloss_measurement
  negativeloss
end
