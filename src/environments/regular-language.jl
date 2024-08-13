export RegularLanguage, evaluate
# TODO figure out what is the max string length in dynarec
#   i think he trains on smaller sequences and then observes behavior in the limit
#   i think he also trains on the same set of sequences
#   he doesn't converge to passing on every single sequence
#   he said to make infinite strings
#   for this, maybe i need to have an output that is not a softmax?
#   i need to do something to separate classification from generation
#   maybe something like 0101:C where C is the classification
#   doesn't backprop over the other logits, so it's not a problem
struct RegularLanguage <: AbstractEnvironment
    regex::Regex
    seq_len::Int
    n_strings::Int
end


function get_strings(env::RegularLanguage)
    @assert env.n_strings % 2 == 0 "Need an even number of strings"
    @assert 2^env.seq_len >= env.n_strings "Not enough strings possible, got $(2^env.seq_len) need $(env.n_strings)"
    # Generate strings
    accept = String[]
    reject = String[]
    # Generate all binary strings up to seq_len
    for i = 0:(2^env.seq_len)-1
        s = string(i, base=2)
        if !isnothing(match(env.regex, s))
            push!(accept, s*":a")
        else
            push!(reject, s*":r")
        end
    end
    @assert length(accept) >= env.n_strings / 2 "Not enough accept strings, got $(length(accept)) need $(env.n_strings / 2)"
    @assert length(reject) >= env.n_strings / 2 "Not enough reject strings, got $(length(reject)) need $(env.n_strings / 2)"
    rng = StableRNG(1)
    accept = sample(rng, accept, Int(env.n_strings / 2), replace=false)
    reject = sample(rng, reject, Int(env.n_strings / 2), replace=false)
    strings = shuffle!(rng, vcat(accept, reject))
    strings
end

preprocess(trf::TransformerPhenotype, batch) = encode(trf.textenc, batch)
function get_preprocessed_batch(env::RegularLanguage, tfr)
    # There appears to be some memory management issue, where GPU OOMs.
    # Allocating a large amount of memory on the CPU appears to alleviate this 
    # issue. Garbage collection does not help. Unable to justify spending
    # more time on this, if it's resolved. On my laptop, this takes ~179μs per call
    size(zeros(1_000_000)) # TODO see if we can change this to fill(undef, 1_000_000)
    if !isdefined(Main, :preprocessed_batch)
        @warn "Creating variable Main.preprocessed_batch"
        strings = [(s,) for s in get_strings(env)]
        batch = batched(strings)[1]
        Main.preprocessed_batch = preprocess(tfr, batch)
    end
    Main.preprocessed_batch |> deepcopy |> gpu
end


function infer(env::RegularLanguage, model::TransformerPhenotype)
    batch = get_preprocessed_batch(env, model)
    logits = model(batch)
    end_idx = size(batch.token, 2)
    # We only optimize for the end-token and the accept/reject before it
    # batch.token includes a start token
    accept_or_reject = @view(batch.token[:,end_idx-1:end_idx,:])
    logits_view = @view(logits[:, end_idx-2:end_idx-1, :])
    loss = -logitcrossentropy(logits_view, accept_or_reject)
    loss, logits, accept_or_reject
end
function step!(env::RegularLanguage, ids::Vector{Int}, models::Vector{TransformerPhenotype})
    # One shot classification of accept / reject
    @assert length(models) == length(ids) == 1
    loss, _, _ = infer(env, models[1])
    [Interaction(ids[1], Int[], loss)]
end
function percent_correct(logits, accept_or_reject)
    # only the first column is the accept/reject, the second is the end token
    # (vocab_size, seq_len , batch_size)
    preds = argmax(logits[:,1,:], dims=1) |> Transformers.tocpudevice
    targets = argmax(accept_or_reject[:,1,:], dims=1) |> Transformers.tocpudevice
    same = preds .== targets
    sum(same) / length(same)
end
function evaluate(env_creator::Creator, individual::Individual)
    model = develop(individual.developer, individual)
    loss, logits, accept_or_reject = infer(env_creator(), model)
    p_correct = percent_correct(logits, accept_or_reject)
    @info "Percent Correct: $p_correct"
    loss
end
