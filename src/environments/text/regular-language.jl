export RegularLanguage, evaluate, AcceptRejectStrings
struct RegularLanguage <: AbstractEnvironment
    regex::Regex
    seq_len::Int
    n_strings::Int
end

struct AcceptRejectStrings <: AbstractEnvironment
    accept::Vector{String}
    reject::Vector{String}
end

function get_strings(env::RegularLanguage)
    @assert env.n_strings % 2 == 0 "Need an even number of strings"
    @assert 2^env.seq_len >= env.n_strings "Not enough strings possible, got $(2^env.seq_len) need $(env.n_strings)"
    # Generate strings
    accept, reject = String[], String[]
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


function get_strings(env::AcceptRejectStrings)
    strings = vcat([s*":a" for s in env.accept], [s*":r" for s in env.reject])
    shuffle!(StableRNG(1), strings)
    strings
end

function get_preprocessed_batch(env::Union{RegularLanguage, AcceptRejectStrings}, m)
    # There appears to be some memory management issue, where GPU OOMs.
    # Allocating a large amount of memory on the CPU appears to alleviate this 
    # issue. Garbage collection does not help. Unable to justify spending
    # more time on this, if it's resolved. On my laptop, this takes ~179μs per call
    size(zeros(1_000_000))
    if !isdefined(Main, :preprocessed_batch)
        @warn "Creating variable Main.preprocessed_batch"
        strings = [(s,) for s in get_strings(env)]
        batch = batched(strings)[1]
        Main.preprocessed_batch = encode(get_text_encoder(m), batch)
    end
    Main.preprocessed_batch |> deepcopy |> gpu
end

# Use a custom kernel to find the index on each row
# dims are (vocab_size, seq_len, batch_size)
function find_end_indices_on_gpu(matrix)
    indices = CUDA.fill(Int32(0), size(matrix, 3)) # batch_size 
    @cuda threads=size(matrix, 3) find_end_indices_kernel!(indices, matrix)
    @assert all(indices .>= 0) "End token not found"
    return indices
end
function find_end_indices_kernel!(indices, matrix)
    test_case = threadIdx().x
    for col in 2:size(matrix, 2)  # seq_len
        if matrix[3, col, test_case] == 1 # hardcoded one-hot end token
            indices[test_case] = col-1  # we want the index before the end token
            return
        end
    end
    indices[test_case] = -1
    nothing
end

# if we do [:, idxs, :] it returns a 3d array, which is not what we want
# we want a 2d array of (vocab_size, batch_size), where rows are the logits
# at each end idx, which can be different.
function get_final_logits_on_gpu(indices, matrix)
    logits = CUDA.fill(0f0, size(matrix, 1), size(matrix, 3)) # vocab_size, batch_size
    @cuda threads=size(matrix, 3) get_final_logits_kernel!(logits, indices, matrix)
    return logits
end

function get_final_logits_kernel!(logits, indices, matrix)
    test_case = threadIdx().x
    idx = indices[test_case]
    for row in 1:size(matrix, 1)  # vocab_size
        logits[row, test_case] = matrix[row, idx, test_case]
    end
    nothing
end

function infer(env::Union{RegularLanguage, AcceptRejectStrings}, model::TextModel)
    batch = get_preprocessed_batch(env, model)
    logits = model(batch)
    # Compute end index per sequence on gpu using custom kernel
    # We assume that the end token is token 3. This uses a one-hot encoding.
    # dims are (vocab_size, seq_len, batch_size)
    end_idxs = find_end_indices_on_gpu(batch.token)
    # We only optimize for the end-token and the accept/reject before it
    # batch.token includes a start token
    accept_or_reject_final = get_final_logits_on_gpu(end_idxs, batch.token)
    logits_final = get_final_logits_on_gpu(end_idxs .- 1, logits)
    loss = -logitcrossentropy(logits_final, accept_or_reject_final)
    loss, logits_final, accept_or_reject_final
end
function step!(env::Union{RegularLanguage, AcceptRejectStrings}, ids::Vector{Int}, models::Vector{TextModel})
    # One shot classification of accept / reject
    @assert length(models) == length(ids) == 1
    loss, _, _ = infer(env, models[1])
    [Interaction(ids[1], Int[], loss)]
end
function percent_correct(logits, accept_or_reject)
    # only the first column is the accept/reject, the second is the end token
    # (vocab_size, seq_len , batch_size)
    pred_logits = logits[:,:] |> Transformers.tocpudevice
    target_logits = accept_or_reject[:,:] |> Transformers.tocpudevice
    preds = argmax(pred_logits, dims=1) 
    targets = argmax(target_logits, dims=1)
    for i in 1:size(preds,2)
        @info "Pred: $(round.(softmax(pred_logits[:,i]), digits=2)) Target: $(targets[i].I[1])"
    end
    same = preds .== targets
    sum(same) / length(same)
end
function evaluate(env_creator::Union{Creator{AcceptRejectStrings}, Creator{RegularLanguage}}, individual::Individual)
    model = develop(individual.developer, individual)
    loss, logits, accept_or_reject = infer(env_creator(), model)
    p_correct = percent_correct(logits, accept_or_reject)
    @info "Percent Correct: $p_correct"
    loss
end
