include("regular-language.jl")
include("repeatsequence.jl")
include("tinystories.jl")
export NegativeLoss, PercentCorrect, RecordPerformance
struct NegativeLoss <: AbstractMetric end
struct PercentCorrect <: AbstractMetric end

#global preprocessed_batch = nothing
function get_preprocessed_batch(env::Union{RepeatSequence, RegularLanguage, AcceptRejectStrings, TinyStoriesDataSet}, tm::TextModel)
    # There appears to be some memory management issue, where GPU OOMs.
    # Allocating a large amount of memory on the CPU appears to alleviate this 
    # issue. Garbage collection does not help. Unable to justify spending
    # more time on this, if it's resolved. On my laptop, this takes ~179μs per call
    encode(tm.textenc, sample_batch(env)) |> gpu
end

RecordPerformance(env_creator;kwargs...) = create_op("Reporter",
        retriever=Jevo.get_individuals,
        operator=(s,is)-> evaluate(env_creator, is[1], generation(s)); kwargs...)
