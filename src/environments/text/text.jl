include("regular-language.jl")
include("repeatsequence.jl")
export NegativeLoss, PercentCorrect
struct NegativeLoss <: AbstractMetric end
struct PercentCorrect <: AbstractMetric end
