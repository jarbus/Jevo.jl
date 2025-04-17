function check_if_outcome_in_cache(cache, id1::Int, id2::Int)
    isnothing(cache) && return false
    if id1 in keys(cache) && id2 in keys(cache[id1]) && id2 in keys(cache) && id1 in keys(cache[id2])
        return true
    end
    false
end
include("./all_vs_all.jl")
include("./solo.jl")
include("./best_vs_best.jl")
include("./best_vs_all.jl")
include("./self_vs_self.jl")
include("./old_vs_new.jl")
include("./random_cohort.jl")
