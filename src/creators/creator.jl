export Creator
struct Creator <: AbstractCreator
    type::Type
    kwargs::Union{NamedTuple,Tuple}
end

Creator(type::Type) = Creator(type, NamedTuple())

# Interface for sampling random individuals
(creator::Creator)() = creator.type(creator.kwargs...)

create(c::Creator) = c()
function create(cs::Vector{<:AbstractCreator})
    @assert length(cs) > 0 "No creators to create"
    T = typeof(cs[1].type)
    T[create(c) for c in cs]
end
