export Creator
struct Creator <: AbstractCreator
    type::Type
    kwargs::Union{NamedTuple,Tuple}
end

Creator(type::Type) = Creator(type, NamedTuple())

# Interface for sampling random individuals
(creator::Creator)() = creator.type(creator.kwargs...)
