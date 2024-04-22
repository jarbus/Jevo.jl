export Creator, PassThrough
"""
A creator is a struct that contains a type to create and the arguments to
pass to the constructor. If arguments contain other creators, those are
recursively created first. We use a creator to allow for creating multiple
objects which may include random initialization with a single creator.
"""
struct Creator{T} <: AbstractCreator where T
    type::Type{T}
    kwargs::Union{NamedTuple,Tuple}
end

"""
A pass-through creator is a struct that contains a creator to create. When the
first create call is made, say on a population, the passthrough creator will return
the creator, instead of actually performing the creation. This allows us to pass
through genotype creators and developers to instantiated populations without creating
them.
"""
struct PassThrough <: AbstractCreator
    creator::AbstractCreator
end
(passthrough::PassThrough)() = passthrough.creator
(creator::Creator{T})() where T = T(create(creator.kwargs)...)

# Empty creator
Creator(type::Type) = Creator(type, NamedTuple())

create(::AbstractState, x) = create(x)
create(x) = x
create(c::AbstractCreator) = c()
create(args::Tuple) = Tuple(create(arg) for arg in args)
create(kwargs::NamedTuple) = NamedTuple(k => create(v) for (k,v) in pairs(kwargs))
function create(cs::Vector{<:AbstractCreator})
    @assert length(cs) > 0 "No creators to create"
    create.(cs)
end

Base.show(io::IO, c::Creator) = print(io, "Creator{$(c.type)}($(c.kwargs[1]))")
