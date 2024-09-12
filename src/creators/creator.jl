export Creator, PassThrough

"""
    struct Creator{T} <: AbstractCreator where T
        type::Type{T}
        kwargs::Union{NamedTuple,Tuple}
    end

A creator is a struct that contains a type to create and the arguments to
pass to the constructor. If arguments contain other creators, those are
recursively created first, unless enclosed in a [PassThrough](@ref). We use a creator to allow for creating multiple objects which may include random initialization with a single creator.

See also: [PassThrough](@ref), [create](@ref)

"""
struct Creator{T} <: AbstractCreator where T
    type::Type{T}
    kwargs::Union{NamedTuple,Tuple}
end

"""
    struct PassThrough <: AbstractCreator
        creator::AbstractCreator
    end

A PassThrough is a struct that contains a creator to create. When the
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
# So we don't need to pass a tuple for a single sub-creator
Creator(type::Type, c::AbstractCreator) = Creator(type, (Creator(c.type, c.kwargs),))

"""
    create(x)

Calls `create` on all elements of `x` that are creators, and returns `x` otherwise. This recursively instantiates all creators. If a creator is enclosed in a [PassThrough](@ref), the PassThrough wrapper is removed, and the creator is kept as is. 

  `create(::AbstractState, x)`: Calls `create(x)`, can be used in an [Operator](@ref).
  `create(c::AbstractCreator)`: Calls the creator function `c`.
  `create(args::Tuple)`: Recursively creates elements of the tuple.
  `create(kwargs::NamedTuple)`: Recursively creates values of the named tuple.
  `create(cs::Vector{<:AbstractCreator})`: Calls `create` for each element in the vector of creators.
  `create(x)`: Returns `x` if x is not a creator and does not contain a creator

All calls to `create` eventually lead, someway or another, to a call to the following constructor

    (creator::Creator{T})() where T = T(create(creator.kwargs)...)

which creates the object of type `T` with the arguments in `creator.kwargs`.

See also: [Creator](@ref), [PassThrough](@ref)
"""
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
