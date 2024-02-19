# A creator is a struct that contains a type to create and the arguments to
# pass to the constructor. If arguments contain other creators, those are
# recursively created first. We use a creator to allow for creating multiple
# objects which may include random initialization with a single creator.
export Creator
struct Creator <: AbstractCreator
    type::Type
    kwargs::Union{NamedTuple,Tuple}
end

# Empty creator
Creator(type::Type) = Creator(type, NamedTuple())

(creator::Creator)() = creator.type(create(creator.kwargs)...)

create(x) = x
create(c::Creator) = c()
create(args::Tuple) = Tuple(create(arg) for arg in args)
create(kwargs::NamedTuple) = NamedTuple(k => create(v) for (k,v) in pairs(kwargs))
function create(cs::Vector{<:AbstractCreator})
    @assert length(cs) > 0 "No creators to create"
    [create(c) for c in cs]
end

Base.show(io::IO, c::Creator) = print(io, "Creator($(c.type), $(c.kwargs[1]))")
