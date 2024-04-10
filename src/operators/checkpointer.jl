export Checkpointer, restore_from_checkpoint
function restore_from_checkpoint(checkpointname)
    !isfile(checkpointname) && return
    state = Serialization.deserialize(checkpointname)
    @info "Restored state from $checkpointname"
    return state
end
function checkpoint(state, checkpointname)
    tmp_name = "$checkpointname.tmp"
    Serialization.serialize(tmp_name, state)
    mv(tmp_name, checkpointname, force=true)
    @info "Serialized state to $checkpointname"
end

@define_op "Checkpointer"
Checkpointer(checkpointname::String="./state.jls"; interval::Int=100,kwargs...) = create_op("Checkpointer",
    condition=(state)->generation(state) % interval == 1,
    operator=(state,_)->checkpoint(state, checkpointname))
