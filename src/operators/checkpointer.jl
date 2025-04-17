using Printf
export Checkpoint, LoadCheckpoint

function restore_from_checkpoint!(state::State, checkpointname::String)
    !isfile(checkpointname) && return state
    loaded_data = Serialization.deserialize(checkpointname)
    loaded_state = loaded_data[:state]
    for field in fieldnames(State)
        setfield!(state, field, getfield(loaded_state, field))
    end
    if haskey(loaded_data, :weight_cache)
        global weight_cache = loaded_data[:weight_cache]
    end
    if haskey(loaded_data, :genotype_cache)
        global genotype_cache = loaded_data[:genotype_cache]
    end
    @info "Restored state from $checkpointname at generation $(generation(state))"
end

function checkpoint(state::State, checkpointname::String)
    checkroot, ext = splitext(checkpointname)
    dash_gen = @sprintf "%05d" (generation(state)-1)
    checkname_withgen = checkroot * "-" * dash_gen * ext

    data_to_save = Dict(
        :state => state,
        :weight_cache => weight_cache,
        :genotype_cache => genotype_cache
    )
    Serialization.serialize(checkname_withgen, data_to_save)
    islink(checkpointname) && rm(checkpointname)
    symlink(checkname_withgen, checkpointname)
    @info "Serialized state to $checkpointname at generation $(generation(state))"
end

@define_op "Checkpoint"
Checkpoint(checkpointname::String="./check.jls"; interval::Int,kwargs...) = create_op("Checkpoint",
    condition=(state)-> (generation(state)-1) % interval == 0,
    operator=(state,_)->checkpoint(state, checkpointname))


@define_op "LoadCheckpoint" "AbstractOperator" "checkpointname:String"
LoadCheckpoint(checkpointname::String="./check.jls"; kwargs...) = 
    create_op("LoadCheckpoint",checkpointname,
        condition=first_gen,
        updater =(state, _)->restore_from_checkpoint!(state, checkpointname),)
