export FileSystemGenotype, FileSystemPhenotype, develop, mutate, GenotypeSum, measure

struct FileSystemGenotype <: AbstractGenotype
    dirpath::String
    null_action_space::Tuple{Vararg{Int}}

end
struct FileSystemPhenotype <: AbstractPhenotype
    dirpath::String
    null_action_space::Tuple{Vararg{Int}}
    run_until_done::Bool
end

function mutate(::AbstractRNG, ::AbstractState, ::AbstractPopulation, genotype::FileSystemGenotype)
    @error "mutate() not defined for FileSystemGenotype"
end

develop(::Creator, genotype::FileSystemGenotype) = FileSystemPhenotype(genotype.dirpath, genotype.null_action_space, false)

function (fsphenotype::FileSystemPhenotype)(observation)
    if fsphenotype.run_until_done
        return zeros(Float32, fsphenotype.null_action_space)
    end
    # serialize to obs.jls
    open(joinpath(fsphenotype.dirpath, "observation.jls"), "w") do io
        serialize(io, observation)
    end

    # poll until action.jls exists, then deserialize and return it
    actionfile = joinpath(fsphenotype.dirpath, "action.jls")
    while !isfile(actionfile)
        sleep(0.3)
    end

    action = open(actionfile, "r") do io
        deserialize(io)
    end
    rm(actionfile)
    action
end
