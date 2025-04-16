export FileSystemGenotype, FileSystemPhenotype, develop, mutate, GenotypeSum, measure

struct FileSystemGenotype <: AbstractGenotype
    dirpath::String
end
struct FileSystemPhenotype <: AbstractPhenotype
    dirpath::String
end

function mutate(::AbstractRNG, ::AbstractState, ::AbstractPopulation, genotype::FileSystemGenotype)
    @error "mutate() not defined for FileSystemGenotype"
end

develop(::Creator, genotype::FileSystemGenotype) = FileSystemPhenotype(genotype.dirpath)

function (fsphenotype::FileSystemPhenotype)(observation)
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
