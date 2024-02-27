abstract type AbstractJevo                              end
abstract type AbstractState         <: AbstractJevo     end
abstract type AbstractData          <: AbstractJevo     end
abstract type AbstractPopulation    <: AbstractJevo     end
abstract type AbstractIndividual    <: AbstractJevo     end
abstract type AbstractInteraction   <: AbstractJevo     end
abstract type AbstractOperator      <: AbstractJevo     end
abstract type AbstractRetriever     <: AbstractJevo     end
abstract type AbstractUpdater       <: AbstractJevo     end
abstract type AbstractEnvironment   <: AbstractJevo     end
abstract type AbstractGeneration    <: AbstractJevo     end
abstract type AbstractGene          <: AbstractJevo     end
abstract type AbstractCreator       <: AbstractJevo     end
abstract type AbstractGenotype      <: AbstractJevo     end
abstract type AbstractPhenotype     <: AbstractJevo     end
abstract type AbstractCounter       <: AbstractData     end
abstract type AbstractMatch         <: AbstractData     end
abstract type AbstractRecord        <: AbstractData     end
# populates the state.matches with matches
abstract type AbstractMatchMaker    <: AbstractOperator end
abstract type AbstractPerformer     <: AbstractOperator end
# Evaluator should enable distributed computing
abstract type AbstractEvaluator     <: AbstractOperator end
abstract type AbstractSelector      <: AbstractOperator end
abstract type AbstractReproducer    <: AbstractOperator end
abstract type AbstractMutator       <: AbstractOperator end
abstract type AbstractMetric        <: AbstractData     end
abstract type AbstractMeasurement   <: AbstractData     end
abstract type AbstractCheckpointer  <: AbstractOperator end
abstract type AbstractReporter      <: AbstractOperator end
abstract type AbstractAssertor      <: AbstractOperator end # Can apply assertions to objects in state

# Export all abstract types
for name in names(@__MODULE__, all=true, imported=false)
    obj = getfield(@__MODULE__, name)
    if isa(obj, DataType) && isabstracttype(obj)
        @eval export $name
    end
end

"Override Base.show to avoid printing empty containers"
function Base.show(io::IO, jevos::Vector{<:AbstractJevo})
    if isempty(jevos)
    elseif !isempty(jevos)
        print(io, typeof(jevos[1]))
        print(io, "[")
        for jevo in jevos[1:end-1]
            print(io, jevo, ", ")
        end
        print(io, jevos[end], "]")
    end
end
function Base.show(io::IO, jevo::Dict{Any, <:AbstractJevo})
    if isempty(jevo)
    elseif !isempty(jevo)
        print(io, typeof(jevo))
        print(io, "{")
        for (k, v) in jevo
            print(io, k, " => ", v, ", ")
        end
        print(io, "}")
    end
end
