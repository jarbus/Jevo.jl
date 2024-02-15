# for use in Counters
export AbstractGene, AbstractIndividual, AbstractGeneration
abstract type Abstract                                  end
abstract type AbstractState                             end
abstract type AbstractData                              end
abstract type AbstractPopulation                        end
abstract type AbstractIndividual                        end
abstract type AbstractInteraction                       end
abstract type AbstractOperator                          end
abstract type AbstractRetriever                         end
abstract type AbstractUpdater                           end
abstract type AbstractEnvironment                       end
abstract type AbstractGeneration                        end
abstract type AbstractGene                              end
abstract type AbstractCreator                           end
abstract type AbstractGenotype                          end
abstract type AbstractPhenotype                         end
abstract type AbstractMeasurement   <: AbstractData     end
abstract type AbstractCounter       <: AbstractData     end
abstract type AbstractMatch         <: AbstractData     end
abstract type AbstractMetric        <: AbstractData     end
abstract type AbstractRecord        <: AbstractData     end
abstract type AbstractMutator       <: AbstractOperator end
abstract type AbstractMatchMaker    <: AbstractOperator end # populates the state.matches with matches
abstract type AbstractScorer        <: AbstractOperator end
abstract type AbstractEvaluator     <: AbstractOperator end # Can be distributed
abstract type AbstractCrossover     <: AbstractOperator end
abstract type AbstractSelector      <: AbstractOperator end
abstract type AbstractReplacer      <: AbstractOperator end
abstract type AbstractCheckpointer  <: AbstractOperator end
abstract type AbstractReporter      <: AbstractOperator end
abstract type AbstractAssertor      <: AbstractOperator end # Can apply assertions to objects in state

abstract type AbstractInitializer end
abstract type AbstractWeights end
abstract type AbstractLayer end
abstract type AbstractMutation end
