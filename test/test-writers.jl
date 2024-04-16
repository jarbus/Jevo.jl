@testset "checkpoint" begin

    checkpointname = "test-checkpoint.jls"
    state = State(rng, AbstractCreator[], [Checkpointer(checkpointname, interval=5)])
    run!(state, 10)
    @test generation(state) == 11
    state = restore_from_checkpoint(checkpointname)
    @test generation(state) == 6
end

@testset "HDF5Logger" begin
  rm("statistics.h5", force=true)
  with_logger(Jevo.HDF5Logger("statistics.h5")) do
      m = Measurement(GenotypeSum, 1, 1)
      sm = StatisticalMeasurement(GenotypeSum, [1,2,3], 2)
      @h5 m
      @h5 sm
  end
  h5open("statistics.h5", "r") do io
      @test haskey(io, "1/GenotypeSum/value")
      @test haskey(io, "1/GenotypeSum/min")
      @test haskey(io, "1/GenotypeSum/max")
      @test haskey(io, "1/GenotypeSum/mean")
      @test haskey(io, "1/GenotypeSum/std")
      @test haskey(io, "1/GenotypeSum/n_samples")
  end
  rm("statistics.h5", force=true)
end
@testset "JevoLogger" begin
  rm("statistics.h5", force=true)
  with_logger(Jevo.JevoLogger()) do
      sm = StatisticalMeasurement(GenotypeSum, [1,2,3], 1)
      # log to hdf5 only
      log(sm, true, false, false)
      # log to text only
      log(sm, false, true, false)
  end
  rm("statistics.h5", force=true)
end
