@testset "checkpoint" begin

    checkpointname = "test-checkpoint.jls"
    state = State("", rng, AbstractCreator[], [Checkpointer(checkpointname, interval=5)], counters=default_counters())
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
      @test haskey(io, "iter/1/GenotypeSum")
      @test haskey(io, "iter/2/GenotypeSum/min")
      @test haskey(io, "iter/2/GenotypeSum/max")
      @test haskey(io, "iter/2/GenotypeSum/mean")
      @test haskey(io, "iter/2/GenotypeSum/std")
      @test haskey(io, "iter/2/GenotypeSum/n_samples")
  end
  rm("statistics.h5", force=true)
end
@testset "JevoLogger" begin
  rm("statistics.h5", force=true)
  with_logger(Jevo.JevoLogger()) do
      sm = StatisticalMeasurement(GenotypeSum, [1,2,3], 1)
      # log to hdf5 only
      log(sm, true, false, false)
      # check that statistics.h5 is not empty
      h5open("statistics.h5", "r") do io
          @test length(io) > 0
      end
      # log to text only
      log(sm, false, true, false)
      # check that run.log is not empty
      open("run.log", "r") do io
          @test length(read(io, String)) > 0
      end
  end
  rm("statistics.h5", force=true)
end
