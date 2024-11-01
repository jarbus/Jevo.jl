export CarRacingV3
using PyCall
global gymnasium


Base.@kwdef mutable struct CarRacingV3 <: AbstractEnvironment
    env = nothing
    done::Bool = false
end

"""
Wrappers to use:
    - FrameStack
    - TimeLimit
    - Record video
    - Resize Observation?
"""
done(env::CarRacingV3) = env.done

function step!(env::CarRacingV3, ids::Vector{Int}, phenotypes::Vector)
    if isnothing(env.env)
        gymnasium = pyimport("gymnasium")
        @info("Creating CarRacing-v3 environment")
        _env = gymnasium.make("CarRacing-v3")
        @info("Wrapping environment")
        _env = gymnasium.wrappers.TimeLimit(_env, max_episode_steps=1000)
        @info("Wrapping environment")
        _env = gymnasium.wrappers.FrameStackObservation(_env, stack_size=4)
        env.env = _env
    end
    obs = env.env.reset()
    # step
    env.done = true
end
#= env = gymnasium.make("CarRacing-v3") =#
#= gymnasium.wrappers.TimeLimit(env, max_episode_steps=1000) =#
#= gymnasium.wrappers.FrameStackObservation(env, stack_size=4) =#

#= observation, info = env.reset() =#
#==#
#= episode_over = false =#
#= while !episode_over =#
#=     action = env.action_space.sample()  # agent policy that uses the observation and info =#
#=     observation, reward, terminated, truncated, info = env.step(action) =#
#==#
#=     episode_over = terminated or truncated =#
#= end =#
