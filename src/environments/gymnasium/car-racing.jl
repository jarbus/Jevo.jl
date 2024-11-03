export CarRacingV3
using Random
using PythonCall
global gymnasium

mutable struct CarRacingV3 <: AbstractEnvironment
    n_envs::Int
    n_steps::Int
    n_stack::Int
    env
    prev_obs
    done::Bool
end

function CarRacingV3(n_envs::Int, n_steps::Int, skip::Int)
    return CarRacingV3(n_envs, n_steps, skip, nothing, nothing, false)
end

#= function sample_once_per_column(rng, matrix::Matrix) =#
#=     n_cols = size(matrix, 2) =#
#=     samples = Vector{Int}(undef, n_cols) =#
#==#
#=     for j in 1:n_cols =#
#=         # Create a categorical distribution for each column =#
#=         dist = Categorical(matrix[:, j]) =#
#=         samples[j] = rand(rng, dist) =#
#=     end =#
#==#
#=     return samples =#
#= end =#

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
        # This works, and isn't much of a slowdown, from what i can tell
        @pyexec"""
        def make_single_env(gym, np, n_steps, n_stack):
            def anonymous_env_creator():
                _env = gym.make("CarRacing-v3")
                _env = gym.wrappers.DtypeObservation(_env, np.float32)
                _env = gym.wrappers.TimeLimit(_env, max_episode_steps=n_steps)
                _env = gym.wrappers.RescaleObservation(_env, min_obs=0.0, max_obs=1.0)
                _env = gym.wrappers.FrameStackObservation(_env, stack_size=n_stack)
                return _env
            return anonymous_env_creator
        """ => make_single_env
        gym = pyimport("gymnasium")
        np = pyimport("numpy")
        _env = gym.vector.SyncVectorEnv([
            make_single_env(gym, np, env.n_steps, env.n_stack) for _ in 1:env.n_envs])
        env.env = _env
        obs, info = env.env.reset()
        obs = pyconvert(Array{Float32}, obs) |> cu
        obs = permutedims(reshape(obs, size(obs, 1), size(obs, 2)*size(obs, 5), size(obs, 3), size(obs, 4)), (3, 4, 2, 1))
        env.prev_obs = obs
    end
    start_time = time()
    actions = phenotypes[1](env.prev_obs) |> cpu
    forward_pass_time = time()
    actions = pyimport("numpy").array(actions).T
    numpy_import_time = time()

    reward = 0.0
    obs = nothing
    for _ in 1:env.n_stack
        obs, _reward, terminated, truncated, info = env.env.step(actions)
        terminated, truncated = PyArray(terminated), PyArray(truncated)

        if any(terminated) != all(terminated) || any(truncated) != all(truncated)
            error("Some environments terminated while others didn't")
        end
        env.done = env.done || any(terminated) || any(truncated)
        reward += sum(PyArray(_reward))
    end
    env_process_time = time()
    @info("Forward pass time: $(forward_pass_time - start_time)")
    @info("Numpy import time: $(numpy_import_time - forward_pass_time)")
    @info("Env process time: $(env_process_time - numpy_import_time)")
    @assert !isnothing(obs)
    obs = PyArray(obs) |> cu
    obs = permutedims(reshape(obs, size(obs, 1), size(obs, 2)*size(obs, 5), size(obs, 3), size(obs, 4)), (3, 4, 2, 1))
    env.prev_obs = obs
    [Interaction(ids[1], [], reward)]
end
