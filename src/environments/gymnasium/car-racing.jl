export CarRacingV3
using Images, FileIO
using Random
using PythonCall

global gym, car_racing_env, record_car_racing_env, np

mutable struct CarRacingV3 <: AbstractEnvironment
    n_envs::Int
    n_steps::Int
    n_stack::Int
    record_prefix::String  
    env
    prev_obs
    step::Int
    done::Int
    reward::Float32
    prev_rewards
    gym
    np
end

function CarRacingV3(n_envs::Int, n_steps::Int, n_stack::Int, record_prefix::String="")
    return CarRacingV3(n_envs, n_steps, n_stack, record_prefix, nothing, nothing, 1, 0, 0, Float64[], nothing, nothing)
end

function done(env::CarRacingV3)
    d = env.done == env.n_envs
    d && !isempty(env.record_prefix) && env.env.close()
    d
end


function get_car_racing_imports(env::CarRacingV3)
    if !isdefined(Jevo, :car_racing_env)
        @info "Creating CarRacing environment"
        gym = pyimport("gymnasium")
        _env = gym.make("CarRacing-v3")
        _env = gym.wrappers.ResizeObservation(_env, (64, 64))
        _env = gym.wrappers.TimeLimit(_env, max_episode_steps=pyint(env.n_steps))
        _env = gym.wrappers.FrameStackObservation(_env, stack_size=pyint(env.n_stack))
        Jevo.car_racing_env = _env
        Jevo.gym = gym
        Jevo.np = pyimport("numpy")
    end
    if !isempty(env.record_prefix)
        if !isdefined(Jevo, :record_car_racing_env)
            @info "Creating CarRacing environment for recording"
            gym = pyimport("gymnasium")
            _env = gym.make("CarRacing-v3", render_mode="rgb_array")
            _env = gym.wrappers.ResizeObservation(_env, (64, 64))

            _env = gym.wrappers.TimeLimit(_env, max_episode_steps=pyint(env.n_steps))
            _env = gym.wrappers.FrameStackObservation(_env, stack_size=pyint(env.n_stack))
            _env = gym.wrappers.RecordVideo(_env, pystr("video/"), name_prefix=pystr(env.record_prefix * "-$(time())"),  episode_trigger=@pyeval("lambda x: True"))
            Jevo.record_car_racing_env = _env
        end
        _env = Jevo.record_car_racing_env
    else
        _env = Jevo.car_racing_env
    end
    Jevo.gym, _env, Jevo.np
end

function step!(env::CarRacingV3, ids::Vector{Int}, phenotypes::Vector)
    if isnothing(env.env)
        env.gym, env.env, env.np = get_car_racing_imports(env)
        obs, info = env.env.reset()
        env.prev_obs = PyArray(obs) |> Array |> deepcopy
    end

    obs = env.prev_obs
    frames = [obs[i, :, :, :] for i in 1:env.n_stack]
    obs = cat(frames..., dims=3)
    obs = obs ./ 255f0
    obs = reshape(obs, 64, 64, 3*env.n_stack, 1)
    action = phenotypes[1].chain(obs)
    action = action[:,1]
    action = env.np.array(action).T

    obs, reward, terminated, truncated, info = env.env.step(action)
    env.step += 1
    reward = pyconvert(Float32, reward)
    push!(env.prev_rewards, reward)

    if Bool(terminated) || Bool(truncated) || 
        length(env.prev_rewards) > 20 && all(env.prev_rewards[end-20:end] .< 0)
        env.done += 1
        env.step = 1
        env.prev_rewards = Float64[]
        env.prev_obs = nothing
        return [Interaction(ids[1], [], reward)]
    end
    env.prev_obs = PyArray(obs) |> Array |> deepcopy
    [Interaction(ids[1], [], reward)]
end
