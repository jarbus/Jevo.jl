using HTTP, JSON, Serialization

export WebGenotype, WebPhenotype, develop, mutate, WebMatchMaker

# ————————————————————————————————————————————————————————————————
# Genotype & Phenotype
# ————————————————————————————————————————————————————————————————

struct WebGenotype <: AbstractGenotype
    obs_ch::Channel{Any}
    act_ch::Channel{Any}
    null_action_space::Tuple{Vararg{Int}}
end

struct WebPhenotype <: AbstractPhenotype
    obs_ch::Channel{Any}
    act_ch::Channel{Any}
    null_action_space::Tuple{Vararg{Int}}
    run_until_done::Bool
end

function mutate(::AbstractRNG, ::AbstractState, ::AbstractPopulation, ::WebGenotype)
    @error "mutate() not defined for WebGenotype"
end

develop(::Creator, g::WebGenotype) = WebPhenotype(g.obs_ch, g.act_ch, g.null_action_space, false)

function (p::WebPhenotype)(observation)
    if p.run_until_done
        return zeros(Float32, p.null_action_space)
    end
    put!(p.obs_ch, observation)               # enqueue obs
    return take!(p.act_ch)                   # block until client POSTs action
end

# ————————————————————————————————————————————————————————————————
# HTTP handler
# ————————————————————————————————————————————————————————————————

function make_handler(obs_ch, act_ch, id_ch, match_ch)
    return function(req::HTTP.Request)
        @info "HTTP $(req.method) $(req.target)"
        if req.method == "GET" && req.target == "/ids"
            ids = take!(id_ch)
            return HTTP.Response(200, JSON.json(ids))
        elseif req.method == "POST" && req.target == "/match"
            matches = JSON.parse(String(req.body))
            put!(match_ch, matches)
            return HTTP.Response(200, "ok")
        elseif req.method == "POST" && req.target == "/action"
            action = JSON.parse(String(req.body))
            put!(act_ch, action)
            obs = take!(obs_ch)
            return HTTP.Response(200, JSON.json(obs))
        else
            return HTTP.Response(404, "not found")
        end
    end
end

# ————————————————————————————————————————————————————————————————
# MatchMaker operator
# ————————————————————————————————————————————————————————————————

@define_op "WebMatchMaker" "AbstractMatchMaker"
function WebMatchMaker(host::String, port::Int;
                       env_creator=nothing,
                       null_action_space::Tuple{Vararg{Int}},
                       kwargs...)
    obs_ch   = Channel{Any}(1)
    act_ch   = Channel{Any}(1)
    id_ch    = Channel{Any}(1)
    match_ch = Channel{Any}(1)

    # start background HTTP server
    @async HTTP.serve(make_handler(obs_ch, act_ch, id_ch, match_ch), host, port)

    create_op("WebMatchMaker";
        retriever = get_individuals,
        operator  = (s, inds) -> make_web_match(s, inds, obs_ch, act_ch, id_ch, match_ch, env_creator, null_action_space),
        updater   = add_matches!,
        kwargs...,
    )
end

function make_web_match(state::AbstractState,
                        individuals::Vector{I},
                        obs_ch, act_ch, id_ch, match_ch,
                        env_creator,
                        null_action_space::Tuple{Vararg{Int}}) where {I<:AbstractIndividual}
    # ensure env_creator is set
    if isnothing(env_creator)
        ecs = get_creators(AbstractEnvironment, state)
        @assert length(ecs)==1
        env_creator = ecs[1]
    end

    # send available IDs to client
    @info "Putting ids"
    put!(id_ch, [ind.id for ind in individuals])
    @info "Waiting for match"

    # wait for match queue from client
    match_list = take!(match_ch)
    @info "rec matchlist $match_list"

    matches = Match[]
    ctr = get_counter(AbstractMatch, state)
    for match_ids in match_list
        inds = Vector{I}(undef, length(match_ids))
        filled = falses(length(match_ids))
        for (i, id) in enumerate(match_ids)
            if id == -1
                inds[i] = Individual(-1, generation(state), Int[],
                                     WebGenotype(obs_ch, act_ch, null_action_space),
                                     Creator(WebPhenotype))
                filled[i] = true
            else
                for ind in individuals
                    if ind.id == id
                        inds[i] = ind
                        filled[i] = true
                        break
                    end
                end
            end
        end
        @assert all(filled)
        push!(matches, Match(inc!(ctr), inds, env_creator))
    end

    return matches
end
