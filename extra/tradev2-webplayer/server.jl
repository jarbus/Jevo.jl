#!/usr/bin/env julia

using HTTP
using JSON

function handler(req::HTTP.Request)::HTTP.Response
    if req.method == "GET" && req.target == "/ids"
        return HTTP.Response(200, JSON.json([1]))
    elseif req.method == "POST" && req.target == "/match"
        # ignore incoming matches
        obs = rand(Float32, 12, 12, 3, 1)
        return HTTP.Response(200, JSON.json(obs))
    elseif req.method == "POST" && req.target == "/action"
        # ignore incoming action
        _ = JSON.parse(String(req.body))
        # generate a 128×128×3×1 random color observation
        obs = rand(Float32, 12, 12, 3, 1) |> x->round(x, digits=1)
        return HTTP.Response(200, JSON.json(obs))
    else
        return HTTP.Response(404, "not found")
    end
end

HTTP.serve(handler, "0.0.0.0", 8081)
