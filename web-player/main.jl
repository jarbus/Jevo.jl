using HTTP, HTTP.WebSockets, URIParser

# Global variable to store the selected id
const selected_id = Ref{Int}(0)

# Helper function to parse query parameters from a query string
function queryparams(query::Union{String, Nothing})
    d = Dict{String, String}()
    if query === nothing || isempty(query)
        return d
    end
    for pair in split(query, "&")
        kv = split(pair, "=")
        if length(kv) == 2
            d[kv[1]] = kv[2]
        end
    end
    return d
end

# HTTP request handler on port 8080
function http_handler(req)
    println("HTTP Request: ", req.target)
    if req.target in ["/", "/index.html"]
        return HTTP.Response(200, Dict("Content-Type" => "text/html"), read("index.html", String))
    elseif startswith(req.target, "/setid")
        # Parse query parameters from the URL
        uri = URI("http://dummy" * req.target)
        q = queryparams(uri.query)
        if haskey(q, "id")
            selected_id[] = parse(Int, q["id"])
            println("Selected id set to ", selected_id[])
        end
        # Redirect to the game page
        return HTTP.Response(302, Dict("Location" => "/game"), "")
    elseif req.target in ["/game", "/game.html"]
        return HTTP.Response(200, Dict("Content-Type" => "text/html"), read("game.html", String))
    else
        return HTTP.Response(404, Dict(), "Not Found")
    end
end

# Start the HTTP server on port 8080 (serving index and game pages)
@async HTTP.serve(http_handler, "0.0.0.0", 8080)

# Start the WebSocket server on port 8081
WebSockets.listen("0.0.0.0", 8081) do ws
    # On connection, send the current selected id to the client
    WebSockets.send(ws, string(selected_id[]))
    # Process incoming messages from the client
    for msg in ws
        if msg == "increment"
            selected_id[] += 1
            println("Incremented id to ", selected_id[])
            WebSockets.send(ws, string(selected_id[]))
        else
            println("Received unknown message: ", msg)
        end
    end
end
