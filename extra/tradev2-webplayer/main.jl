module InteractServer

using HTTP, HTTP.URIs, Sockets, JSON
using Images, ColorTypes, FileIO
using Base64: base64encode

# === Utility ===
function flatten_obs(obs)
    # obs is nested W×H×3×1 Vec{Any} of floats in [0,1]
    W = length(obs[1][1][1])
    H = length(obs[1][1])
    # build direct Float32 array W×H×3
    arr = Array{Float32}(undef, W, H, 3)
    for x in 1:W, y in 1:H, c in 1:3
        arr[x, y, c] = Float32(obs[1][c][y][x])
    end
    # permute to channel-first (3×W×H) for colorview
    arr_perm = permutedims(arr, (3,1,2))    # 3×W×H
    # create an RGB image
    img = colorview(RGB, arr_perm)          # W×H RGB image
    # reshape back to W×H×3
    arr3 = permutedims(channelview(img), (2,3,1))  # W×H×3
    # encode PNG into buffer
    buf = IOBuffer()
    save(Stream{format"PNG"}(buf), img)
    return base64encode(take!(buf))         # return base64 string
end

# === HTML Generators ===
function chooser_html(ids)
    items = ["<li><button onclick='choose($i)'>$i</button></li>" for i in ids]
    list  = join(items)
    return """
<!doctype html>
<html>
  <head><meta charset=\"utf-8\"><title>Select ID</title>
    <style>
      body { font-family: sans-serif; text-align: center }
      button { font-size: 1.2rem; padding: .4rem .8rem; margin: .2rem }
    </style>
  </head>
  <body>
    <h2>Select an ID to interact with</h2>
    <ul style=\"list-style:none; padding:0\">$list</ul>
    <script>
      let sent = false;
      function choose(id) {
        if (sent) return;
        sent = true;
        fetch('/match', {
          method: 'POST',
          headers: {'Content-Type':'application/x-www-form-urlencoded'},
          body: new URLSearchParams({selected_id: id})
        })
        .then(r => r.text())
        .then(html => { document.open(); document.write(html); document.close(); });
      }
    </script>
  </body>
</html>"""
end

function observation_html(; inline_image::String, x::Int=0, y::Int=0, pick::Int=0, place::Int=0)
    return """
<!doctype html>
<html>
  <head><meta charset=\"utf-8\"><title>Interact</title>
    <style>
      .container { text-align: center; margin-top: 20px; }
      #obs-img { width: 400px; height: auto; }
      .vector-container { display: flex; justify-content: center; margin-top: 20px; }
      .vector-box { width: 60px; height: 60px; border: 1px solid #000; display: flex;
                     align-items: center; justify-content: center; margin: 0 2px;
                     font-size: 22px; font-weight: bold; user-select: none; }
      .controls { display: flex; justify-content: center; margin-top: 20px; }
      .controls button { margin: 0 10px; padding: 10px 20px; font-size: 16px; }
    </style>
  </head>
  <body>
    <div class=\"container\">
      <img id=\"obs-img\" src=\"data:image/png;base64,$inline_image\" alt=\"Observation\">
      red
      <div class=\"vector-container\">
        <div class=\"vector-box\" id=\"xbox\">$x</div>
        <div class=\"vector-box\" id=\"ybox\">$y</div>
        <div class=\"vector-box\" id=\"pickbox\">$pick</div>
        <div class=\"vector-box\" id=\"placebox\">$place</div>
      </div>
      green
      <div class=\"controls\"><button id=\"enter\">Enter</button></div>
    </div>
    <script>
      const xbox = document.getElementById('xbox');
      const ybox = document.getElementById('ybox');
      const pickbox = document.getElementById('pickbox');
      const placebox = document.getElementById('placebox');
      const clamp = v => Math.max(-1, Math.min(1, v));

      document.addEventListener('keydown', e => {
        let x = parseInt(xbox.textContent);
        let y = parseInt(ybox.textContent);
        if (e.key === 'd')      { y = clamp(y + 1); ybox.textContent = y; }
        else if (e.key === 'a') { y = clamp(y - 1); ybox.textContent = y; }
        else if (e.key === 'w') { x = clamp(x - 1); xbox.textContent = x; }
        else if (e.key === 's') { x = clamp(x + 1); xbox.textContent = x; }
      });

      [pickbox, placebox].forEach(box => {
        box.addEventListener('wheel', e => {
          e.preventDefault();
          let v = parseInt(box.textContent);
          v += (e.deltaY < 0 ? 1 : -1);
          box.textContent = v;
        });
      });

      document.getElementById('enter').addEventListener('click', () => {
        const params = new URLSearchParams({ x: xbox.textContent, y: ybox.textContent,
                                           pick: pickbox.textContent, place: placebox.textContent });
        fetch('/action', {
          method: 'POST',
          headers: {'Content-Type':'application/x-www-form-urlencoded'},
          body: params
        })
        .then(r => r.json())
        .then(obj => { document.getElementById('obs-img').src =
                        'data:image/png;base64,' + obj['image']; });
      });
    </script>
</html>"""
end

# === Request Handler ===
function handler(req::HTTP.Request)
    if req.method == "POST" && req.target == "/match"
        params = URIs.queryparams(String(req.body))
        id     = parse(Int, params["selected_id"])
        remote = HTTP.request("POST", "http://localhost:8081/match",
                              ["Content-Type"=>"application/json"],
                              JSON.json([[id, -1]]))
        parsed = JSON.parse(String(remote.body))
        b64    = flatten_obs(parsed)
        return HTTP.Response(200, ["Content-Type"=>"text/html"],
                              observation_html(inline_image=b64))

    elseif req.method == "POST" && req.target == "/action"
        params        = URIs.queryparams(String(req.body))
        x, y, pick, place = parse.(Int,
                          [params["x"], params["y"], params["pick"], params["place"]])
        remote        = HTTP.request("POST", "http://localhost:8081/action",
                              ["Content-Type"=>"application/json"],
                              JSON.json(Float32[x, y, pick, place]))
        parsed        = JSON.parse(String(remote.body))
        b64           = flatten_obs(parsed)
        return HTTP.Response(200, ["Content-Type"=>"application/json"],
                              JSON.json(Dict("image"=>b64)))

    elseif req.method == "GET" && req.target == "/"
        resp = try HTTP.get("http://localhost:8081/ids") catch e
            return HTTP.Response(500, "Error fetching IDs: $(e)")
        end
        ids  = JSON.parse(String(resp.body))
        if !isempty(ids)
            return HTTP.Response(200, ["Content-Type"=>"text/html"], chooser_html(ids))
        else
            remote = HTTP.request("POST", "http://localhost:8081/action",
                                  ["Content-Type"=>"application/json"],
                                  JSON.json(Float32[]))
            parsed = JSON.parse(String(remote.body))
            b64    = flatten_obs(parsed)
            return HTTP.Response(200, ["Content-Type"=>"text/html"],
                                  observation_html(inline_image=b64))
        end       
    else
        return HTTP.Response(404, "Not found")
    end
end

function main()
    HTTP.serve(handler, Sockets.localhost, 8080)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
