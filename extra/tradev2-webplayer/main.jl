using HTTP, Sockets, Images, FileIO, Colors, Random

const pic_path = "picture.jpg"

# Create picture.jpg with random data if it doesn't exist on startup
if !isfile(pic_path)
    img = rand(RGB{N0f8}, 200, 200)
    save(pic_path, img)
end

# Helper function to parse URL-encoded form data
function parse_form_data(body::String)
    params = Dict{String, String}()
    for pair in split(body, "&")
        kv = split(pair, "=")
        if length(kv) == 2
            params[kv[1]] = kv[2]
        end
    end
    return params
end

function request_handler(req)
    # Serve picture if the target starts with "/picture.jpg" (to handle query parameters)
    if startswith(req.target, "/picture.jpg")
        if isfile(pic_path)
            content = read(pic_path)
            return HTTP.Response(200, [
                "Content-Type" => "image/jpeg",
                "Cache-Control" => "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma" => "no-cache"
            ], content)
        else
            return HTTP.Response(404, ["Content-Type" => "text/plain"], "Picture not found")
        end
    end

    # Default values
    x_val = 0; y_val = 0; pick_val = 0; place_val = 0
    if req.method == "POST"
        body_str = String(req.body)
        params = parse_form_data(body_str)
        try   x_val = parse(Int, get(params, "x", "0")); catch; end
        try   y_val = parse(Int, get(params, "y", "0")); catch; end
        try   pick_val = parse(Int, get(params, "pick", "0")); catch; end
        try   place_val = parse(Int, get(params, "place", "0")); catch; end

        # Delete the current picture, wait one second, then generate a new random picture.
        if isfile(pic_path)
            rm(pic_path)
        end
        sleep(1)
        img = fill(rand(RGB{N0f8}), 200, 200)
        save(pic_path, img)
    end

    # Poll until the picture file exists
    while !isfile(pic_path)
        sleep(0.5)
    end

    # Build the HTML page with inline JavaScript.
    # Notice that we use string interpolation (e.g. $(x_val)) and
    # the response is returned with headers before the body.
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Julia Web Server</title>
      <style>
        .container { text-align: center; margin-top: 50px; }
        .vector-container { display: flex; justify-content: center; margin-top: 20px; }
        .vector-box { width: 50px; height: 50px; border: 1px solid #000; margin: 0; padding: 0; box-sizing: border-box; }
        .controls { display: flex; justify-content: center; align-items: center; margin-top: 20px; }
        .controls button { margin: 0 10px; padding: 10px 20px; font-size: 16px; }
      </style>
    </head>
    <body>
      <div class="container">
        <img src="/picture.jpg?t=$(rand())" alt="Picture">
        <div class="vector-container">
          <div class="vector-box" id="xbox"></div>
          <div class="vector-box" id="ybox"></div>
          <div class="vector-box" id="pickbox"></div>
          <div class="vector-box" id="placebox"></div>
        </div>
        <div class="controls">
          <button id="enter">Enter</button>
          <button id="done">Done</button>
        </div>
      </div>
      <script>
        // Initialize variables using server-provided values
        var x = $(x_val);
        var y = $(y_val);
        var pick = $(pick_val);
        var place = $(place_val);

        function clamp(val, min, max) {
          return Math.max(min, Math.min(val, max));
        }

        function updateDisplay() {
          let xText = x === 1 ? "Right" : x === -1 ? "Left" : "Neutral";
          let yText = y === 1 ? "Up" : y === -1 ? "Down" : "Neutral";
          document.getElementById("xbox").innerText = "X: " + xText;
          document.getElementById("ybox").innerText = "Y: " + yText;
          document.getElementById("pickbox").innerText = "Pick: " + pick;
          document.getElementById("placebox").innerText = "Place: " + place;
        }

        // WASD keys update X and Y; Enter sends POST
        document.addEventListener("keydown", function(e) {
          if(e.key === "Enter") {
            e.preventDefault();
            sendFrame();
            return;
          }
          switch(e.key.toLowerCase()) {
            case "w": y = clamp(y + 1, -1, 1); break;
            case "s": y = clamp(y - 1, -1, 1); break;
            case "a": x = clamp(x - 1, -1, 1); break;
            case "d": x = clamp(x + 1, -1, 1); break;
          }
          updateDisplay();
        });

        // Scroll wheel updates for 'pick' and 'place'
        document.getElementById("pickbox").addEventListener("wheel", function(e) {
          e.preventDefault();
          if(e.deltaY < 0) { pick = clamp(pick + 1, -10, 10); }
          else if(e.deltaY > 0) { pick = clamp(pick - 1, -10, 10); }
          updateDisplay();
        });
        document.getElementById("placebox").addEventListener("wheel", function(e) {
          e.preventDefault();
          if(e.deltaY < 0) { place = clamp(place + 1, -10, 10); }
          else if(e.deltaY > 0) { place = clamp(place - 1, -10, 10); }
          updateDisplay();
        });

        // Function to send a POST request with the current frame values
        function sendFrame() {
          let data = new URLSearchParams();
          data.append("x", x);
          data.append("y", y);
          data.append("pick", pick);
          data.append("place", place);
          fetch("/", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: data.toString()
          })
          .then(response => response.text())
          .then(html => {
            document.open();
            document.write(html);
            document.close();
          });
        }
        document.getElementById("enter").addEventListener("click", sendFrame);
        window.onload = updateDisplay;
      </script>
    </body>
    </html>
    """
    return HTTP.Response(200, ["Content-Type" => "text/html"], html)
end

HTTP.serve(request_handler, Sockets.localhost, 8080)
