import http.server
import socketserver
import os

PORT = 8080
DIRECTORY = "web"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"[*] Frontend Dashboard successfully booted.")
    print(f"[*] Access the UI at: http://localhost:{PORT}")
    print(f"[*] Stop the server with Ctrl+C")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server shutdown gracefully.")
