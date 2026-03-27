"""
Lightweight server for the embedding visualization.
Serves the static frontend and the pre-computed embedding data.

Usage:
    python serve.py [--port 8080]
    Then open http://localhost:8080
"""
import http.server
import socketserver
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Embedding Viz Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on")
    args = parser.parse_args()

    # Serve from the project root so /data/ and /frontend/ are both accessible
    project_root = os.path.dirname(os.path.abspath(__file__))

    # We need to serve index.html from frontend/ and data/ from data/
    # Simplest approach: create a handler that maps routes correctly
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=project_root, **kw)

        def translate_path(self, path):
            # Route / and /index.html to frontend/index.html
            if path == "/" or path == "/index.html":
                return os.path.join(project_root, "frontend", "index.html")
            # Route /data/* to data/*
            if path.startswith("/data/"):
                return os.path.join(project_root, path.lstrip("/"))
            # Default
            return os.path.join(project_root, "frontend", path.lstrip("/"))

        def log_message(self, format, *a):
            # Quieter logging
            if self.path != "/favicon.ico":
                super().log_message(format, *a)

    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        print(f"\n  ╔══════════════════════════════════════╗")
        print(f"  ║  Embedding Vector Space Viewer        ║")
        print(f"  ║  http://localhost:{args.port:<19} ║")
        print(f"  ║  Ctrl+C to stop                       ║")
        print(f"  ╚══════════════════════════════════════╝\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
            httpd.shutdown()

if __name__ == "__main__":
    main()
