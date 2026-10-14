"""本機測試伺服器: py dev_server.py -> http://localhost:8000"""
import json, os, sys
from http.server import SimpleHTTPRequestHandler, HTTPServer
from analysis_core import handle

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PORT = int(os.environ.get("PORT", 8000))

class DevHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        clean_path = self.path.split("?")[0].rstrip("/")
        if clean_path.endswith("api/analyze") or clean_path == "" or clean_path == "/api/analyze":
            n = int(self.headers.get("content-length", 0) or 0)
            try:
                body_raw = self.rfile.read(n) if n > 0 else b"{}"
                payload = json.loads(body_raw.decode("utf-8", errors="replace"))
            except Exception:
                payload = {}
            code, out = handle(payload)
            body = json.dumps(out).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404, "Endpoint not found")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Dev server running at http://localhost:{PORT}")
    try:
        httpd = HTTPServer(("", PORT), DevHandler)
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dev server...")
        httpd.server_close()
