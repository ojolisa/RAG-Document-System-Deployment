#!/usr/bin/env python3
"""
Simple HTTP server to serve the frontend and handle CORS for API calls
"""
import http.server
import socketserver
import webbrowser
import os
import sys
from urllib.parse import urlparse, parse_qs
import json
import requests


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods',
                         'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def main():
    PORT = 3000

    # Change to the frontend directory
    frontend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(frontend_dir)

    # Create server
    with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
        print(f"Frontend server running at http://localhost:{PORT}")
        print("Make sure your RAG API is running on http://localhost:8000")
        print("\nTo start the API, run the start_api.bat file in the api folder")
        print("Press Ctrl+C to stop the server")

        # Open browser automatically
        webbrowser.open(f'http://localhost:{PORT}')

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()
