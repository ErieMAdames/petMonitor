#!/usr/bin/python3

import io
import logging
import socketserver
from http import server
from threading import Condition, Thread

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import asyncio
import websockets

PAGE = """\
<html>
<head>
<title>Picamera2 MJPEG Streaming Demo</title>
<script>
let ws;

function initWebSocket() {
    ws = new WebSocket('ws://' + window.location.hostname + ':8765/ws');
    ws.onopen = () => console.log("WebSocket connection established");
    ws.onclose = () => console.log("WebSocket connection closed");
}

function handleKey(event) {
    const key = event.key.toLowerCase();
    if (['w', 'a', 's', 'd'].includes(key)) {
        ws.send(JSON.stringify({ key: key, action: event.type }));
    }
}

window.onload = () => {
    initWebSocket();
    document.addEventListener('keydown', handleKey);
    document.addEventListener('keyup', handleKey);
};
</script>
</head>
<body>
<h1>Picamera2 MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


async def websocket_handler(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        key = data.get("key")
        action = data.get("action")
        if key and action:
            print(f"Key: {key}, Action: {action}")


async def start_websocket_server():
    async with websockets.serve(websocket_handler, "0.0.0.0", 8765):
        await asyncio.Future()  # Run forever


picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
output = StreamingOutput()
picam2.start_recording(JpegEncoder(), FileOutput(output))

try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    http_server_thread = Thread(target=server.serve_forever)
    http_server_thread.daemon = True
    http_server_thread.start()

    asyncio.run(start_websocket_server())
finally:
    picam2.stop_recording()
