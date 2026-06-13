import asyncio
import base64
import threading
import cv2
import websockets


class ImageWebSocketServer:

    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self._thread = threading.Thread(
            target=self._run,
            daemon=True
        )

    def start(self):
        self._thread.start()

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(
            self._handler,
            self.host,
            self.port
        ):
            print(f'Image WebSocket server on ws://{self.host}:{self.port}')
            await asyncio.Future()  # run forever

    async def _handler(self, websocket):
        self.clients.add(websocket)
        print(f'GUI connected: {websocket.remote_address}')
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)
            print(f'GUI disconnected')

    def send_frame(self, frame):
        if not self.clients:
            return

        _, jpeg = cv2.imencode(
            '.jpg',
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 60]
        )
        b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        asyncio.run_coroutine_threadsafe(
            self._broadcast(b64),
            self.loop
        )

    async def _broadcast(self, b64):
        if not self.clients:
            return
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(b64)
            except Exception:
                disconnected.add(client)
        self.clients -= disconnected