import base64
import signal
import time
import cv2
import imutils
import numpy as np
from fastapi import Response
from nicegui import Client, app, core, run, ui

# In case you don't have a webcam, this will provide a black placeholder image.
black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')

class VideoPlayer:
    def __init__(self, video_source: str = None):
        self.video_source = video_source
        self.video_capture = None
        self.fps = 30  # Default to 30 FPS if the frame rate is not available
        self.video_image = None

    def convert(self, frame: np.ndarray) -> bytes:
        """Converts a frame from OpenCV to a JPEG image."""
        _, imencode_image = cv2.imencode('.jpg', frame)
        return imencode_image.tobytes()

    def setup(self) -> None:
        # OpenCV is used to access the webcam or video file.
        if self.video_source:
            self.video_capture = cv2.VideoCapture(self.video_source)
        else:
            self.video_capture = cv2.VideoCapture(0)

        # Retrieve the frame rate of the video
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS).__round__()
        if self.fps <= 0:
            self.fps = 30  # Default to 30 FPS if the frame rate is not available

        @app.get('/video/frame')
        async def grab_video_frame() -> Response:
            if not self.video_capture.isOpened():
                return placeholder
            # The `video_capture.read` call is a blocking function.
            # So we run it in a separate thread (default executor) to avoid blocking the event loop.
            ret, frame = await run.io_bound(self.video_capture.read)
            
            if not ret:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            if frame is None:
                return placeholder
            
            frame = imutils.resize(frame, height=640)

            # `convert` is a CPU-intensive function, so we run it in a separate process to avoid blocking the event loop and GIL.
            jpeg = await run.cpu_bound(self.convert, frame)
            return Response(content=jpeg, media_type='image/jpeg')

        # For non-flickering image updates and automatic bandwidth adaptation an interactive image is much better than `ui.image()`.
        self.video_image = ui.interactive_image()
        # A timer constantly updates the source of the image.
        # Because data from same paths is cached by the browser,
        # we must force an update by adding the current timestamp to the source.
        ui.timer(interval=1.0 / self.fps, callback=lambda: self.video_image.set_source(f'/video/frame?{time.time()}'))

        async def disconnect() -> None:
            """Disconnect all clients from current running server."""
            for client_id in Client.instances:
                await core.sio.disconnect(client_id)

        def handle_sigint(signum, frame) -> None:
            # `disconnect` is async, so it must be called from the event loop; we use `ui.timer` to do so.
            ui.timer(0.1, disconnect, once=True)
            # Delay the default handler to allow the disconnect to complete.
            ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)

        async def cleanup() -> None:
            # This prevents ugly stack traces when auto-reloading on code change,
            # because otherwise disconnected clients try to reconnect to the newly started server.
            await disconnect()
            # Release the webcam hardware so it can be used by other applications again.
            self.video_capture.release()

        app.on_shutdown(cleanup)
        # We also need to disconnect clients when the app is stopped with Ctrl+C,
        # because otherwise they will keep requesting images which lead to unfinished subprocesses blocking the shutdown.
        signal.signal(signal.SIGINT, handle_sigint)

    def get_ui_component(self):
        return self.video_image

# All the setup is only done when the server starts. This avoids the webcam being accessed
# by the auto-reload main process (see https://github.com/zauberzeug/nicegui/discussions/2321).
app.on_startup(VideoPlayer.setup())

ui.run()