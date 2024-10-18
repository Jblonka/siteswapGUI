#!/usr/bin/env python3
import base64
import signal
import time

import argparse

import cv2
import imutils
import numpy as np
from fastapi import Response

from nicegui import Client, app, core, run, ui

# In case you don't have a webcam, this will provide a black placeholder image.
black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

video_is_active = True
fps = 30

def convert(frame: np.ndarray) -> bytes:
    """Converts a frame from OpenCV to a JPEG image.

    This is a free function (not in a class or inner-function),
    to allow run.cpu_bound to pickle it and send it to a separate process.
    """
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()

def get_interval():
    if fps <= 0:
        return 1.0 / 30
    return 1.0 / fps

def setup() -> None:
    # OpenCV is used to access the webcam.
    if args["video"]:
        video_capture = cv2.VideoCapture(args["video"])
    else:
        video_capture = cv2.VideoCapture(0)

    # # Retrieve the frame rate of the video
    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    # if fps <= 0:
    #     fps = 30  # Default to 30 FPS if the frame rate is not available

    @app.get('/video/frame')
    # Thanks to FastAPI's `app.get` it is easy to create a web route which always provides the latest image from OpenCV.
    async def grab_video_frame() -> Response:
        if not video_capture.isOpened():
            return placeholder
        # The `video_capture.read` call is a blocking function.
        # So we run it in a separate thread (default executor) to avoid blocking the event loop.
        ret, frame = await run.io_bound(video_capture.read)
        
        if not ret:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        if frame is None:
            return placeholder
        
        frame = imutils.resize(frame, height=640)

        # `convert` is a CPU-intensive function, so we run it in a separate process to avoid blocking the event loop and GIL.
        jpeg = await run.cpu_bound(convert, frame)
        return Response(content=jpeg, media_type='image/jpeg')

    # For non-flickering image updates and automatic bandwidth adaptation an interactive image is much better than `ui.image()`.
    video_image = ui.interactive_image()

    def update_video_frame():
        if video_is_active:
            video_image.set_source(f'/video/frame?{time.time()}')
    
    # A timer constantly updates the source of the image.
    # Because data from same paths is cached by the browser,
    # we must force an update by adding the current timestamp to the source.
    ui.timer(interval=get_interval(), callback=update_video_frame)

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
        video_capture.release()

    app.on_shutdown(cleanup)
    # We also need to disconnect clients when the app is stopped with Ctrl+C,
    # because otherwise they will keep requesting images which lead to unfinished subprocesses blocking the shutdown.
    signal.signal(signal.SIGINT, handle_sigint)

def stop_video():
    global video_is_active
    video_is_active = False

def resume_video():
    global video_is_active
    video_is_active = True

def change_fps(value):
    global fps
    fps = value
    print(fps)

# All the setup is only done when the server starts. This avoids the webcam being accessed
# by the auto-reload main process (see https://github.com/zauberzeug/nicegui/discussions/2321).
app.on_startup(setup)

app.on_exception(app.stop())

# Everything above is just to play the video
# ToDo figure out how to move the video player in its own class
with ui.row():
    ui.button('Pause', on_click=stop_video).props('outline')
    ui.button('Resume', on_click=resume_video).props('outline')
    ui.button('Reset FPS', on_click=lambda: change_fps(30)).props('outline')
    ui.slider(min=0, max=100, value=fps, on_change=lambda e: change_fps(e.value))

ui.run()