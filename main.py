import argparse
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from nicegui import app, ui
from cv2 import VideoCapture
from pathlib import Path

from ball_tracking import ball_tracking

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

green_lower = (29, 86, 6)
green_upper = (100, 255, 150)

if not args.get("video", False):
    raise ValueError("No video file provided")

# Serve the directory containing the video file
app.mount("/static", StaticFiles(directory="result"), name="static")

def start_tracking():
    video_path = ball_tracking(VideoCapture(args["video"]), green_lower, green_upper)
    print(f"Video saved to {video_path}")
    update_ui()

def update_ui():
    with ui.column():
        ui.label('Video Player')
        ui.video('/static/tracked_video.mp4')

ui.button('Start tracking', on_click=start_tracking)


ui.run()