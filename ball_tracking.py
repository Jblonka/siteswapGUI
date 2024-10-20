from collections import deque
import cv2
from cv2 import VideoCapture
import numpy as np
import os
from typing import List

def ball_tracking(video: VideoCapture, hsv_lower, hsv_upper):
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50, detectShadows=False)
    pts = deque(maxlen=32)
    prev_center = None
    fps = video.get(cv2.CAP_PROP_FPS)
    counter = 0

    # VideoWriter object to save the output video
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frameSize: cv2.typing.Size = (int(width), int(height))
    fourcc = cv2.VideoWriter.fourcc(*'mpv4')
    os.makedirs('result', exist_ok=True)
    out = cv2.VideoWriter(filename='result/tracked_video.mp4', fourcc=fourcc, fps=fps, frameSize=frameSize)

    print('Starting tracking now')

    while True:
        ret, frame = video.read()

        counter += 1
        print('Reading frame', counter)

        if not ret:
            break

        # Apply background subtraction
        fgmask = fgbg.apply(frame)

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        ball_color_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        ball_color_mask = cv2.erode(ball_color_mask, None, iterations=2)
        ball_color_mask = cv2.dilate(ball_color_mask, None, iterations=2)

        # Combine the foreground mask with the ball color mask
        combined_mask = cv2.bitwise_and(fgmask, ball_color_mask)

        center = None
        contours, _ = cv2.findContours(combined_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Only proceed if at least one contour was found
        if len(contours) > 0:
            # Find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                center = prev_center

            # Only proceed if the radius meets a minimum size
            if radius > 10:
                # Draw the circle and centroid on the frame, then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

        pts.appendleft(center)

        # Loop over the set of tracked points
        for i in range(1, len(pts)):
            # If either of the tracked points are None, ignore them
            if pts[i - 1] is None or pts[i] is None:
                continue
            # Otherwise, compute the thickness of the line and draw the connecting lines
            thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        prev_center = center
        out.write(frame)

    video.release()

    out.release()

    print("Video writing completed")

    return 'result/tracked_video.mp4'