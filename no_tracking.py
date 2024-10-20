import cv2

cap = cv2.VideoCapture("samples/Sample1.mp4")
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

fourcc = cv2.VideoWriter.fourcc(*'mpv4')
out = cv2.VideoWriter('output5.mp4', fourcc, 30, (width, height), isColor=True)

while cap.isOpened():
    # get validity boolean and current frame
    ret, frame = cap.read()

    # if valid tag is false, loop back to start
    if not ret:
        break
    else:
        # frame = cv2.resize(frame, (width, height))
        out.write(frame)


cap.release()
out.release()