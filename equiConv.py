import cv2
import numpy as np
import math
import sys
from omnicv import fisheyeImgConv

# Video path


# Open the video file
cap = cv2.VideoCapture('equirect.mp4')

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    sys.exit(1)

# Creating a mapper object
mapper = fisheyeImgConv()

# Get the screen resolution
screen_width = 1920  # Replace with your screen width
screen_height = 1080  # Replace with your screen height

while True:
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop at the end of the video

    # Generating cubemap from the current frame
    cubemap = mapper.equirect2cubemap(frame, side=512, dice=1)  # Adjust the side resolution as needed

    # Resize the cubemap to fit your screen
    cubemap = cv2.resize(cubemap, (screen_width, screen_height))

    cv2.imshow("Cubemap", cubemap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit the loop

cap.release()
cv2.destroyAllWindows()
