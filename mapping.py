#!/usr/bin/env/python
import cv2
import numpy as np
import math
import time
import sys
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO('yolov8n.pt')

class ImgConv:
    def __init__(self,
                 param_file_path=None
                 ):

        self.Hd = None
        self.Wd = None
        self.map_x = None
        self.map_y = None
        self.singleLens = False
        self.filePath = param_file_path

    
    def equirect2cubemap(self,
                         srcFrame,
                         side=256,
                         modif=False,
                         dice=False
                         ):

        self.dice = dice
        self.side = side

        inShape = srcFrame.shape[:2]
        mesh = np.stack(
            np.meshgrid(
                np.linspace(-0.5, 0.5, num=side, dtype=np.float32),
                -np.linspace(-0.5, 0.5, num=side, dtype=np.float32),
            ),
            -1,
        )

        # Creating a matrix that contains x,y,z values of all 6 faces
        facesXYZ = np.zeros((side, side * 6, 3), np.float32)

        if modif:
            # Front face (z = 0.5)
            facesXYZ[:, 0 * side: 1 * side, [0, 2]] = mesh
            facesXYZ[:, 0 * side: 1 * side, 1] = -0.5

            # Right face (x = 0.5)
            facesXYZ[:, 1 * side: 2 * side, [1, 2]] = np.flip(mesh, axis=1)
            facesXYZ[:, 1 * side: 2 * side, 0] = 0.5

            # Back face (z = -0.5)
            facesXYZ[:, 2 * side: 3 * side, [0, 2]] = mesh
            facesXYZ[:, 2 * side: 3 * side, 1] = 0.5

            # Left face (x = -0.5)
            facesXYZ[:, 3 * side: 4 * side, [1, 2]] = np.flip(mesh, axis=1)
            facesXYZ[:, 3 * side: 4 * side, 0] = -0.5

            # Up face (y = 0.5)
            facesXYZ[:, 4 * side: 5 * side, [0, 1]] = mesh[::-1]
            facesXYZ[:, 4 * side: 5 * side, 2] = 0.5

            # Down face (y = -0.5)
            facesXYZ[:, 5 * side: 6 * side, [0, 1]] = mesh
            facesXYZ[:, 5 * side: 6 * side, 2] = -0.5

        else:
            # Front face (z = 0.5)
            facesXYZ[:, 0 * side: 1 * side, [0, 1]] = mesh
            facesXYZ[:, 0 * side: 1 * side, 2] = 0.5

            # Right face (x = 0.5)
            facesXYZ[:, 1 * side: 2 * side, [2, 1]] = mesh
            facesXYZ[:, 1 * side: 2 * side, 0] = 0.5

            # Back face (z = -0.5)
            facesXYZ[:, 2 * side: 3 * side, [0, 1]] = mesh
            facesXYZ[:, 2 * side: 3 * side, 2] = -0.5

            # Left face (x = -0.5)
            facesXYZ[:, 3 * side: 4 * side, [2, 1]] = mesh
            facesXYZ[:, 3 * side: 4 * side, 0] = -0.5

            # Up face (y = 0.5)
            facesXYZ[:, 4 * side: 5 * side, [0, 2]] = mesh
            facesXYZ[:, 4 * side: 5 * side, 1] = 0.5

            # Down face (y = -0.5)
            facesXYZ[:, 5 * side: 6 * side, [0, 2]] = mesh
            facesXYZ[:, 5 * side: 6 * side, 1] = -0.5

        # Calculating the spherical coordinates phi and theta for given XYZ
        # coordinate of a cube face
        x, y, z = np.split(facesXYZ, 3, axis=-1)
        # phi = tan^-1(x/z)
        phi = np.arctan2(x, z)
        # theta = tan^-1(y/||(x,y)||)
        theta = np.arctan2(y, np.sqrt(x ** 2 + z ** 2))

        h, w = inShape
        # Calculating corresponding coordinate points in
        # the equirectangular image
        eqrec_x = (phi / (2 * np.pi) + 0.5) * w
        eqrec_y = (-theta / np.pi + 0.5) * h
        # Note: we have considered equirectangular image to
        # be mapped to a normalised form and then to the scale of (pi,2pi)

        self.map_x = eqrec_x
        self.map_y = eqrec_y

        dstFrame = cv2.remap(srcFrame,
                             self.map_x,
                             self.map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)

        if self.dice:
            line1 = np.hstack(
                (
                    dstFrame[:, 4 * side: 5 * side, :] * 0,
                    cv2.flip(dstFrame[:, 4 * side: 5 * side, :], 0),
                    dstFrame[:, 4 * side: 5 * side, :] * 0,
                    dstFrame[:, 4 * side: 5 * side, :] * 0,
                )
            )
            line2 = np.hstack(
                (
                    dstFrame[:, 3 * side: 4 * side, :],
                    dstFrame[:, 0 * side: 1 * side, :],
                    cv2.flip(dstFrame[:, 1 * side: 2 * side, :], 1),
                    cv2.flip(dstFrame[:, 2 * side: 3 * side, :], 1),
                )
            )
            line3 = np.hstack(
                (
                    dstFrame[:, 5 * side: 6 * side, :] * 0,
                    dstFrame[:, 5 * side: 6 * side, :],
                    dstFrame[:, 5 * side: 6 * side, :] * 0,
                    dstFrame[:, 5 * side: 6 * side, :] * 0,
                )
            )
            dstFrame = np.vstack((line1, line2, line3))
            
        return dstFrame

# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture('equirect.mp4')

# Background Subtraction
fgbg = cv2.createBackgroundSubtractorMOG2()

while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Creating a mapper object
    mapper = ImgConv()

    # Generating cubemap from equirectangular image
    cubemap = mapper.equirect2cubemap(frame,side=256,dice=1)

    # results = model.predict(cubemap)
    # for r in results:
    #     annotator = Annotator(cubemap)

    #     boxes = r.boxes
    #     for box in boxes:
    #         b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
    #         c = box.cls
    #         class_name = model.names[int(c)]
    #         if class_name=='bicycle':
    #             x_min, y_min, x_max, y_max = b

    #             print(f"Class: {class_name}, Coordinates: (x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max})")

    #             annotator.box_label(b, model.names[int(c)])


    
    # img = annotator.result()
    # cv2.imshow('YOLO V8 Detection', img)
    cv2.imshow("cubemap",cubemap)

    # # Edge Detection
    # edge_detect = cv2.Canny(cubemap, 100, 200)
    # cv2.imshow('Edge detect', edge_detect)

    # # Harris Conner Detector
    # operatedImage = cv2.cvtColor(cubemap, cv2.COLOR_BGR2GRAY) 
    # operatedImage = np.float32(operatedImage)
    # dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07) 
    # dest = cv2.dilate(dest, None) 
    # cubemap[dest > 0.01 * dest.max()]=[0, 0, 255] 
    # cv2.imshow('Image with Borders', cubemap) 

    cv2.waitKey(1)

# release the video capture object
cap.release()

# Closes all the windows currently opened.
cv2.destroyAllWindows()