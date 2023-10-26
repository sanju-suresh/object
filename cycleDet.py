from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator

model = YOLO('yolov8n.pt')

# Load your image
image_path = 'cycle.png'
img = cv2.imread(image_path)

# BGR to RGB conversion is performed under the hood

#COMMENT
# see: https://github.com/ultralytics/ultralytics/issues/2575
results = model.predict(img)

for r in results:
    annotator = Annotator(img)

    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
        c = box.cls
        class_name = model.names[int(c)]
        if class_name=='bicycle':
            x_min, y_min, x_max, y_max = b

            print(f"Class: {class_name}, Coordinates: (x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max})")

            annotator.box_label(b, model.names[int(c)])

img = annotator.result()
cv2.imshow('YOLO V8 Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()





