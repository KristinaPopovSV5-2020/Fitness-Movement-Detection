import cv2
import numpy as np
import MP.PoseModule as pm
from MP.calculate_noi import noi_pushup
import mediapipe as mp

net = cv2.dnn.readNet("yolov4-custom_last.weights", "yolodemo/yolov4-custom.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

detector = pm.PoseDetector()

cap = cv2.VideoCapture("C:\\Users\\kikap\\Downloads\\push-Up.mp4")

boxes = []
confidences = []
class_ids = []

num=0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    num+=1

    if num==10:
        num=0
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-maximum suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            roi = frame[y:y + h, x:x + w]

            # Mediapipe Pose estimation on the ROI
            #image_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img = detector.findPose(frame, False)
            lmList = detector.findPosition(img, False)
            # print(lmList)
            if len(lmList) != 0:
                noi_pushup(detector,img)

                # Check to ensure right form before starting the program


    # Display the resulting frame
    cv2.imshow('Exercise Assessment', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
