import cv2
import numpy as np
import MP.PoseModule as pm
from MP.calculate_noi import noi
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import time
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


last_class = 1
"""def yolo_detect():
    global boxes,confidences,class_ids

    boxes = []
    confidences = []
    class_ids = []

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

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

    return cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


def svm_data():
    values=[0,1,0]
    i=0
    for c in completion:
        if c+0.05>values[i]>c-0.05:
            i+=1
            if i==2:
                break
    if i==2:
        y=0
    else:
        y=1
    #print(y)
    svm_x.append(completion)
    svm_y.append(y)

    #plt.plot(list(range(len(completion))), completion)
    #plt.show()
"""

if __name__ == '__main__':

    net = cv2.dnn.readNet("yolov4-custom_last.weights", "yolodemo/yolov4-custom.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    detector = pm.PoseDetector()

    folder_path="C:\\Users\\zoric\\Downloads\\pull Up\\"
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4') or f.endswith(".MOV")]

    svm_x = []
    svm_y = []
    with open('svm_x.txt', 'r') as file:
        for line in file:
            # Use eval to convert the string representation of a list to an actual list
            sublist = eval(line.strip())
            svm_x.append(sublist)
    with open('svm_y.txt', 'r') as file:
        for line in file:
            # Use eval to convert the string representation of a list to an actual list
            sublist = eval(line.strip())
            svm_y.append(sublist)

    with open('svm_classifier.pkl', 'rb') as file:
        classifier = pickle.load(file)

    """for video_file in video_files:
        print(video_file)
        cap = cv2.VideoCapture(os.path.join(folder_path, video_file))

        boxes = []
        confidences = []
        class_ids = []

        num=0
        indices = []
        num_frames=0
        num_exercise=0

        completion=[]
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))

            height, width, channels = frame.shape

            if num_frames%50==0:
                num += 1
                indices=yolo_detect()
                if len(indices)>0:
                    last_class = class_ids[indices[0]]

                if num>2:
                    svm_data()
                    if len(completion)==100:

                        prediction=svm_classifier.predict([completion])
                        if prediction==0:
                            num_exercise+=
                    completion = completion[50:]

            num_frames+=1

            frame = detector.findPose(frame, False)
            lmList = detector.findPosition(frame, False)
            if len(lmList) != 0:
                completion.append(noi(last_class,detector,frame))

            #cv2.imshow('Exercise Assessment', frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
        #print("Video "+video_file+": "+str(num_exercise))"""

    x=[]
    y=[]
    for i in range(len(svm_x)):
        if len(svm_x[i])==100:
            x.append(svm_x[i])
            y.append(svm_y[i])
    svm_x = np.array(x, 'float32')
    svm_y = np.array(y, 'int')
    x_train, x_test, y_train, y_test = train_test_split(svm_x, svm_y, test_size=0.2, random_state=42)
    #classifier = SVC(kernel='rbf', C=1.0, probability=True)
    #classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    #with open('svm_classifier1.pkl', 'wb') as file:
     #   pickle.dump(classifier, file)

    """file_path = "svm_x1.txt"
    with open(file_path, 'w') as file:
        for inner_list in svm_x:
            line = ','.join(map(str, inner_list)) + '\n'
            file.write(line)
    file_path = "svm_y1.txt"
    with open(file_path, 'w') as file:
        for inner_list in svm_y:
            line = ','.join(map(str, inner_list)) + '\n'
            file.write(line)"""
