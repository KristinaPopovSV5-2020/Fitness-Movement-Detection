import cv2
import numpy as np
from matplotlib import pyplot as plt

import MP.PoseModule as pm
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
from keras.models import Sequential
from keras.layers import Dense

from tensorflow.keras.optimizers import SGD
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from MP.calculate_noi import noi

last_class = None
detector = pm.PoseDetector()
classifier_SVM = False


def yolo_detect():
    global boxes, confidences, class_ids

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
                new_width = int(detection[2] * width)
                new_height = int(detection[3] * height)

                new_x = int(center_x - new_width / 2)
                new_y = int(center_y - new_height / 2)

                boxes.append([new_x, new_y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


def load_dataset():
    values = [0, 1, 0]
    i = 0
    for c in completion:
        if c + 0.05 > values[i] > c - 0.05:
            i += 1
            if i == 2:
                break
    if i == 2:
        y = 0
    else:
        y = 1
    # print(y)
    x_data.append(completion)
    y_data.append(y)

    plt.plot(list(range(len(completion))), completion)
    plt.show()


def svm(svm_x, svm_y):
    x_train, x_test, y_train, y_test = train_test_split(svm_x, svm_y, test_size=0.2, random_state=42)
    classifier = SVC(kernel='rbf', C=1.0, probability=True)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")


def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(1, input_dim=100, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann


def train_ann(ann, svm_x, svm_y, epochs):
    x_train, x_test, y_train, y_test = train_test_split(svm_x, svm_y, test_size=0.2, random_state=42)
    print("\nTraining started...")
    sgd = SGD(learning_rate=0.01, momentum=0.7)
    ann.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
    ann.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=True)
    print("\nTraining completed...")
    predictions = (ann.predict(x_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    with open('tensorFlow_classifier.pkl', 'wb') as file:
        pickle.dump(ann, file)


def load_data():
    svm_x = []
    svm_y = []
    with open('svm_x.txt', 'r') as file:
        for line in file:
            sublist = eval(line.strip())
            svm_x.append(sublist)
    with open('svm_y.txt', 'r') as file:
        for line in file:
            sublist = eval(line.strip())
            svm_y.append(sublist)

    x = []
    y = []
    for i in range(len(svm_x)):
        if len(svm_x[i]) == 100:
            x.append(svm_x[i])
            y.append(svm_y[i])
    svm_x = np.array(x, 'float32')
    svm_y = np.array(y, 'int')
    return svm_x, svm_y


def train_data():
    svm_x, svm_y = load_data()
    if classifier_SVM:
        svm(svm_x, svm_y)
    else:
        ann = create_ann(output_size=1)
        train_ann(ann, svm_x, svm_y, epochs=1000)


if __name__ == '__main__':
    net = cv2.dnn.readNet("yolov4-custom_last.weights", "yolodemo/yolov4-custom.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    folder_path = "C:\\Users\\zoric\\Downloads\\pull Up\\"
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

    if classifier_SVM:
        with open('svm_classifier.pkl', 'rb') as file:
            classifier = pickle.load(file)
    else:
        with open('tensorFlow_classifier.pkl', 'rb') as file:
            classifier = pickle.load(file)

    #train_data()
    x_data = []
    y_data = []

    for video_file in video_files:
        cap = cv2.VideoCapture(os.path.join(folder_path, video_file))

        boxes = []
        confidences = []
        class_ids = []

        num = 0
        indices = []
        num_frames = 0
        num_exercise = 0

        completion = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))

            height, width, channels = frame.shape

            if num_frames % 50 == 0:
                num += 1
                indices = yolo_detect()
                #x, y, w, h = boxes[indices[0]]
                #roi = frame[y:y + h, x:x + w]
                if len(indices) > 0:
                    last_class = class_ids[indices[0]]

                if num > 2:
                    #load_dataset()
                    if len(completion) == 100:
                        prediction = classifier.predict([completion])
                        if classifier_SVM:
                            if prediction==0:
                                num_exercise += 1
                        else:
                            if prediction > 0.65:
                                num_exercise += 1
                    completion = completion[50:]

            num_frames += 1

            frame = detector.findPose(frame, False)
            lmList = detector.findPosition(frame, False)
            if len(lmList) != 0:
                completion.append(noi(last_class, detector, frame))

            cv2.imshow('Exercise Assessment', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Video " + video_file + ": " + str(num_exercise))
