## Fitness-Movement-Detection
### Introduction

In this project we used Yolov4 model and Mediapipe to be able to detect when the user executes an exercise correctly. Yolov4 was used for exercise type detection(crunch, push up, pull up, squat) which helped us to extract important nodes (using Mediapipe) for each exercise, which form an angle that changes during the exercise and then analyze how the angle changed in time to be able to see if the exercise was done correctly. The change of the angle in time forms a signal that represents the movement, which we used to train SVM and TensorFlow classifiers to be able to separate right and wrong movement signals. Signals of right movement were labeled with 0 which represent Completeness and signals of wrong movements were labeled with 1 which represents No-completeness. These classifiers were later used to classify each 100 frames of a video and detect if certain 100 frames contain the right movement signal.

## Dataset
We used 64 videos of different people doing each exercise from different angles and in different lighting. 

**Yolov4 dataset**- we iterated through the videos and saved an image every 10 frames. Later we used those images to extract labels in Yolov4 format by selecting RoI in each image using labelImg tool.

https://github.com/HumanSignal/labelImg

**Classifier dataset**- we iterated through the videos and calculated completeness for each frame. Each 50 frames we added the completeness signal formed in the previous 100 frames to the training data set along with the label 0 or 1 depending on the signal (if the signal went from 0 to 1 and then to 0 again it means that the exercise is complete).

## Training
Yolov4 was trained using Google Collaboratory and then saved to the file “yolov4-custom_last .weights“. Parameters used for training can be found in the file “yolov4-custom.cfg”.

**SVM** was trained on 80% of classifier data and the other 20% was used for testing. We used rbf kernel and parameter C set to 1.0 which gave the best results. 

**TensorFlow** was trained on the same data as SVM. We used Artifical Neural Network for binary classification with an input dimension of 100 features and  Stochastic Gradient Descent (SGD) as the optimizer, and trained the ANN on the provided data. The training parameters include 0.01 learning rate, 0.7 momentum, and 1 as the batch size, with 1000 epochs.

## Testing and validation

Both SVM and TensorFlow classifier were tested on same data (20% of the classifier dataset) and gave results which we then compared with expected results and calculated Accuracy, Precision, Recall and F1 score. SVM had a higher precision score, TensorFlow had a higher Recall score and Accuracy and F1 score were similar for both classifiers. Scores for both classifiers can be found on the poster.


## Requirements
Before running the project, to be able to run the program run these commands:
1. **Python 3**

   You can download and install Python 3 from the [official Python website](https://www.python.org/).

2. **TensorFlow and Keras**

   ```bash
   pip install tensorflow keras
3. **NumPy**
   
   ```bash
   pip install numpy
4. **Matplotlib**
   
   ```bash
   pip install matplotlib
5. **Scikit-learn**
   
   ```bash
   pip install scikit-learn
6. **Pickle**
   ```bash
   pip install pickle

## Run
1. **To clone the repository execute:**
   ```bash
   git clone https://github.com/KristinaPopovSV5-2020/Fitness-Movement-Detection.git

2. **To run the project execute:**
    ```bash
    python main.py
## Presentation
[Fitness Movement Detection](https://github.com/KristinaPopovSV5-2020/Fitness-Movement-Detection/blob/main/FitnessMovementDetection.pptx)
## Poster
![Fitness Movement Detection](https://github.com/KristinaPopovSV5-2020/Fitness-Movement-Detection/blob/dataset/poster.png)
![Fitness Movement Detection](https://github.com/KristinaPopovSV5-2020/Fitness-Movement-Detection/blob/dataset/yolodemo/yolo_classes.jpg)

