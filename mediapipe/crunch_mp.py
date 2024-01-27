import cv2
import mediapipe as mp
import numpy as np
import PoseModule as pm

detector = pm.PoseDetector()



def process_video(video_path):
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)

    while True:
        frame_num += 1

        grabbed, frame = cap.read()

        # if the frame is not captured
        if not grabbed:
            break
        frame = detector.findPose(frame, False)
        lmList = detector.findPosition(frame, False)

        if len(lmList) != 0:
            print("Kljucne tacke da se izdvoje kao kod skleka")