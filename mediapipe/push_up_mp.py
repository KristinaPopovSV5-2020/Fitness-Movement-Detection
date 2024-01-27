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
            elbow_left = detector.find_angle(frame, 11, 13, 15)
            elbow_right = detector.find_angle(frame, 12, 14, 16)
            shoulder_left = detector.find_angle(frame, 13, 11, 23)
            shoulder_right = detector.find_angle(frame, 14, 12, 24)
            hip_left = detector.find_angle(frame, 11, 23, 25)
            hip_right = detector.find_angle(frame, 12, 24, 26)


