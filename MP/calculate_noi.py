import MP.PoseModule as pm


def competition_noi(angle_noi,start_angle_noi, end_angle_noi):
    competition = (angle_noi - start_angle_noi)/(end_angle_noi-start_angle_noi)
    return competition

def noi_pushup(detector,frame):
    shoulder_left = detector.find_angle(frame, 13, 11, 23)
    shoulder_right = detector.find_angle(frame, 14, 12, 24)

    elbow_left = detector.find_angle(frame, 11, 13, 15)
    elbow_right = detector.find_angle(frame, 12, 14, 16)

    hip_left = detector.find_angle(frame, 11, 23, 25)
    hip_right = detector.find_angle(frame, 12, 24, 26)


def noi_squat(detector,frame):
    hip_left = detector.find_angle(frame, 11, 23, 25)
    hip_right = detector.find_angle(frame, 12, 24, 26)

    knee_left = detector.find_angle(frame, 23, 25, 27)
    knee_right = detector.find_angle(frame, 24, 26, 28)


def noi_crunch(detector,frame):
    shoulder_left = detector.find_angle(frame, 13, 11, 23)
    shoulder_right = detector.find_angle(frame, 14, 12, 24)

    hip_left = detector.find_angle(frame, 11, 23, 25)
    hip_right = detector.find_angle(frame, 12, 24, 26)


