import MP.PoseModule as pm


def completion_noi(angle_noi, start_angle_noi, end_angle_noi):
    completion = (angle_noi - start_angle_noi) / (end_angle_noi - start_angle_noi)
    return round(completion, 2)


def noi(type, detector, frame):
    if type == 0:
        angles = noi_crunch(detector, frame)
        return completion_noi(angles[0] or angles[1], 120, 40)
    elif type == 1:
        angles = noi_pushup(detector, frame)
        return completion_noi(angles[0] or angles[1], 80, 170)
    elif type == 2:
        angles = noi_squat(detector, frame)
        return completion_noi(angles[0] or angles[1], 80, 170)
    elif type == 3:
        angles = noi_pullup(detector, frame)
        return completion_noi(angles[0] or angles[1], 30, 165)


def noi_pushup(detector, frame):
    elbow_left = detector.find_angle(frame, 11, 13, 15)
    elbow_right = detector.find_angle(frame, 12, 14, 16)
    return elbow_left, elbow_right


def noi_pullup(detector, frame):
    elbow_left = detector.find_angle(frame, 11, 13, 15)
    elbow_right = detector.find_angle(frame, 12, 14, 16)
    return elbow_left, elbow_right


def noi_squat(detector, frame):
    hip_left = detector.find_angle(frame, 11, 23, 25)
    hip_right = detector.find_angle(frame, 12, 24, 26)
    return hip_left, hip_right


def noi_crunch(detector, frame):
    hip_left = detector.find_angle(frame, 11, 23, 25)
    hip_right = detector.find_angle(frame, 12, 24, 26)
    return hip_left, hip_right
