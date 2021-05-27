from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame
import time
import dlib
import cv2

pygame.mixer.init()
pygame.mixer.music.load('alert.wav')

EYE_ASPECT_RATIO_THRESHOLD = 0.3
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50
COUNTER = 0

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2 * C)
    return ear


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

video_capture = cv2.VideoCapture(0)

time.sleep(2)

while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)

    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD:
            COUNTER += 1
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                pygame.mixer.music.play(-1)
                cv2.putText(frame, "Drowsy", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        else:
            pygame.mixer.music.stop()
            COUNTER = 0

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
