# Driver_Drowsiness_Detection_Using_Dlib_And_Computer-Vision


1.scipy (distance can compute the Euclidean distance between facial landmarks points in the eye aspect ratio calculation)

2.imutils (face_utils extract indexes of facial landmarks for the left and right eye)

3.numpy()

4.pygame(will be used to paly alert)

5.time(will be used for camera to initialize)

6.dlib(detect face and face landmarks in a frame)

7.cv2()

8.shape_predictor_68_face_landmarks.dat(dlib’s pre-trained facial landmark
detector)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#After starting webcam
# Detect facial points through detector function
   faces = detector(gray, 0)

   shape = predictor(gray, face)
   shape = face_utils.shape_to_np(shape)
# Get array of coordinates of leftEye and rightEye
   leftEye = shape[lStart:rEnd]
   rightEye = shape[rStart:rEnd]
