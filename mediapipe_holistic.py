import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) \
        as holistic:

    while cap.isOpened():

        success, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        # Process the image and detect the holistic
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            nose = landmarks[mp_holistic.PoseLandmark.NOSE.value]
            nose_y_ratio = 1 / nose.y
            nose_z_ratio = 1 / nose.z
            if abs(nose_y_ratio) < 1.5 or abs(nose_z_ratio) < 0.35:
                text = "LEAN IN"
            elif abs(nose_y_ratio) > 2 or abs(nose_z_ratio) > 1:
                text = "LEAN OUT"
            else:
                text = "normal posture"
        except:
            pass

        # mp_drawing.draw_landmarks(
        #     image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # FACEMESH_CONTOURS FACEMESH_TESSELATION
        # mp_drawing.draw_landmarks(
        #     image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(
        #     image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.putText(image, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

        cv2.imshow('MediaPipe Holistic', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()