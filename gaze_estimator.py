import mediapipe as mp
import cv2
import gaze


mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model


def get_gaze_direction(image):
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,  # number of faces to track in each frame
            refine_landmarks=True,  # includes iris landmarks in the face mesh model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV

        if results.multi_face_landmarks:
            text = gaze.gaze(image, results.multi_face_landmarks[0])  # gaze estimation

        return text