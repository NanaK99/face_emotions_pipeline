import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

FACE_TILT = .5

EYE_OPEN_HEIGHT = .35

BROWS_RAISE = .16
BROW_RAISE_LEFT = .0028
BROW_RAISE_RIGHT = .025
FURROWED_BROWS = 0.05


def get_aspect_ratio(top, bottom, right, left):
    height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
    width = dist.euclidean([right.x, right.y], [left.x, left.y])
    return height / width


def get_face_movement(image):
        results = face_mesh.process(image)
        image.flags.writeable = True

        text = ""

        if results.multi_face_landmarks is None:
            return None

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            face_landmarks = results.multi_face_landmarks[0]
            face = face_landmarks.landmark

            face_mid_right = face[234]
            face_mid_left = face[454]
            cheek_mid_right = face[50]
            cheek_mid_left = face[280]

            # HEAD SHAKE
            if len(text) == 0:
                try:

                    if cheek_mid_right.x < face_mid_right.x:
                        text = "right"
                    elif cheek_mid_left.x > face_mid_left.x:
                        text = "left"

                except:
                    pass

            eyeR_inner = face[133]
            eyeL_inner = face[362]

            # HEAD NOD
            if len(text) == 0:
                try:
                    nodR_inner_ratio = 1 / eyeR_inner.z
                    nodL_inner_ratio = 1 / eyeL_inner.z

                    if nodL_inner_ratio < -0.6 and nodR_inner_ratio < -0.6:
                        text = "up"
                    elif nodL_inner_ratio > -0.5 and nodR_inner_ratio > -0.5:
                        text = "down"

                except:
                    pass
            else:
                return text

            if len(text) == 0:
                text = ""
                return text


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--start', metavar='N', type=int,
                        help='an integer for the accumulator')
    parser.add_argument('--video', metavar='N', type=str,
                        help='an integer for the accumulator')
    parser.add_argument('--anot', choices=('True', 'False'))

    args = parser.parse_args()
    start = args.start
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video file")
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            face_mov = get_face_movement(frame)

            if face_mov is not None:
                # mov = face_mov

                if args.anot == 'True':
                    cv2.putText(frame, face_mov, (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Frame', frame)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            print("error")
            break
    cap.release()