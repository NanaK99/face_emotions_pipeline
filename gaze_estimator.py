import mediapipe as mp
import cv2
import gaze
import argparse


mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,  # number of faces to track in each frame
        refine_landmarks=True,  # includes iris landmarks in the face mesh model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)


def get_gaze_direction(image):
    text = ""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV
    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        gazee = gaze.gaze(image, results.multi_face_landmarks[0])  # gaze estimation
        if gaze is not None:
            return gazee
        else:
            return None
    else:
        return None


if __name__ == '__main__':

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
            gazee = get_gaze_direction(frame)

            if gazee is not None:
                p1_left = gazee[0]
                p1_right = gazee[1]
                p2 = gazee[2]
                gaze_text = gazee[3]

                if args.anot == 'True':
                    cv2.line(frame, p1_left, p2, (0, 0, 255), 2)
                    cv2.line(frame, p1_right, p2, (0, 0, 255), 2)

                    cv2.putText(frame, gaze_text, (300, 30),
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
