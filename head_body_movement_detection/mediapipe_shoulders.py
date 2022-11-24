from scipy.spatial import distance as dist
import mediapipe as mp
import math as m
import argparse
import cv2


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 -y1)*(-y1) / (m.sqrt(
        (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    degree = int(180/m.pi)*theta
    return degree


def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist


def sendWarning(x):
    pass


def get_aspect_ratio(top, bottom, right, left):
  height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
  width = dist.euclidean([right.x, right.y], [left.x, left.y])
  return height / width


def get_body_movement(image):
        # Process the image and detect the holistic
        results = holistic.process(image)
        # results = pose.process(image)
        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks is None:
            return None

        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        shoulder_angle = findAngle(left_shoulder.x, left_shoulder.y, right_shoulder.x, right_shoulder.y)

        return (shoulder_angle, (left_shoulder.y + right_shoulder.y)/2, (left_shoulder.z + right_shoulder.z)/2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--start', metavar='N', type=int,
    #                     help='an integer for the accumulator')
    parser.add_argument('--video', metavar='N')
    # parser.add_argument('--anot', choices=('True', 'False'))

    args = parser.parse_args()
    # start = args.start
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(args.video)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    if cap.isOpened() == False:
        print("Error opening video file")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            body_mov = get_body_movement(frame)
            # one_up_down.append(body_mov[0])
            # both_up_down.append(body_mov[1])
            # lean_in_out.append(body_mov[2])
            # print("BODY MOV", body_mov)

            if body_mov is not None:
                # if args.anot == 'True':
                print("####",body_mov[2])
                    # cv2.putText(frame, f"{body_mov[0], body_mov[1]}", (200, 30),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


            cv2.imshow('Frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            print("error")
            break
    cap.release()