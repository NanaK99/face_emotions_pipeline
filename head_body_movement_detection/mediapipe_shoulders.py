from scipy.spatial import distance as dist
import mediapipe as mp
import math as m
import argparse
import cv2
import logging


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


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
        mouth_left = landmarks[mp_holistic.PoseLandmark.MOUTH_LEFT.value]
        mouth_right = landmarks[mp_holistic.PoseLandmark.MOUTH_RIGHT.value]

        left_sh = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
        right_sh = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])

        # SHOULDER LINE COEFFICIENTS
        a = left_sh[1] - right_sh[1]
        b = left_sh[0] - right_sh[0]
        c = right_sh[0] * left_sh[1] - left_sh[0] * right_sh[1]

        # LEFT DISTANCE
        m_l = np.array([mouth_left.x, mouth_left.y, mouth_left.z])
        left_distance = np.abs((a * m_l[0] + b * m_l[1] + c) / np.sqrt(a**2 + b**2))
        left_dists.append(left_distance)

        # RIGHT DISTANCE
        m_r = np.array([mouth_right.x, mouth_right.y, mouth_right.z])
        right_distance = np.abs((a * m_r[0] + b * m_r[1] + c) / np.sqrt(a**2 + b**2))
        right_dists.append(right_distance)

        return right_dists, left_dists
        # logging.info(f"shoulders, {shoulder_angle}, {(left_shoulder.y + right_shoulder.y)/2}, {(left_shoulder.z + right_shoulder.z)/2}")
        # return (shoulder_angle, (left_shoulder.y + right_shoulder.y)/2, (left_shoulder.z + right_shoulder.z)/2)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--start', metavar='N', type=int,
    #                     help='an integer for the accumulator')
    parser.add_argument('--video', metavar='N')
    # parser.add_argument('--anot', choices=('True', 'False'))

    args = parser.parse_args()
    # start = args.start
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    cap = cv2.VideoCapture(args.video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("FRAME COUNT ---->", frame_count)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print("FPS ---->", fps)

    frame_counter = 1
    ll = []
    ds = []
    sh_l = []
    sh_r = []
    mo_l = []
    mo_r = []
    stdevs_dist = []
    stdevs_shl = []
    stdevs_shr = []
    stdevs_mor = []
    stdevs_mol = []

    right_dists = []
    left_dists = []
    right_stds = []
    left_stds = []

    if cap.isOpened() == False:
        print("Error opening video file")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            print("video ended")
            # print(ll)
            # print(ds)
            # print(stdevs_dist) # for head nod [0.09089610246745139, 0.1374637160754863, 0.12203126734415094, 0.11126796700972709, 0.10053258742756763]
            # print(stdevs_shl)
            # print(stdevs_shr)
            # print(stdevs_mol)
            # print(stdevs_mor)
            print(right_stds)
            print(left_stds)

            break
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # print("Current frame number:", frame_index)
        # print("FRAME COUNTER -------------->", frame_counter)
        results = pose.process(frame)
        if results.pose_landmarks is None:
            continue
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        mouth_left = landmarks[mp_holistic.PoseLandmark.MOUTH_LEFT.value]
        mouth_right = landmarks[mp_holistic.PoseLandmark.MOUTH_RIGHT.value]

        # one_up_down.append(body_mov[0])
        # both_up_down.append(body_mov[1])
        # lean_in_out.append(body_mov[2])
        # print("BODY MOV", body_mov)

        # if body_mov is not None:
        #     # if args.anot == 'True':
        #     print("####",body_mov[2])
        # cv2.putText(frame, f"{body_mov[0], body_mov[1]}", (200, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        import numpy as np

        left_sh = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
        right_sh = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
        # SHOULDER LINE COEFFICIENTS
        a = left_sh[1] - right_sh[1]
        b = left_sh[0] - right_sh[0]
        c = right_sh[0] * left_sh[1] - left_sh[0] * right_sh[1]
        # LEFT DISTANCE
        m_l = np.array([mouth_left.x, mouth_left.y, mouth_left.z])
        left_distance = np.abs((a * m_l[0] + b * m_l[1] + c) / np.sqrt(a**2 + b**2))
        left_dists.append(left_distance)

        # RIGHT DISTANCE
        m_r = np.array([mouth_right.x, mouth_right.y, mouth_right.z])
        right_distance = np.abs((a * m_r[0] + b * m_r[1] + c) / np.sqrt(a**2 + b**2))
        right_dists.append(right_distance)

        # P1 = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.y])
        # P2 = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
        # P3 = np.array([mouth_left.x, mouth_left.y, mouth_left.z])
        # P4 = np.array([mouth_right.x, mouth_right.y, mouth_right.z])
        #
        # u = P2 - P1
        # v = P4 - P3
        #
        # d = np.linalg.norm(np.cross(u, v)) / np.linalg.norm(v)
        # ds.append(d)
        # sh_l.append(left_shoulder.y)
        # sh_r.append(right_shoulder.y)
        # mo_l.append(mouth_left.y)
        # mo_r.append(mouth_right.y)

        # ll.append([left_shoulder.y, right_shoulder.y, mouth_left.y, mouth_right.y])

        if frame_counter % 5 == 0:
            import statistics

            # stdev_dist = statistics.stdev(ds)
            # stdevs_dist.append(stdev_dist)
            #
            # stdev_shl = statistics.stdev(sh_l)
            # stdevs_shl.append(stdev_shl)
            #
            # stdev_shr = statistics.stdev(sh_r)
            # stdevs_shr.append(stdev_shr)
            #
            # stdev_mol = statistics.stdev(mo_l)
            # stdevs_mol.append(stdev_mol)
            #
            # stdev_mor = statistics.stdev(sh_r)
            # stdevs_mor.append(stdev_mor)

            # RIGHT AND LEFT DISTANCES
            stdev_right = statistics.stdev(right_dists)
            right_stds.append(stdev_right)

            stdev_left = statistics.stdev(left_dists)
            left_stds.append(stdev_left)

            # ds = []
            # sh_l = []
            # sh_r = []
            # mo_l = []
            # mo_r = []
            # stdevs_dist = []
            # stdevs_shl = []
            # stdevs_shr = []
            # stdevs_mor = []
            # stdevs_mol = []
            right_dists = []
            left_dists = []
            # left_stds = []
            # right_stds = []


        # draws lines in 3d graph, do not need now
        # mp_drawing.draw_landmarks(
        #     frame,
        #     results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow('Frame', frame)

        frame_counter += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
