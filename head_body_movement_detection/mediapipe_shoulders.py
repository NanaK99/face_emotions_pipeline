from scipy.spatial import distance as dist
import mediapipe as mp
import math as m
import argparse
import cv2
import logging
import scipy


font = cv2.FONT_HERSHEY_SIMPLEX

blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

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


def f_test(arr1, arr2):
    f, p = scipy.stats.f_oneway(arr1, arr2)
    return f, p


def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist


def sendWarning(x):
    pass


def get_aspect_ratio(top, bottom, right, left):
  height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
  width = dist.euclidean([right.x, right.y], [left.x, left.y])
  return height / width


def get_shoulder_visbility(right_shoulder, left_shoulder):
    if right_shoulder.visibility >= 0.8 and left_shoulder.visibility >= 0.8:
        return "Visible"
    elif (right_shoulder.visibility >= 0.8 and left_shoulder.visibility < 0.8) or (left_shoulder.visibility >= 0.8 and right_shoulder.visibility < 0.8):
        return "Partially visible"
    else:
        "Not visible"


def midpoint(x1, x2, y1, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2


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

        ML_x = mouth_left.x
        ML_y = mouth_left.y
        MR_x = mouth_right.x
        MR_y = mouth_right.y

        l_shldr_x = left_shoulder.x
        l_shldr_y = left_shoulder.y
        r_shldr_x = right_shoulder.x
        r_shldr_y = right_shoulder.y

        mid_mouth_x, mid_mouth_y = midpoint(ML_x, MR_x, ML_y, MR_y)
        mid_should_x, mid_should_y = midpoint(l_shldr_x, r_shldr_x, l_shldr_y, r_shldr_y)

        middle_mouth_shoulder_dist = findDistance(mid_mouth_x, mid_mouth_y, mid_should_x, mid_should_y)

        return middle_mouth_shoulder_dist


if __name__ == "__main__":
    text = ""
    mids_stdev = 0
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
    mids = []
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('shoulder_nod_res.mp4', fourcc, 20.0, (frame_width, frame_height))

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
            # print(right_stds)
            # print(left_stds)
            break

            # break
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

        # print("LEFT",left_shoulder)
        # print("RIGHT",right_shoulder)

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

        # left_sh = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
        # right_sh = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
        # SHOULDER LINE COEFFICIENTS
        # a = left_sh[1] - right_sh[1]
        # b = left_sh[0] - right_sh[0]
        # c = right_sh[0] * left_sh[1] - left_sh[0] * right_sh[1]
        # LEFT DISTANCE
        # m_l = np.array([mouth_left.x, mouth_left.y, mouth_left.z])
        # left_distance = np.abs((a * m_l[0] + b * m_l[1] + c) / np.sqrt(a**2 + b**2))
        # left_dists.append(left_distance)

        # RIGHT DISTANCE
        # m_r = np.array([mouth_right.x, mouth_right.y, mouth_right.z])
        # right_distance = np.abs((a * m_r[0] + b * m_r[1] + c) / np.sqrt(a**2 + b**2))
        # right_dists.append(right_distance)

        ML_x = mouth_left.x
        ML_y = mouth_left.y

        MR_x = mouth_right.x
        MR_y = mouth_right.y

        l_shldr_x = left_shoulder.x
        l_shldr_y = left_shoulder.y
        # Right shoulder
        r_shldr_x = right_shoulder.x
        r_shldr_y = right_shoulder.y

        mid_mouth_x, mid_mouth_y = midpoint(ML_x, MR_x, ML_y, MR_y)
        # print("!!!", ML_x, ML_y, MR_x, MR_y, mid_mouth_y, mid_mouth_x)
        mid_should_x, mid_should_y = midpoint(l_shldr_x, r_shldr_x, l_shldr_y, r_shldr_y)
        # print("@@@",mid_mouth_x, mid_mouth_y, mid_should_x, mid_should_y)

        middle_mouth_shoulder_dist = findDistance(mid_mouth_x, mid_mouth_y, mid_should_x, mid_should_y)
        # print("###",middle_mouth_shoulder_dist)
        mids.append(middle_mouth_shoulder_dist)

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
            # stdev_right = statistics.stdev(right_dists)
            # right_stds.append(stdev_right)
            #
            # stdev_left = statistics.stdev(left_dists)
            # left_stds.append(stdev_left)

            # print()
            # cv2.putText(frame, f"right sh {r_shldr_x:.2f}, {r_shldr_y:.2f}", font, 0.9, blue, 2)

            mids_stdev = statistics.stdev(mids)
            # print(mids)
            mids_stdev = mids_stdev*10000
            print(mids_stdev)
            # print(mids_stdev)
            if mids_stdev > 50:
                print(f"{mids_stdev}: NOD")
                text = "NOD"
                # cv2.putText(frame, f"{mids_stdev :3f}; NOD", (150, 60), font, 1.5, blue, 2)
            elif (mids_stdev < 50 and mids_stdev > 35):
                text = "SHOULDERS"
                print(f"{mids_stdev}: SHOULDERS")
                # cv2.putText(frame, f"{mids_stdev :3f}; SHOULDERS", (150, 60), font, 1.5, blue, 2)
            # Get the resolution of the frames in the video stream
            # frame_width = int(cap.get(3))
            # frame_height = int(cap.get(4))
            #
            # # Define the codec and create a VideoWriter object
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # out = cv2.VideoWriter('shoulder_nod_res.mp4', fourcc, 20.0, (frame_width, frame_height))
            # out.write(frame)

            # cv2.putText(frame, f"mid stdevs {mids_stdev}", (150, 30), font, 0.9, blue, 2)
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
            # right_dists = []
            # left_dists = []
            # left_stds = []
            # right_stds = []
            mids = []



        # draws lines in 3d graph, do not need now
        # mp_drawing.draw_landmarks(
        #     frame,
        #     results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.putText(frame, f"{mids_stdev :3f}; {text}", (150, 60), font, 1.5, blue, 2)
        mids_stdev = 0
        text = ""

        cv2.imshow('Frame', frame)
        out.write(frame)

        frame_counter += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

