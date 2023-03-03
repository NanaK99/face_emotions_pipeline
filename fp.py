from head_body_movement_detection import mediapipe_face, mediapipe_shoulders
from face_expr_detection import face_visible_expressions
from gaze_detection import gaze_estimator
from emotion_detection import inference

from configparser import ConfigParser
from collections import Counter
import argparse
import statistics
from statistics import mean
import logging
import cv2
import sys
import os


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('Error: %s\n' % message)
        self.print_help()
        sys.exit(2)


config_object = ConfigParser()
config_object.read("./static/config.ini")
paths = config_object["PATHS"]


parser = MyParser()

parser.add_argument('--verbose',
                    help='a boolean indicating the mode for logging, '
                         'in case of True, prints will also be visible in the terminal; '
                         'otherwise the logs will be kept only in the log file',
                    required=False,
                    action='store_true')

parser.add_argument('--gaze',
                    help='gaze only',
                    required=False,
                    action='store_true')

parser.add_argument('--expressions',
                    help='expressions only',
                    required=False,
                    action='store_true')

parser.add_argument('--body',
                    help='body only',
                    required=False,
                    action='store_true')

parser.add_argument('--emotions',
                    help='emotions only',
                    required=False,
                    action='store_true')


args = parser.parse_args()

if args.verbose:
    verbose = True
else:
    verbose = False

if args.gaze:
    gaze = True
else:
    gaze = False

if args.body:
    body = True
else:
    body = False

if args.expressions:
    expressions = True
else:
    expressions = False

if args.emotions:
    emotions = True
else:
    emotions = False

# type_of_run = sys.argv[1].strip("--")
log_folder_path = paths["LOG_FOLDER_PATH"]

if not os.path.exists(log_folder_path):
    os.makedirs(log_folder_path, exist_ok=True)

# log_file_path = os.path.join(log_folder_path, type_of_run+".txt")
log_file_path = os.path.join(log_folder_path, "log_file.txt")
log_file_exists = os.path.exists(log_file_path)
if log_file_exists:
    os.remove(log_file_path)

logging.basicConfig(filename=log_file_path, level=logging.INFO,  format="%(asctime)s %(message)s", filemode="a")

VID_EXTENSIONS = ["mp4"]
TEXTGRID_EXTENSIONS = ["TextGrid"]

cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)  # 25
font = cv2.FONT_HERSHEY_SIMPLEX

model = inference.Model()

BODY_MOVEMENT = config_object["BODY_MOVEMENT"]

# verbose = ast.literal_eval(verbose)

video_parameters = config_object["VIDEO_PARAMETERS"]
min_num_of_frames = int(video_parameters["MIN_NUM_OF_FRAMES"])

sh_std_upper_bound = float(BODY_MOVEMENT["SHOULDER_STDEV_UPPER_BOUND"])
sh_std_lower_bound = float(BODY_MOVEMENT["SHOULDER_STDEV_LOWER_BOUND"])
NOD_MEAN = float(BODY_MOVEMENT["NOD_MEAN"])
HEAD_SHAKE_STDEV = float(BODY_MOVEMENT["SHAKE_STDEV"])

gaze_texts = []
head_nods = []
faceexpr_texts = []
emotion_texts = []
mids = []
head_shakes = []
frame_idx = 0

rights = []
lefts = []
degrees = []

gaze_label = ""
emotion_label = ""
body_label = ""
expression_label = ""
if not cap.isOpened():
    print("Error opening video file")

while cap.isOpened():
    success, img = cap.read()

    if not success:
        break

    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model

    # GAZE
    if gaze:
        gaze_all = gaze_estimator.get_gaze_direction(img, )
        if gaze_all is not None:
            p1_left = gaze_all[0]
            p1_right = gaze_all[1]
            p2 = gaze_all[2]
            # print(p2)
            g_text = gaze_all[3]
            cv2.line(img, p1_left, p2, (0, 0, 255), 2)
            cv2.line(img, p1_right, p2, (0, 0, 255), 2)
            # cv2.rectangle(img, p1, p4, (0, 255, 0), -1)

        else:
            g_text = ""
        if verbose:
            print(f"Frame {frame_idx} Gaze {g_text}")
        logging.info(f"Frame {frame_idx} Gaze {g_text}")

        gaze_texts.append(g_text)

    # BODY
    if body:
        landmarks = mediapipe_shoulders.get_landmarks(img)
        if landmarks is not None:
            pitch, roll, yaw = mediapipe_shoulders.get_shake_nod(landmarks)
            # HEAD SHAKE
            if yaw is not None:
                head_shakes.append(yaw)
                if verbose:
                    print(f"Frame {frame_idx}, yaw: {yaw}")
                logging.info(f"Frame {frame_idx}, yaw: {yaw}")
                # print("yaw", yaw)

            ##################################
            import mediapipe as mp
            mp_holistic = mp.solutions.holistic
            left_sh_x = int(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x * img.shape[1])
            left_sh_y = int(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y * img.shape[0])

            right_sh_x = int(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x * img.shape[1])
            right_sh_y = int(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y * img.shape[0])

            nose_x = int(landmarks[mp_holistic.PoseLandmark.NOSE.value].x * img.shape[1])
            nose_y = int(landmarks[mp_holistic.PoseLandmark.NOSE.value].y * img.shape[0])

            # print(right_sh_y, type(right_sh_y))

            # print(right_sh, type(right_sh))
            cv2.circle(img, (left_sh_x, left_sh_y), 5, (0, 0, 255), -1)
            cv2.circle(img, (right_sh_x, right_sh_y), 5, (0, 0, 255), -1)
            cv2.line(img, (left_sh_x, left_sh_y), (nose_x, nose_y), color=(255, 0, 0),
                     thickness=3)
            cv2.line(img, (right_sh_x, right_sh_y), (nose_x, nose_y), color=(255, 0, 0),
                     thickness=3)
            cv2.line(img, (right_sh_x, right_sh_y), (left_sh_x, left_sh_y), color=(255, 0, 0),
                     thickness=3)

            left_sh = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
            left_sh_vector = (left_sh.x, left_sh.y, left_sh.z)
            right_sh = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
            right_sh_vector = (right_sh.x, right_sh.y, right_sh.z)

            import numpy as np
            import math
            dot_product = np.dot(right_sh_vector, left_sh_vector)
            magnitude_left_sh = np.linalg.norm(left_sh_vector)
            magnitude_right_sh = np.linalg.norm(right_sh_vector)
            cos_theta = dot_product / (magnitude_left_sh * magnitude_right_sh)
            theta_radians = math.acos(cos_theta)
            theta_degrees = math.degrees(theta_radians)
            degrees.append(theta_degrees)
            # print(theta_degrees)

            #######################################################

            # HEAD NOD
            if pitch is not None:
                head_nods.append(pitch)
                if verbose:
                    print(f"Frame {frame_idx}, pitch: {pitch}")
                logging.info(f"Frame {frame_idx}, pitch: {pitch}")
                # print("pitch", pitch)
                # print("roll", roll)
            # SHOULDER MOVEMENT
            mouth_sh_dist = mediapipe_shoulders.get_shoulder_movement(landmarks)
            if mouth_sh_dist is not None:
                mids.append(mouth_sh_dist)
                if verbose:
                    print(f"Frame {frame_idx} body move {mouth_sh_dist}")
                logging.info(f"Frame {frame_idx} body move {mouth_sh_dist}")
                # print("mouth sh dist",mouth_sh_dist)
            # SHOULDER MOVEMENT_EXPERIMENT
            left, right = mediapipe_shoulders.get_shoulder_movement_exp(landmarks)
            if left is not None and right is not None:
                lefts.append(left)
                rights.append(right)

    # FACE EXPRESSION
    if expressions:
        face_expr = face_visible_expressions.get_face_expression(img)
        if verbose:
            print(f"Frame {frame_idx} face expr {face_expr}")
        logging.info(f"Frame {frame_idx} face expr {face_expr}")

        if face_expr is not None:
            faceexpr_texts.append(face_expr)

    # EMOTION
    if emotions:
        emotion_label = model.fer(img)
        emotion_texts.append(emotion_label)

        if verbose:
            print(f"Frame {frame_idx} face emotion {emotion_label}")
        logging.info(f"Frame {frame_idx} face emotion {emotion_label}")

    if frame_idx % min_num_of_frames == 0:

        if verbose:
            print(f"Summarizing the interval results.")
        logging.info(f"Summarizing the interval results.")

        # GAZE LABEL
        gaze_counter = Counter(gaze_texts)

        try:
            most_common_gaze = gaze_counter.most_common(1)[0][0]
        except:
            most_common_gaze = ""

        gaze_label = most_common_gaze

        # FACE EXPRESSION LABEL
        face_expr_counter = Counter(faceexpr_texts)

        try:
            most_common_face_expr = face_expr_counter.most_common(1)[0][0]
        except:
            most_common_face_expr = ""

        expression_label = most_common_face_expr

        # HEAD SHAKE
        if len(head_shakes) >= 2:
            head_shake_stdev = statistics.stdev(head_shakes)
            head_shake_stdev = head_shake_stdev * 10000
        else:
            head_shake_stdev = 0.0

        if head_shake_stdev >= HEAD_SHAKE_STDEV:
            body_label = "HEAD SHAKE"
        else:
            body_label = ""

        if len(mids) >= 2:
            mids_stdev = statistics.stdev(mids)
            mids_stdev = mids_stdev * 10000
        else:
            mids_stdev = 0.0

        if len(degrees) >= 2:
            degrees_stdev = statistics.stdev(degrees)
            degrees_stdev = degrees_stdev * 10000
        else:
            degrees_stdev = 0.0

        if len(body_label) == 0:
            # print("mids stdev",mids_stdev)
            # print("degrees", degrees_stdev)
            try:
                ratio = degrees_stdev // mids_stdev
            except ZeroDivisionError as err:
                ratio = 0.0
            # print("ratio", ratio)
            if mids_stdev < 130 and mids_stdev > 30 and ratio > 50:
                body_label = "SHOULDER MOVEMENT"
            elif mids_stdev > 100 and degrees_stdev > 2000 and ratio < 50:
                body_label = "HEAD NOD"
            else:
                body_label = ""
        # if len(lefts) >= 2 and len(rights) >=2:
        #     lefts_stdev = statistics.stdev(lefts)
        #     lefts_stdev = lefts_stdev * 10000
        #     rights_stdev = statistics.stdev(rights)
        #     rights_stdev = rights_stdev * 10000
        # else:
        #     lefts_stdev = 0.0
        #     rights_stdev = 0.0
        # if len(body_label) == 0:
        #     print("degrees stdev", degrees_stdev)
        #     print("mids stdev",mids_stdev)
        #     if degrees_stdev >= 3000 and mids_stdev < 40:
        #         body_label = "SHOULDER MOVEMENT"
        #     else:
        #         body_label = ""

        # HEAD NOD
        # if len(body_label) == 0:
        #     if len(head_nods) >= 2:
        #         head_nod_stdev = statistics.stdev(head_nods)
        #         head_nod_stdev = head_nod_stdev * 10000
        #     else:
        #         head_nod_stdev = 0.0
            # print("head nod stdev", head_nod_stdev)
        #     if head_nod_stdev >= 10000:
        #         body_label = "HEAD NOD"
        #     else:
        #         body_label = ""

        # SHOULDER MOVEMENT EXPERIMENT
        # if len(body_label) == 0:
        #     if rights_stdev > 175 and lefts_stdev > 175:
        #         body_label = "SHOULDER MOVEMENT"
        #     else:
        #         body_label = ""
        #
        #     print(lefts_stdev, rights_stdev)

        # SHOULDER MOVEMENT
        # if len(body_label) == 0:
        #     # print("mids stdev",mids_stdev)
        #     if mids_stdev > 80:
        #         body_label = "HEAD NOD"
        #     # elif mids_stdev > 30:
        #     #     body_label = "SHOULDER MOVEMENT"
        #     else:
        #         body_label = ""

        # print(body_label)

        # EMOTION LABEL
        emotion_counter = Counter(emotion_texts)

        try:
            most_common_emotion = emotion_counter.most_common(1)[0][0]
        except:
            most_common_emotion = ""

        emotion_label = most_common_emotion
        if emotion_label == "null":
            emotion_label = ""

        head_nods = []
        gaze_texts = []
        faceexpr_texts = []
        emotion_texts = []
        head_shakes = []
        mids = []
        degrees = []

    if body:
        cv2.putText(img, f"BODY: {body_label}", (20, 130), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA, False)
    if gaze:
        cv2.putText(img, f"GAZE: {gaze_label}", (20, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA, False)
    if expressions:
        cv2.putText(img, f"EXPRESSION: {expression_label}", (20, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA, False)
    if emotions:
        cv2.putText(img, f"EMOTION: {emotion_label}", (20, 160), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA, False)

        # for saving the video with labels (expr, gaze, body)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('analyzed video', img)

    frame_idx += 1
        # out.write(img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
