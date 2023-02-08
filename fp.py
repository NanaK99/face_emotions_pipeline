from head_body_movement_detection import mediapipe_face, mediapipe_shoulders
from face_expr_detection import face_visible_expressions
from gaze_detection import gaze_estimator
from emotion_detection import inference
from utils import textgrid_generation, merge_speakers, preprocess_tg, head_nod

from configparser import ConfigParser
from collections import Counter
from praatio import textgrid
import numpy as np
import argparse
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

parser.add_argument('--video', type=str,
                    help='path to the input video', required=True)
parser.add_argument('--input_textgrid', type=str,
                    help='path to the base textgrid', required=True)
parser.add_argument('--output_dir_name', type=str,
                    help='name of the directory where the generated textgrid files should be saved', required=True)
parser.add_argument('--verbose', type=bool,
                    help='a boolean indicating the mode for logging, '
                         'in case of True, prints will also be visible in the terminal; '
                         'otherwise the logs will be kept only in the log file', required=False, default=False)
parser.add_argument('--gaze', default=False,
                    help='gaze only', required=False, action='store_true')
parser.add_argument('--expressions', default=False,
                    help='expressions only', required=False, action='store_true')
parser.add_argument('--body', default=False,
                    help='body only', required=False, action='store_true')
parser.add_argument('--emotions', default=False,
                    help='emotions only', required=False, action='store_true')


parser.add_argument('--debug', default=False,
                    help='emotions only', required=False, action='store_true')

args = parser.parse_args()

video_path = args.video
input_textgrid_path = args.input_textgrid
output_dir_name = args.output_dir_name
verbose = args.verbose

gaze = args.gaze
body = args.body
expressions = args.expressions
emotions = args.emotions

debug = args.debug

type_of_run = sys.argv[1].strip("--")
log_file = type_of_run+paths["LOG_FILE"]

log_file_exists = os.path.exists(log_file)
if log_file_exists:
    os.remove(log_file)
logging.basicConfig(filename=log_file, level=logging.INFO,  format="%(asctime)s %(message)s", filemode="a")

VID_EXTENSIONS = ["mp4"]
TEXTGRID_EXTENSIONS = ["TextGrid"]

error_path = ""

vid_exist = os.path.exists(video_path)
textgrid_path_exist = os.path.exists(input_textgrid_path)

if (not vid_exist):
    error_path = video_path
if (not textgrid_path_exist):
    error_path = input_textgrid_path

if len(error_path) != 0:
    sys.stderr.write(f"{error_path} does not exist!")
    sys.exit(2)

if video_path.split(".")[-1] not in VID_EXTENSIONS:
    sys.stderr.write(f"{video_path} is not in the correct format!")
    sys.exit(2)


if input_textgrid_path.split(".")[-1] not in TEXTGRID_EXTENSIONS:
    sys.stderr.write(f"{input_textgrid_path} is not in the correct format!")
    sys.exit(2)

directory_path = os.path.join("./", output_dir_name)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)  # 25
font = cv2.FONT_HERSHEY_SIMPLEX

model = inference.Model()

BODY = config_object["BODY_MOVEMENT"]
IGNORE_EXPRS = config_object["IGNORE_EXPRS"]

HEAD_NOD = config_object["HEAD_MOVEMENT"]

paths = config_object["PATHS"]
merged_tg_path = paths["MERGED_TEXTGRID_PATH"]
final_tg_path = paths["PREPROCESSED_TEXTGRID_PATH"]

merged_tg_path = "".join(list(merged_tg_path.split(".TextGrid"))+[type_of_run, ".TextGrid"])
final_tg_path = "".join(list(final_tg_path.split(".TextGrid"))+[type_of_run, ".TextGrid"])


merged_textgrid_path = merge_speakers.main(input_textgrid_path, merged_tg_path, verbose)
final_tg_path = preprocess_tg.main(merged_textgrid_path, final_tg_path, verbose)
tg = textgrid.openTextgrid(final_tg_path, includeEmptyIntervals=True)

video_parameters = config_object["VIDEO_PARAMETERS"]
min_num_of_frames = int(video_parameters["MIN_NUM_OF_FRAMES"])

ignore_exprs = []
for expr in IGNORE_EXPRS:
    ignore_exprs.append(IGNORE_EXPRS[expr])


one_shoulder_movement = int(BODY["SHOULDER_ANGLE_STD"])
both_shoulder_movement = float(BODY["BOTH_SHOULDERS_Y_STD"])
lean_in_out_thresh = float(BODY["BOTH_SHOULDERS_Z_DIFF"])
num_of_diff_head_positions = int(BODY["NUM_OF_DIFF_HEAD_POSITIONS"])

headnod_normal = float(HEAD_NOD["HEADNOD_MEAN"])
headnod_std = float(HEAD_NOD["HEADNOD_STD"])

gaze_texts = []
headshake_texts = []
headnod_rights = []
headnod_lefts = []
faceexpr_texts = []
shoulder_texts = []
emotion_texts = []

one_up_down = []
both_up_down = []
lean_in_out = []
head_shake_dir = []
head_nod_dir = []
mids = []

shoulder_text = ""
head_nod_text = ""

new_tier_name_list = []

tier_name_list = tg.tierNameList

# out = cv2.VideoWriter('debug.avi', -1, 20.0, (640,480))  # for saving the video with outputs labels (gaze, expression, body)


while cap.isOpened():
    tg_gaze = textgrid.Textgrid()
    tg_expr = textgrid.Textgrid()
    tg_body = textgrid.Textgrid()
    tg_emotion = textgrid.Textgrid()

    for tier_name in tier_name_list:
        if verbose:
            print(f"Working on {tier_name}.")
        logging.info(f"Working on {tier_name}.")
        gaze_entrylist = []
        expr_entrylist = []
        body_entrylist = []
        emotion_entrylist = []
        tier = tg.tierDict[tier_name]
        entryList = tier.entryList

        try:
            for idx, entry in enumerate(entryList):
                if verbose:
                    print(f"Working on interval {idx}...")
                logging.info(f"WORKING ON INTERVAL {idx}...")
                start = entry.start
                end = entry.end
                label = entry.label
                total_sec = end - start
                num_frames = total_sec * 25
                cap.set(cv2.CAP_PROP_POS_FRAMES, round(start * fps))
                if label != "0" or label in ignore_exprs:
                    logging.info(f"FRAME NUMBER {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
                    # print("num of frames, minim num of frames", int(num_frames), min_num_of_frames)
                    if int(num_frames) > min_num_of_frames:
                        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        while frame_idx < end*fps:
                            success, img = cap.read()
                            # To improve performance, optionally mark the image as not writeable to pass by reference.
                            img.flags.writeable = False
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model

                            # GAZE
                            if gaze:
                                gaze_all = gaze_estimator.get_gaze_direction(img, )
                                if gaze_all is not None:
                                    g_text = gaze_all[3]
                                else:
                                    g_text = ""
                                logging.info(f"Frame {frame_idx} Gaze {g_text}")

                                gaze_texts.append(g_text)
                                logging.info(f"Frame {frame_idx} Gaze {g_text}")

                            if body:
                                # HEAD SHAKE
                                face_mov = mediapipe_face.get_face_movement(img)
                                logging.info(f"Frame {frame_idx} face mov {face_mov}")

                                if face_mov is not None:

                                    if face_mov == "left" or face_mov == "right":
                                        if len(head_shake_dir) > 0:
                                            if head_shake_dir[-1] != face_mov:
                                                head_shake_dir.append(face_mov)
                                        else:
                                            head_shake_dir.append(face_mov)

                                    else:
                                        head_shake_dir.append(face_mov)
                                else:
                                    head_shake_dir.append("")

                                # HEAD  NOD or SHOULDER MOVEMENT
                                mids.append(mediapipe_shoulders.get_body_movement(img))

                                body_move = mediapipe_shoulders.get_body_movement(img)
                                logging.info(f"Frame {frame_idx} body move {body_move}")

                            if expressions:
                                # FACE EXPRESSION
                                face_expr = face_visible_expressions.get_face_expression(img)
                                logging.info(f"Frame {frame_idx} face expr {face_expr}")

                                if face_expr is not None:
                                    faceexpr_texts.append(face_expr)

                            if emotions:
                                # EMOTION
                                emotion_label = model.fer(img)
                                emotion_texts.append(emotion_label)

                            frame_idx += 1

                        else:
                            if not debug:
                            
                                logging.info(f"Summarizing the interval results.")

                                gaze_counter = Counter(gaze_texts)
                                face_expr_counter = Counter(faceexpr_texts)
                                emotion_counter = Counter(emotion_texts)

                                try:
                                    most_common_emotion = emotion_counter.most_common(1)[0][0]
                                except:
                                    most_common_emotion = ""

                                head_shake_dir_len = len(head_shake_dir)

                                if len(head_shake_dir) / 3 > 1:
                                    head_shake_text = "HEAD SHAKE"
                                else:
                                    head_shake_text = ""

                                try:
                                    most_common_gaze = gaze_counter.most_common(1)[0][0]
                                except:
                                    most_common_gaze = ""

                                try:
                                    most_common_face_expr = face_expr_counter.most_common(1)[0][0]
                                except:
                                    most_common_face_expr = ""

                                # HEAD NOD or SHOULDER MOVEMENT
                                mids_stdev = statistics.stdev(mids)
                                mids_stdev = mids_stdev * 10000
                                if mids_stdev > NOD_STDEV:
                                    text = "NOD"
                                elif (mids_stdev < SHOULDER_STDEV_UPPER_BOUND and mids_stdev > SHOULDER_STDEV_LOWER_BOUND):
                                    text = "SHOULDERS"
                                else:
                                    text = ""

                                gaze_texts = []
                                headshake_texts = []
                                faceexpr_texts = []
                                emotion_texts = []
                                head_shake_dir = []
                                mids = []

                                gaze_label = most_common_gaze
                                expression_label = most_common_face_expr

                                if text == "NOD":
                                    head_nod_text = "HEAD NOD"
                                elif text == "SHOULDERS":
                                    shoulder_text = "SHOULDER MOVEMENT"

                                if len(head_shake_text) == 0 and len(head_nod_text) == 0:
                                    most_common_head_move = ""
                                elif len(head_shake_text) != 0 and len(head_nod_text) == 0:
                                    most_common_head_move = f"{head_shake_text}"
                                elif len(head_shake_text) == 0 and len(head_nod_text) != 0:
                                    most_common_head_move = f"{head_nod_text}"
                                else:
                                    most_common_head_move = f"{head_shake_text}, {head_nod_text}"

                                if len(shoulder_text) == 0 and len(most_common_head_move) == 0:
                                    body_label = ""
                                elif len(shoulder_text) == 0 and len(most_common_head_move) != 0:
                                    body_label = most_common_head_move
                                elif len(shoulder_text) != 0 and len(most_common_head_move) == 0:
                                    body_label = shoulder_text
                                else:
                                    body_label = f"{most_common_head_move}, {shoulder_text}"

                                emotion_label = most_common_emotion
                                if emotion_label == "null":
                                    emotion_label = ""

                                gaze_entrylist.append((start, end, gaze_label))
                                expr_entrylist.append((start, end, expression_label))
                                body_entrylist.append((start, end, body_label))
                                emotion_entrylist.append((start, end, emotion_label))

                                # for saving the video with labels (expr, gaze, body)
                                final_text = f"{gaze_label}, {expression_label}, {body_label}"
                                # cv2.putText(img, final_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
                                # cv2.imshow('Recording...', img)
                                # out.write(img)
                                
                            else:
                                gaze_label = ','.join(gaze_texts)
                                expression_label = ','.join(faceexpr_texts)
                                body_label = ','.join(headmove_texts)
                                emotion_label = ','.join(emotion_texts)

                                gaze_entrylist.append((start, end, gaze_label))
                                expr_entrylist.append((start, end, expression_label))
                                body_entrylist.append((start, end, body_label))
                                emotion_entrylist.append((start, end, emotion_label))

                    else:
                        label = ""
                        gaze_entrylist.append((start, end, label))
                        expr_entrylist.append((start, end, label))
                        body_entrylist.append((start, end, label))
                        emotion_entrylist.append((start, end, label))

            if verbose:
                print(f"Done with {tier_name}.")
            logging.info(f"Done with {tier_name}.")
            textgrid_paths = textgrid_generation.save_textgrids(tier, gaze_entrylist, expr_entrylist, body_entrylist,
                                               output_dir_name, tg_gaze, tg_expr, tg_body, tier_name, type_of_run)

        except KeyboardInterrupt:
            logging.exception("Keyboard Interrupt.")
            textgrid_generation.save_textgrids(tier, gaze_entrylist, expr_entrylist, body_entrylist,
                                      output_dir_name, tg_gaze, tg_expr, tg_body, tier_name,type_of_run)

            sys.exit()
            cap.release()
            cv2.destroyAllWindows()

    if verbose:
        print(f"File {textgrid_paths[0].split('/')[-1]} successfully saved!")
        print(f"File {textgrid_paths[1].split('/')[-1]} successfully saved!")
        print(f"File {textgrid_paths[2].split('/')[-1]} successfully saved!")
        print(f"File {textgrid_paths[3].split('/')[-1]} successfully saved!")

    logging.info(f"File {textgrid_paths[0].split('/')[-1]} successfully saved!")
    logging.info(f"File {textgrid_paths[1].split('/')[-1]} successfully saved!")
    logging.info(f"File {textgrid_paths[2].split('/')[-1]} successfully saved!")
    logging.info(f"File {textgrid_paths[3].split('/')[-1]} successfully saved!")

    cap.release()
    cv2.destroyAllWindows()
