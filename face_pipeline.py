from head_body_movement_detection import mediapipe_face, mediapipe_shoulders
from face_expr_detection import face_visible_expressions
from gaze_detection import gaze_estimator
from emotion_detection import inference
from utils import textgrid_generation

from configparser import ConfigParser
from collections import Counter
from praatio import tgio
import numpy as np
import argparse
import cv2
import sys
import os


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('Error: %s\n' % message)
        self.print_help()
        sys.exit(2)


parser = MyParser()

parser.add_argument('--video', type=str,
                    help='path to the input video', required=True)
parser.add_argument('--input_textgrid', type=str,
                    help='path to the base textgrid', required=True)
parser.add_argument('--output_dir_name', type=str,
                    help='name of the directory where the generated textgrid files should be saved', required=True)

args = parser.parse_args()

video_path = args.video
input_textgrid_path = args.input_textgrid
output_dir_name = args.output_dir_name

VID_EXTENSIONS = ["mp4"]
TEXTGRID_EXTENSIONS = ["TextGrid"]

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
tg = tgio.openTextgrid(input_textgrid_path)

fps = cap.get(cv2.CAP_PROP_FPS)
font = cv2.FONT_HERSHEY_SIMPLEX

model = inference.Model()

config_object = ConfigParser()
config_object.read("./static/config.ini")
BODY = config_object["BODY_MOVEMENT"]

one_shoulder_movement = int(BODY["SHOULDER_ANGLE_STD"])
both_shoulder_movement = float(BODY["BOTH_SHOULDERS_Y_STD"])
lean_in_out_thresh = float(BODY["BOTH_SHOULDERS_Z_DIFF"])
num_of_diff_head_positions = int(BODY["NUM_OF_DIFF_HEAD_POSITIONS"])

gaze_texts = []
headmove_texts = []
faceexpr_texts = []
shoulder_texts = []
emotion_texts = []

one_up_down = []
both_up_down = []
lean_in_out = []
head_shake_dir = []
head_nod_dir = []
head_shake_idxs = []
head_nod_idxs = []

tier_name_list = tg.tierNameList


while cap.isOpened():
    # tier_name_list = ["Nicole-Dressel - words"]  # for quick experiments
    tg_gaze = tgio.Textgrid()
    tg_expr = tgio.Textgrid()
    tg_body = tgio.Textgrid()
    tg_emotion = tgio.Textgrid()

    for tier_name in tier_name_list:
        print(f"Working on {tier_name}")
        gaze_entrylist = []
        expr_entrylist = []
        body_entrylist = []
        emotion_entrylist = []
        tier = tg.tierDict[tier_name]
        entryList = tier.entryList

        try:
            for idx, entry in enumerate(entryList):
                success, img = cap.read()
                if not success: break
                print(f"Working on interval {idx}.....")
                start = entry.start
                end = entry.end
                label = entry.label
                total_sec = end - start
                num_frames = total_sec * 25
                cap.set(cv2.CAP_PROP_POS_FRAMES, round(start * fps))
                if label != "0":
                    if int(num_frames) > 0:
                        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        while frame_idx < end*fps:
                            # To improve performance, optionally mark the image as not writeable to pass by reference.
                            img.flags.writeable = False
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model

                            # GAZE
                            gaze_all = gaze_estimator.get_gaze_direction(img)
                            if gaze_all is not None:
                                g_text = gaze_all[3]
                            else:
                                g_text = ""
                            gaze_texts.append(g_text)

                            # HEAD MOVEMENT
                            face_mov = mediapipe_face.get_face_movement(img)

                            if face_mov is not None:

                                if face_mov in ["up", "down", "right", "left"]:
                                    if face_mov == "left" or face_mov == "right":
                                        if len(head_shake_idxs) > 0:
                                            if head_shake_idxs[-1] < frame_idx:
                                                if head_shake_dir[-1] != face_mov:
                                                    head_shake_idxs.append(frame_idx)
                                                    head_shake_dir.append(face_mov)
                                            else:
                                                head_shake_idxs.append(frame_idx)
                                        else:
                                            head_shake_idxs.append(frame_idx)
                                            head_shake_dir.append(face_mov)

                                    elif face_mov == "up" or face_mov == "down":
                                        if len(head_nod_idxs) > 0:
                                            if head_nod_idxs[-1] < frame_idx:
                                                if head_nod_dir[-1] != face_mov:
                                                    head_nod_idxs.append(frame_idx)
                                                    head_nod_dir.append(face_mov)
                                            else:
                                                head_nod_idxs.append(frame_idx)
                                        else:
                                            head_nod_idxs.append(frame_idx)
                                            head_nod_dir.append(face_mov)

                                else:
                                    headmove_texts.append(face_mov)
                            else:
                                headmove_texts.append("")

                            # SHOULDER MOVEMENT
                            body_move = mediapipe_shoulders.get_body_movement(img)
                            if body_move is not None:
                                one_up_down.append(body_move[0])
                                both_up_down.append(body_move[1])
                                lean_in_out.append(body_move[2])

                            # FACE EXPRESSION
                            face_expr = face_visible_expressions.get_face_expression(img)

                            if face_expr is not None:
                                faceexpr_texts.append(face_expr)

                            # EMOTION
                            emotion_label = model.fer(img)
                            emotion_texts.append(emotion_label)

                            frame_idx += 1

                        else:
                            gaze_counter = Counter(gaze_texts)
                            mediapipe_counter = Counter(headmove_texts)
                            face_expr_counter = Counter(faceexpr_texts)
                            emotion_counter = Counter(emotion_texts)

                            try:
                                most_common_emotion = emotion_counter.most_common(1)[0][0]
                            except:
                                most_common_emotion = ""

                            one_shoulder_up_down = np.std(one_up_down)
                            both_shoulder_up_down = np.std(both_up_down)

                            if len(one_up_down) > 0 and len(both_up_down) and len(lean_in_out) > 0:
                                if one_shoulder_up_down > one_shoulder_movement or both_shoulder_up_down > both_shoulder_movement:
                                    most_common_body_move = "SHOULDER MOVEMENT"
                                elif (abs(np.min(lean_in_out)) - abs(np.mean(lean_in_out))) < -1 * lean_in_out_thresh:
                                    most_common_body_move = "LEAN IN"
                                elif (abs(np.max(lean_in_out)) - abs(np.mean(lean_in_out))) > lean_in_out_thresh:
                                    most_common_body_move = "LEAN OUT"
                                else:
                                    most_common_body_move = ""
                            else:
                                most_common_body_move = ""

                            head_nod_dir_len = len(head_nod_dir)
                            head_shake_dir_len = len(head_shake_dir)

                            if len(headmove_texts) > 0:
                                mediapipe_counter_most_common = mediapipe_counter.most_common(1)[0][1]
                            else:
                                mediapipe_counter_most_common = 0

                            if head_shake_dir_len // num_of_diff_head_positions > mediapipe_counter_most_common and head_shake_dir_len // num_of_diff_head_positions > head_nod_dir_len // num_of_diff_head_positions:
                                most_common_head_move = "HEAD SHAKE"
                            elif head_nod_dir_len // num_of_diff_head_positions > mediapipe_counter_most_common and head_nod_dir_len // num_of_diff_head_positions > head_shake_dir_len // num_of_diff_head_positions:
                                most_common_head_move = "HEAD NOD"
                            else:
                                most_common_head_move = ""

                            try:
                                most_common_gaze = gaze_counter.most_common(1)[0][0]
                            except:
                                most_common_gaze = ""

                            try:
                                most_common_face_expr = face_expr_counter.most_common(1)[0][0]
                            except:
                                most_common_face_expr = ""

                            gaze_texts = []
                            headmove_texts = []
                            faceexpr_texts = []
                            shoulder_texts = []
                            emotion_texts = []

                            head_nod_idxs = []
                            head_shake_idxs = []
                            head_shake_dir = []
                            head_nod_dir = []
                            one_up_down = []
                            both_up_down = []
                            lean_in_out = []

                            gaze_label = most_common_gaze
                            expression_label = most_common_face_expr

                            if len(most_common_body_move) == 0 and len(most_common_head_move) == 0:
                                body_label = ""
                            elif len(most_common_body_move) == 0 and len(most_common_head_move) !=0:
                                body_label = most_common_head_move
                            elif len(most_common_head_move) == 0 and len(most_common_body_move) != 0:
                                body_label = most_common_body_move
                            else:
                                body_label = f"{most_common_head_move}, {most_common_body_move}"

                            emotion_label = most_common_emotion
                            if emotion_label == "null":
                                emotion_label = ""

                            gaze_entrylist.append((start, end, gaze_label))
                            expr_entrylist.append((start, end, expression_label))
                            body_entrylist.append((start, end, body_label))
                            emotion_entrylist.append((start, end, emotion_label))

                    else:
                        gaze_entrylist.append((start, end, label))
                        expr_entrylist.append((start, end, label))
                        body_entrylist.append((start, end, label))
                        emotion_entrylist.append((start, end, label))

            print(f"Done with {tier_name}")
            textgrid_paths = textgrid_generation.save_textgrids(tier, gaze_entrylist, expr_entrylist, body_entrylist, emotion_entrylist,
                                               output_dir_name, tg_gaze, tg_expr, tg_body, tg_emotion)
            # gaze_tier = tier.new(entryList=gaze_entrylist)
            # expr_tier = tier.new(entryList=expr_entrylist)
            # body_tier = tier.new(entryList=body_entrylist)
            # emotion_tier = tier.new(entryList=emotion_entrylist)
            #
            # tg_gaze.addTier(gaze_tier)
            # tg_expr.addTier(expr_tier)
            # tg_body.addTier(body_tier)
            # tg_emotion.addTier(emotion_tier)
            #
            # gaze_output_file_path = os.path.join(directory_path, 'gaze_output.TextGrid')
            # expr_output_file_path = os.path.join(directory_path, 'expr_output.TextGrid')
            # body_output_file_path = os.path.join(directory_path, 'body_output.TextGrid')
            # emotion_output_file_path = os.path.join(directory_path, 'emotion_output.TextGrid')
            #
            # tg_gaze.save(gaze_output_file_path, useShortForm=False)
            # tg_expr.save(expr_output_file_path, useShortForm=False)
            # tg_body.save(body_output_file_path, useShortForm=False)
            # tg_emotion.save(emotion_output_file_path, useShortForm=False)

            print(f"File {textgrid_paths[0]} successfully saved!")
            print(f"File {textgrid_paths[1]} successfully saved!")
            print(f"File {textgrid_paths[2]} successfully saved!")
            print(f"File {textgrid_paths[3]} successfully saved!")

        except:
            textgrid_generation.save_textgrids(tier, gaze_entrylist, expr_entrylist, body_entrylist, emotion_entrylist,
                                      output_dir_name, tg_gaze, tg_expr, tg_body, tg_emotion)
            # gaze_tier = tier.new(entryList=gaze_entrylist)
            # expr_tier = tier.new(entryList=expr_entrylist)
            # body_tier = tier.new(entryList=body_entrylist)
            # emotion_tier = tier.new(entryList=emotion_entrylist)
            #
            # tg_gaze.addTier(gaze_tier)
            # tg_expr.addTier(expr_tier)
            # tg_body.addTier(body_tier)
            # tg_emotion.addTier(emotion_tier)
            #
            # directory_path = os.path.join("./", output_dir_name)
            # if not os.path.exists(directory_path):
            #     os.makedirs(directory_path)
            #
            # gaze_output_file_path = os.path.join(directory_path, 'gaze_output.TextGrid')
            # expr_output_file_path = os.path.join(directory_path, 'expr_output.TextGrid')
            # body_output_file_path = os.path.join(directory_path, 'body_output.TextGrid')
            # emotion_output_file_path = os.path.join(directory_path, 'emotion_output.TextGrid')
            #
            # tg_gaze.save(gaze_output_file_path, useShortForm=False)
            # tg_expr.save(expr_output_file_path, useShortForm=False)
            # tg_body.save(body_output_file_path, useShortForm=False)
            # tg_emotion.save(emotion_output_file_path, useShortForm=False)

            sys.exit()
            cap.release()
            cv2.destroyAllWindows()