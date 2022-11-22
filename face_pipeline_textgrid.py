import face_visible_expressions
import gaze_estimator
import cv2
import time
import mediapipe_face
from collections import Counter
from praatio import tgio
from timeit import default_timer as timer


video_path = "vid.mp4"
cap = cv2.VideoCapture(video_path)

tg = tgio.openTextgrid("trott.TextGrid")

# frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
# print(f"Frames per second is equal to {fps}.")
# duration_sec = frame_count/fps
# duration_min = duration_sec/60
# print(f"Duration of {video_path} is {duration_min:.2f} minutes.")

# interval_sec = 3
# number_of_frames_in_interval = fps * interval_sec

font = cv2.FONT_HERSHEY_SIMPLEX

gaze_texts = []
mediapipe_texts = []
faceexpr_texts = []

head_shake_dir = []
head_nod_dir = []
head_shake_idxs = []
head_nod_idxs = []

# frame_idx = 0
# interval_idx = 1
# interval_size = duration_sec/interval_sec

# x_min = 0
# x_max = interval_sec

while cap.isOpened():
    tier_name_list = ["Diane-Fetterman---Gap-International - words", "Jeannet-Trott - words", "Nicole-Dressel - words",
                       "TERRENCE - words", "mgarner - words",]
    tg_new = tgio.Textgrid()

    for tier_name in tier_name_list:
        print(f"Working on {tier_name}")
        new_entrylist = []
        tier = tg.tierDict[tier_name]
        entryList = tier.entryList
        for idx, entry in enumerate(entryList):
            success, img = cap.read()
            if not success: break
            print(f"Working on interval {idx}.....")
            # print(f"Working on {entry} interval")
            start = entry.start
            end = entry.end
            label = entry.label
            total_sec = end - start
            num_frames = total_sec * 25
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(start * fps))
            # print("########", num_frames, label)
            if label != "0":
                if int(num_frames) > 0:
                    frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    # print("frame idx, end", frame_idx, end*fps)
                    while frame_idx < end*fps:
                        # print("%%%%%%")
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

                        # FACE MOVEMENT
                        face_mov = mediapipe_face.get_face_movement(img)

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
                            mediapipe_texts.append(face_mov)

                        # FACE EXPRESSION
                        face_expr = face_visible_expressions.get_face_expression(img)

                        if face_expr is not None:
                            faceexpr_texts.append(face_expr)

                        frame_idx += 1
                        # print(len(gaze_texts), len(mediapipe_texts), len(faceexpr_texts))

                    else:
                        # print(gaze_texts, mediapipe_texts, faceexpr_texts)
                        gaze_counter = Counter(gaze_texts)
                        mediapipe_counter = Counter(mediapipe_texts)
                        face_expr_counter = Counter(faceexpr_texts)

                        head_nod_dir_len = len(head_nod_dir)
                        head_shake_dir_len = len(head_shake_dir)

                        if len(mediapipe_texts) > 0:
                            mediapipe_counter_most_common = mediapipe_counter.most_common(1)[0][1]
                        else:
                            mediapipe_counter_most_common = 0

                        if head_shake_dir_len // 3 > mediapipe_counter_most_common and head_shake_dir_len // 3 > head_nod_dir_len // 3:
                            most_common_mediapipe = "HEAD SHAKE"
                        elif head_nod_dir_len // 3 > mediapipe_counter_most_common and head_nod_dir_len // 3 > head_shake_dir_len // 3:
                            most_common_mediapipe = "HEAD NOD"
                        else:
                            most_common_mediapipe = "NORMAL POSTURE"

                        try:
                            most_common_gaze = gaze_counter.most_common(1)[0][0]
                        except:
                            most_common_gaze = ""

                        try:
                            most_common_face_expr = face_expr_counter.most_common(1)[0][0]
                        except:
                            most_common_face_expr = ""

                        gaze_texts = []
                        mediapipe_texts = []
                        faceexpr_texts = []

                        head_nod_idxs = []
                        head_shake_idxs = []
                        head_shake_dir = []
                        head_nod_dir = []

                        label = label + f", {most_common_gaze}, {most_common_mediapipe}, {most_common_face_expr}"
                        # print("label", label)
                        new_entrylist.append((start, end, label))
                        # print("new entrylist", len(new_entrylist))
                else:
                    new_entrylist.append((start, end, label))
        print(f"Done with {tier_name}")
        new_tier = tier.new(entryList=new_entrylist)
        tg_new.addTier(new_tier)
        output_file_path = 'all_output.TextGrid'
        tg_new.save(output_file_path, useShortForm=False)
        print(f"File {output_file_path} successfully saved!")
    cap.release()
    cv2.destroyAllWindows()




