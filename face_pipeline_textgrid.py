import face_visible_expressions
import gaze_estimator
import mediapipe_holistic
import cv2
from collections import Counter


new_data = ['File type = "ooTextFile"\n', 'Object class = "TextGrid"\n', '\n',
            'xmin = 0\n', 'xmax = 10175.872\n', 'tiers? <exists>\n',
            'size = 5\n', '    class = "IntervalTier"\n',
            '    xmin = 0\n', '    xmax = 10175.872\n', '    intervals: size = 3391\n']


video_path = "vid.mp4"
cap = cv2.VideoCapture(video_path)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second is equal to {fps}.")
duration_sec = frame_count/fps
duration_min = duration_sec/60
print(f"Duration of {video_path} is {duration_min:.2f} minutes.")

interval_sec = 3
number_of_frames_in_interval = fps * interval_sec

font = cv2.FONT_HERSHEY_SIMPLEX

gaze_text = []
mediapipe_text = []
faceexpr_text = []

head_shake_dir = []
head_nod_dir = []
head_shake_idxs = []
head_nod_idxs = []

frame_idx = 0
interval_idx = 1
interval_size = 3391

x_min = 0
x_max = interval_sec

while cap.isOpened():

    if interval_idx <= 5:
        print(f"Processing {interval_idx} interval")
        if frame_idx <= number_of_frames_in_interval:
            print(f"Processing {frame_idx} frame")
            success, img = cap.read()
            if not success: break

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
            gaze = gaze_estimator.get_gaze_direction(img)
            if gaze is not None:
                gaze_text.append(gaze_estimator.get_gaze_direction(img))

            body_mov = mediapipe_holistic.get_body_movement(img)

            if body_mov in ["up", "down", "right", "left"]:
                if body_mov == "left" or body_mov == "right":
                    if len(head_shake_idxs) > 0:
                        if head_shake_idxs[-1] + 10 < frame_idx:
                            if head_shake_dir[-1] != body_mov:
                                head_shake_idxs.append(frame_idx)
                                head_shake_dir.append(body_mov)
                        else:
                            head_shake_idxs.append(frame_idx)
                    else:
                        head_shake_idxs.append(frame_idx)
                        head_shake_dir.append(body_mov)

                elif body_mov == "up" or body_mov == "down":
                    if len(head_nod_idxs) > 0:
                        if head_nod_idxs[-1] + 10 < frame_idx:
                            if head_nod_dir[-1] != body_mov:
                                head_nod_idxs.append(frame_idx)
                                head_nod_dir.append(body_mov)
                        else:
                            head_nod_idxs.append(frame_idx)
                    else:
                        head_nod_idxs.append(frame_idx)
                        head_nod_dir.append(body_mov)

            else:
                mediapipe_text.append(body_mov)

            face_expr = face_visible_expressions.get_face_expression(img)
            if face_expr is not None:
                faceexpr_text.append(face_expr)

            frame_idx += 1

        else:
            gaze_counter = Counter(gaze_text)
            mediapipe_counter = Counter(mediapipe_text)
            face_expr_counter = Counter(faceexpr_text)

            head_nod_dir_len = len(head_nod_dir)
            head_shake_dir_len = len(head_shake_dir)
            if head_shake_dir_len // 3 > mediapipe_counter.most_common(1)[0][1] and head_shake_dir_len // 3 > head_nod_dir_len // 3:
                most_common_mediapipe = "HEAD SHAKE"
            if head_nod_dir_len // 3 > mediapipe_counter.most_common(1)[0][1] and head_nod_dir_len // 3 > head_shake_dir_len // 3:
                most_common_mediapipe = "HEAD NOD"
            else:
                most_common_mediapipe = mediapipe_counter.most_common(1)[0][0]

            most_common_gaze = gaze_counter.most_common(1)[0][0]
            most_common_face_expr = face_expr_counter.most_common(1)[0][0]

            final_text = f"{most_common_gaze}, {most_common_mediapipe}, {most_common_face_expr}"

            new_data.append("    " + f"intervals [{interval_idx}]" + ":\n")
            new_data.append("            " + f"xmin = {x_min}" + "\n")
            new_data.append("            " + f"xmax = {x_max}" + "\n")
            new_data.append("            " + f"text = {final_text}" + "\n")
            x_min += 3
            x_max += 3
            final_text = ""
            frame_idx = 0
            interval_idx += 1

    else:
        cap.release()
        cv2.destroyAllWindows()

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

with open('face_pipeline_output.TextGrid', 'w') as file:
    file.writelines(new_data)
