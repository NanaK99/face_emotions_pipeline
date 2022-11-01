import face_visible_expressions_pipeline
import gaze_estimator_pipeline
import mediapipe_holistic_pipeline
import cv2


CAMERA = 0
cap = cv2.VideoCapture(CAMERA)
font = cv2.FONT_HERSHEY_SIMPLEX

final_text = ""
head_shake_dir = []
head_nod_dir = []
head_shake_idxs = []
head_nod_idxs = []
img_ind = 0

while cap.isOpened():

    success, img = cap.read()
    if not success: break

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model

    final_text += gaze_estimator_pipeline.get_gaze_direction(img)

    body_mov = mediapipe_holistic_pipeline.get_body_movement(img)
    # print("IND", img_ind)
    # print("BODY MOV",body_mov, img_ind)
    # if isinstance(body_mov, tuple):

    if body_mov in ["up", "down", "right", "left"]:
        if body_mov == "left" or body_mov == "right":
            if len(head_shake_idxs) < 3:
                if len(head_shake_idxs) > 0:
                    if head_shake_dir[-1] != body_mov:
                        head_shake_idxs.append(img_ind)
                        head_shake_dir.append(body_mov)
                else:
                    head_shake_idxs.append(img_ind)
                    head_shake_dir.append(body_mov)
            else:
                if head_shake_idxs[-1] - head_shake_idxs[0] < 30:
                    final_text += "HEAD SHAKE, "
                    head_shake_idxs = []
                    head_shake_dir = []
                else:
                    head_shake_idxs = []
                    head_shake_dir = []

        elif body_mov == "up" or body_mov == "down":
            if len(head_nod_idxs) < 3:
                if len(head_nod_idxs) > 0:
                    if head_nod_dir[-1] != body_mov:
                        head_nod_idxs.append(img_ind)
                        head_nod_dir.append(body_mov)
                else:
                    head_nod_idxs.append(img_ind)
                    head_nod_dir.append(body_mov)

            else:
                if head_nod_idxs[-1] - head_nod_idxs[0] < 30:
                    final_text += "HEAD NOD, "
                    head_nod_idxs = []
                    head_nod_dir = []
                else:
                    head_nod_idxs = []
                    head_nod_dir = []
        # print("HEAD NOD", head_nod_idxs, head_nod_dir)

    else:
        final_text += body_mov

    final_text += face_visible_expressions_pipeline.get_face_expression(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.putText(img, final_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.imshow('output window', img)

    final_text = ""
    img_ind += 1

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()