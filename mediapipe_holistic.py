import cv2
import mediapipe as mp
from scipy.spatial import distance as dist


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def get_aspect_ratio(top, bottom, right, left):
  height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
  width = dist.euclidean([right.x, right.y], [left.x, left.y])
  return height / width


cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) \
        as holistic:

    img_idxs = []
    head_shake_ratios = []
    head_shake_dir = []

    head_nod_dir = []
    head_nod_ratios = []
    head_nod_idxs = []

    img_ind = 0
    turn = ""
    nod = ""

    while cap.isOpened():

        text = ""

        success, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        # Process the image and detect the holistic
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        landmarks = results.pose_landmarks.landmark

        # LEAN IN and LEAN OUT
        if len(text) == 0:
            try:
                eyeR_outer = landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER.value]
                eyeL_outer = landmarks[mp_holistic.PoseLandmark.LEFT_EYE_OUTER.value]
                eyeR_inner = landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_INNER.value]
                eyeL_inner = landmarks[mp_holistic.PoseLandmark.LEFT_EYE_INNER.value]
                eyeR_len = dist.euclidean([eyeR_inner.x, eyeR_inner.y], [eyeR_outer.x, eyeR_outer.y])
                eyeL_len = dist.euclidean([eyeL_inner.x, eyeL_inner.y], [eyeL_outer.x, eyeL_outer.y])
                if eyeR_len > 0.06 and eyeL_len > 0.06:
                    text = "LEAN IN"
                elif eyeR_len < 0.05 and eyeL_len < 0.05:
                    text = "LEAN OUT"
                else:
                    text = ""

            except:
                pass

        # HEAD SHAKE
        if len(text) == 0:
            try:
                earL = landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value]
                earR = landmarks[mp_holistic.PoseLandmark.RIGHT_EAR.value]
                eyeR_outer = landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER.value]
                eyeL_outer = landmarks[mp_holistic.PoseLandmark.LEFT_EYE_OUTER.value]
                right_to_left_ratio = get_aspect_ratio(earR, eyeR_outer, earL, eyeL_outer)

                mouthR = landmarks[mp_holistic.PoseLandmark.MOUTH_RIGHT.value]
                mouthL = landmarks[mp_holistic.PoseLandmark.MOUTH_LEFT.value]
                mouth_len = dist.euclidean([mouthR.x, mouthR.y], [mouthL.x, mouthL.y])

                if len(head_shake_dir) < 3:

                    if len(turn) == 0:
                        if right_to_left_ratio > 1.3:
                            turn = "right"
                            head_shake_dir.append(turn)
                            img_idxs.append(img_ind)
                        elif right_to_left_ratio < 0.7:
                            turn = "left"
                            head_shake_dir.append(turn)
                            img_idxs.append(img_ind)

                    elif right_to_left_ratio > 1.3:
                        turn = "right"
                        if turn != head_shake_dir[-1]:
                            head_shake_dir.append(turn)
                            img_idxs.append(img_ind)
                        else:
                            continue

                    elif right_to_left_ratio < 0.7:
                        turn = "left"
                        if turn != head_shake_dir[-1]:
                            head_shake_dir.append(turn)
                            img_idxs.append(img_ind)
                        else:
                            continue

                else:
                    if img_idxs[-1] - img_idxs[0] < 50:
                        text = "HEAD SHAKE"
                        turn = ""
                        head_shake_dir = []
                        img_idxs = []
                    else:
                        turn = ""
                        head_shake_dir = []
                        img_idxs = []

            except:
                pass

        #HEAD NOD
        if len(text) == 0:
            try:
                eyeR_outer = landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER.value]
                eyeL_outer = landmarks[mp_holistic.PoseLandmark.LEFT_EYE_OUTER.value]
                eyeR_inner = landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_INNER.value]
                eyeL_inner = landmarks[mp_holistic.PoseLandmark.LEFT_EYE_INNER.value]
                nodR_inner_ratio = 1 / eyeR_inner.z
                nodR_outer_ratio = 1 / eyeR_outer.z
                nodL_inner_ratio = 1 / eyeL_inner.z
                nodL_outer_ratio = 1 / eyeL_outer.z

                if len(head_nod_dir) < 3:
                    if len(nod) == 0:

                        if abs(nodL_inner_ratio) > 0.33 and abs(nodR_inner_ratio) > 0.33:
                            nod = "down"
                            head_nod_dir.append(nod)
                            head_nod_idxs.append(img_ind)

                        elif abs(nodR_inner_ratio) < 0.27 and abs(nodR_inner_ratio) < 0.27:
                            nod = "up"
                            head_nod_dir.append(nod)
                            head_nod_idxs.append(img_ind)

                    elif abs(nodL_inner_ratio) > 0.33 and abs(nodR_inner_ratio) > 0.33:
                        nod = "down"
                        if nod != head_nod_dir[-1]:
                            head_nod_dir.append(nod)
                            head_nod_idxs.append(img_ind)
                        else:
                            continue

                    elif abs(nodR_inner_ratio) < 0.27 and abs(nodR_inner_ratio) < 0.27:
                        nod = "up"
                        if nod != head_nod_dir[-1]:
                            head_nod_dir.append(nod)
                            head_nod_idxs.append(img_ind)
                        else:
                            continue

                else:
                    if head_nod_idxs[-1] - head_nod_idxs[0] < 50:
                        text = "HEAD NOD"
                        nod = ""
                        head_nod_dir = []
                        head_nod_idxs = []
                    else:
                        nod = ""
                        head_nod_dir = []
                        head_nod_idxs = []

            except:
                pass

        # SHOULDER MOVEMENT
        if len(text) == 0:
            try:
                shoulderL = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
                shoulderR = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
            except:
                pass

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # print("Text before drawing:", text)
        cv2.putText(image, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

        cv2.imshow('MediaPipe Holistic', image)

        img_ind += 1

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()