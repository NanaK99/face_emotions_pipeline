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

    img_ind = 0
    turn = False
    shake = False
    pre_shake = False

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

        # LEAN IN and LEAN OUT
        # try:
        #     landmarks = results.pose_landmarks.landmark
        #     nose = landmarks[mp_holistic.PoseLandmark.NOSE.value]
        #     nose_y_ratio = 1 / nose.y
        #     nose_z_ratio = 1 / nose.z
        #     if abs(nose_y_ratio) < 1.5 or abs(nose_z_ratio) < 0.35:
        #         text = "LEAN IN"
        #     elif abs(nose_y_ratio) > 2 or abs(nose_z_ratio) > 1:
        #         text = "LEAN OUT"
        #     # else:
        #     #     text = "normal posture"
        #     else:
        #         text = ""
        # except:
        #     pass

        # HEAD SHAKE
        try:
            landmarks = results.pose_landmarks.landmark
            earL = landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value]
            earR = landmarks[mp_holistic.PoseLandmark.RIGHT_EAR.value]
            # nose = landmarks[mp_holistic.PoseLandmark.NOSE.value]
            eyeR_outer = landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER.value]
            eyeL_outer = landmarks[mp_holistic.PoseLandmark.LEFT_EYE_OUTER.value]
            right_to_left_ratio = get_aspect_ratio(earR, eyeR_outer, earL, eyeL_outer)
            # print("LEFT EAR", earL)
            # print("RIGHT EAR", earR)
            # print("NOSE", nose)
            # print("RATIO", right_to_left_ratio)
            mouthR = landmarks[mp_holistic.PoseLandmark.MOUTH_RIGHT.value]
            mouthL = landmarks[mp_holistic.PoseLandmark.MOUTH_LEFT.value]
            # print("mouth RIGHT", mouthR)
            # print("mouth LEFT", mouthL)
            mouth_len = dist.euclidean([mouthR.x, mouthR.y], [mouthL.x, mouthL.y])
            # print("MOUTH RATIO", mouth_len)

            current = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #     if mouth_len < 0.06:
        #         if not turn:
        #             if right_to_left_ratio > 1.3:
        #                 turn = True
        #                 turn_dir = "right"
        #                 print("turned to right")
        #                 curr_img_ind = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #
        #             elif right_to_left_ratio < 0.7:
        #                 print("turned to left")
        #                 turn = True
        #                 turn_dir = "left"
        #                 curr_img_ind = img_ind
        #
        #         if not pre_shake:
        #             if turn:
        #                 if turn_dir == "left":
        #                     if right_to_left_ratio > 1.3:
        #                         if img_ind < curr_img_ind + 9:
        #                             print("111111")
        #                             pre_shake = True
        #                             turn_dir = "right"
        #                             curr_img_ind = img_ind
        #                         else:
        #                             print("222222")
        #
        #                             pre_shake = False
        #                             curr_img_ind = img_ind
        #                             turn = False
        #
        #                 elif turn_dir == "right":
        #                     if right_to_left_ratio < 0.7:
        #                         if img_ind < curr_img_ind + 9:
        #                             print("33333")
        #
        #                             pre_shake = True
        #                             turn_dir = "left"
        #                             curr_img_ind = img_ind
        #                         else:
        #                             print("4444")
        #
        #                             pre_shake = False
        #                             curr_img_ind = img_ind
        #                             turn = False
        #
        #         if pre_shake:
        #             if turn_dir == "left":
        #                 if right_to_left_ratio > 1.3:
        #                     if img_ind < curr_img_ind + 9:
        #                         print("55555")
        #
        #                         shake = True
        #                     else:
        #                         print("66666")
        #
        #                         pre_shake = False
        #                         curr_img_ind = img_ind
        #                         turn = False
        #
        #             elif turn_dir == "right":
        #                 if right_to_left_ratio < 0.7:
        #                     if img_ind < curr_img_ind + 9:
        #                         print("77777")
        #
        #                         shake = True
        #                     else:
        #                         print("8888")
        #
        #                         pre_shake = False
        #                         curr_img_ind = img_ind
        #                         turn = False
        #
        #         if shake:
        #             text = "HEAD SHAKE"
        #             print(text)
        #             shake = False
        #             turn = False
        #             pre_shake = False
        #             curr_img_ind - img_ind
        #
        except:
            pass
        # mp_drawing.draw_landmarks(
        #     image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # FACEMESH_CONTOURS FACEMESH_TESSELATION
        # mp_drawing.draw_landmarks(
        #     image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(
        #     image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        print("text before drawing", text)
        cv2.putText(image, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

        cv2.imshow('MediaPipe Holistic', image)

        img_ind += 1  # print("HEAD SHAKE")

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()