import cv2
import mediapipe as mp
from scipy.spatial import distance as dist


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def get_aspect_ratio(top, bottom, right, left):
  height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
  width = dist.euclidean([right.x, right.y], [left.x, left.y])
  return height / width


def get_body_movement(image):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) \
            as holistic:

        img_ind = 0
        turn = ""
        nod = ""

        # Process the image and detect the holistic
        results = holistic.process(image)
        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        landmarks = results.pose_landmarks.landmark

        # LEAN IN and LEAN OUT
        try:
            eyeR_outer = landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER.value]
            eyeL_outer = landmarks[mp_holistic.PoseLandmark.LEFT_EYE_OUTER.value]
            eyeR_inner = landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_INNER.value]
            eyeL_inner = landmarks[mp_holistic.PoseLandmark.LEFT_EYE_INNER.value]
            eyeR_len = dist.euclidean([eyeR_inner.x, eyeR_inner.y], [eyeR_outer.x, eyeR_outer.y])
            eyeL_len = dist.euclidean([eyeL_inner.x, eyeL_inner.y], [eyeL_outer.x, eyeL_outer.y])
            if eyeR_len > 0.06 and eyeL_len > 0.06:
                text = ", LEAN IN, "
            elif eyeR_len < 0.05 and eyeL_len < 0.05:
                text = ", LEAN OUT, "
            else:
                text = ""

            return text

        except:
            pass

        # HEAD SHAKE
        try:
            earL = landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value]
            earR = landmarks[mp_holistic.PoseLandmark.RIGHT_EAR.value]
            eyeR_outer = landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER.value]
            eyeL_outer = landmarks[mp_holistic.PoseLandmark.LEFT_EYE_OUTER.value]
            right_to_left_ratio = get_aspect_ratio(earR, eyeR_outer, earL, eyeL_outer)

            if right_to_left_ratio > 1.3:
                turn = "right"

            elif right_to_left_ratio < 0.7:
                turn = "left"

            return turn, img_ind

        except:
            pass

        #HEAD NOD
        try:
            eyeR_inner = landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_INNER.value]
            eyeL_inner = landmarks[mp_holistic.PoseLandmark.LEFT_EYE_INNER.value]
            nodR_inner_ratio = 1 / eyeR_inner.z
            nodL_inner_ratio = 1 / eyeL_inner.z

            if abs(nodL_inner_ratio) > 0.33 and abs(nodR_inner_ratio) > 0.33:
                nod = "down"

            elif abs(nodR_inner_ratio) < 0.27 and abs(nodR_inner_ratio) < 0.27:
                nod = "up"

            return nod, img_ind

        except:
            pass

        # SHOULDER MOVEMENT
        try:
            shoulderL = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
            shoulderR = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        except:
            pass

        return text
