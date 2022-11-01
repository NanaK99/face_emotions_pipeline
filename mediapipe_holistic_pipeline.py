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

        text = ""

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
            if eyeR_inner.z < -3 and eyeR_outer.z < -3 and eyeL_inner.z < -3 and eyeL_outer.z < -3:
                text = "LEAN IN, "
            elif eyeR_inner.z > -1 and eyeR_outer.z > -1 and eyeL_inner.z > -1 and eyeL_outer.z > -1:
                text = "LEAN OUT, "

            if len(text) != 0:
                return text

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

                if right_to_left_ratio < 0.9:
                    text = "right"
                else:
                    text = "left"

                # return turn
                if len(text) != 0:
                    return text

            except:
                pass

        else:
            return text

        #HEAD NOD
        try:
            eyeR_inner = landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_INNER.value]
            eyeL_inner = landmarks[mp_holistic.PoseLandmark.LEFT_EYE_INNER.value]
            nodR_inner_ratio = 1 / eyeR_inner.z
            nodL_inner_ratio = 1 / eyeL_inner.z

            if nodL_inner_ratio < -0.6 and nodR_inner_ratio < -0.6:
                text = "up"
            else:
                text = "down"

            if len(text) != 0:
                return text

        except:
            pass

        # SHOULDER MOVEMENT
        try:
            shoulderL = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
            shoulderR = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        except:
            pass

        # print("text before returning", text)
        if len(text) == 0:
            return "NO BODY MOVEMENT, "
        else:
            return text
