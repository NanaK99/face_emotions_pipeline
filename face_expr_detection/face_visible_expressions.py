from configparser import ConfigParser
import mediapipe as mp
import argparse
import cv2


from scipy.spatial import distance as dist

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

config_object = ConfigParser()
config_object.read("./static/config.ini")

expressions_section = config_object["EXPRESSIONS"]
FACE_TILT = float(expressions_section["FACE_TILT"])
EYE_OPEN_HEIGHT = float(expressions_section["EYE_OPEN_HEIGHT"])
BROWS_RAISE = float(expressions_section["BROWS_RAISE"])
BROW_RAISE_LEFT = float(expressions_section["BROW_RAISE_LEFT"])
BROW_RAISE_RIGHT = float(expressions_section["BROW_RAISE_RIGHT"])
FURROWED_BROWS = float(expressions_section["FURROWED_BROWS"])


def get_aspect_ratio(top, bottom, right, left):
    height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
    width = dist.euclidean([right.x, right.y], [left.x, left.y])
    return height / width


def get_face_expression(image):
        results = face_mesh.process(image)
        image.flags.writeable = True

        text = ""

        if results.multi_face_landmarks is None:
            return text

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            face_landmarks = results.multi_face_landmarks[0]
            face = face_landmarks.landmark

            face_mid_right = face[234]
            face_mid_left = face[454]
            face_mid_top = face[10]
            face_mid_bottom = face[152]
            cheek_mid_right = face[50]
            cheek_mid_left = face[280]

            if cheek_mid_right.x < face_mid_right.x:
                text = "HEAD TURN RIGHT"

            elif cheek_mid_left.x > face_mid_left.x:
                text = "HEAD TURN LEFT"

            face_angle = (face_mid_top.x - face_mid_bottom.x) / (
                    face_mid_right.x - face_mid_left.x)

            if len(text) == 0:
                if face_angle > FACE_TILT:
                    text = "HEAD TILT RIGHT"
                    return text
                elif face_angle < -FACE_TILT:
                    text = "HEAD TILT LEFT"
                    return text
            else:
                return text

            # Narrowed eyes and furrowed brows
            eyeR_top = face[159]
            eyeR_bottom = face[145]
            eyeR_inner = face[133]
            eyeR_outer = face[33]
            eyeR_ar = get_aspect_ratio(eyeR_top, eyeR_bottom, eyeR_outer, eyeR_inner)

            eyeL_top = face[386]
            eyeL_bottom = face[374]
            eyeL_inner = face[362]
            eyeL_outer = face[263]
            eyeL_ar = get_aspect_ratio(eyeL_top, eyeL_bottom, eyeL_outer, eyeL_inner)
            eyeA_ar = (eyeR_ar + eyeL_ar) / 2

            browL_inner_bottom = face[285]
            browR_inner_bottom = face[55]
            eyeR_inner = face[133]
            eyeL_inner = face[362]
            brow_eyeR_inner_dist = dist.euclidean([browR_inner_bottom.x, browR_inner_bottom.y],
                                                  [eyeR_inner.x, eyeR_inner.y])
            brow_eyeL_inner_dist = dist.euclidean([browL_inner_bottom.x, browL_inner_bottom.y],
                                                  [eyeL_inner.x, eyeL_inner.y])

            if eyeR_ar < EYE_OPEN_HEIGHT and eyeL_ar < EYE_OPEN_HEIGHT:
                text = "NARROWED EYES"
                return text

            if brow_eyeL_inner_dist < FURROWED_BROWS and brow_eyeR_inner_dist < FURROWED_BROWS:
                if eyeA_ar > 0.15:
                    text = "FURROWED BROWS"
                    return text
                else:
                    text = "NARROWED EYES"
                    return text

            # Brows, widened eyes and tense
            browR_top = face[52]
            browR_bottom = face[223]
            browR_eyeR_lower_dist = dist.euclidean([browR_bottom.x, browR_bottom.y],
                                                   [eyeR_top.x, eyeR_top.y])
            browR_eyeR_upper_dist = dist.euclidean([browR_top.x, browR_top.y],
                                                   [eyeR_top.x, eyeR_top.y])
            browR_eyeR_dist = (browR_eyeR_lower_dist + browR_eyeR_upper_dist) / 2

            browL_top = face[443]
            browL_bottom = face[257]
            browL_eyeL_lower_dist = dist.euclidean([browL_bottom.x, browL_bottom.y],
                                                   [eyeL_top.x, eyeL_top.y])
            browL_eyeL_upper_dist = dist.euclidean([browL_top.x, browL_top.y],
                                                   [eyeL_top.x, eyeL_top.y])
            browL_eyeL_dist = (browL_eyeL_lower_dist + browL_eyeL_upper_dist) / 2

            brows_avg_raise = (browR_eyeR_dist + browL_eyeL_dist) / (
                    face_mid_bottom.y - face_mid_top.y)

            if eyeA_ar > 0.7:
                if brows_avg_raise > BROWS_RAISE:
                    text = "WIDENED EYES"
                    return text
                else:
                    text = "TENSE"
                    return text
            else:
                if brows_avg_raise > BROWS_RAISE and eyeA_ar > EYE_OPEN_HEIGHT:
                    text = "BROWS RAISED"
                    return text

            brows_relative_raise = browR_eyeR_dist - browL_eyeL_dist

            if brows_relative_raise < BROW_RAISE_LEFT:
                text = "LEFT BROW RAISED"
                return text
            elif brows_relative_raise > BROW_RAISE_RIGHT:
                text = "RIGHT BROW RAISED"
                return text

            # Smile
            mouth_outer_right = face[76]
            mouth_outer_left = face[206]

            mouth_face_dist_ratio = get_aspect_ratio(mouth_outer_left, mouth_outer_right,
                                                     face_mid_right, face_mid_left)

            if mouth_face_dist_ratio < 0.21:
                text = "SMILING"
                return text

            if len(text) == 0:
                text = ""
                return text


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--start', metavar='N', type=int,
                        help='an integer for the accumulator')
    parser.add_argument('--video', metavar='N', type=str,
                        help='an integer for the accumulator')
    parser.add_argument('--anot', choices=('True', 'False'))

    args = parser.parse_args()
    start = args.start
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video file")
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            face_expr = get_face_expression(frame)

            if face_expr is not None:
                # mov = face_mov

                if args.anot == 'True':
                    cv2.putText(frame, face_expr, (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Frame', frame)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            print("error")
            break
    cap.release()