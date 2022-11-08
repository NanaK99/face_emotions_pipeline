import face_visible_expressions
import gaze_estimator
import mediapipe_holistic
import cv2
from collections import Counter


CAMERA = 0
cap = cv2.VideoCapture(CAMERA)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID') #*'MJPG'
out = cv2.VideoWriter('output.avi', fourcc, 20.0, size)


font = cv2.FONT_HERSHEY_SIMPLEX

gaze_text = ""
faceexpr_text = ""

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
    gazee = gaze_estimator.get_gaze_direction(img)

    face_expr = face_visible_expressions.get_face_expression(img)
    if face_expr is not None:
        faceexpr_text += face_expr

    if gazee is not None:
        p1_left = gazee[0]
        p1_right = gazee[1]
        p2 = gazee[2]
        gaze_text = gazee[3]

        cv2.line(img, p1_left, p2, (0, 0, 255), 2)
        cv2.line(img, p1_right, p2, (0, 0, 255), 2)
    else:
        gaze_text = ""

    final_text = f"{gaze_text}, {faceexpr_text}"
    cv2.putText(img, final_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.imshow('Recording...', img)
    out.write(img)
    final_text = ""


    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
# ffmpeg -i /home/nana/Desktop/face_emotions_pipeline/output.avi -c:v copy -c:a copy -y /home/nana/Desktop/face_emotions_pipeline/output.mp4
