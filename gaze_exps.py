import cv2
import numpy as np
from helpers import relative, relativeT
from scipy.spatial import distance as dist
from math import atan


def gaze(frame, points):
    """
    The gaze function gets an image and face landmarks from mediapipe framework.
    The function draws the gaze direction into the frame.
    """

    '''
    2D image points.
    relative takes mediapipe points that is normalized to [-1, 1] and returns image points
    at (x,y) format
    '''
    image_points = np.array([
        relative(points.landmark[4], frame.shape),  # Nose tip
        relative(points.landmark[152], frame.shape),  # Chin
        relative(points.landmark[263], frame.shape),  # Left eye left corner
        relative(points.landmark[33], frame.shape),  # Right eye right corner
        relative(points.landmark[287], frame.shape),  # Left Mouth corner
        relative(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    '''
    2D image points.
    relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points
    at (x,y,0) format
    '''
    image_points1 = np.array([
        relativeT(points.landmark[4], frame.shape),  # Nose tip
        relativeT(points.landmark[152], frame.shape),  # Chin
        relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
        relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
        relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
        relativeT(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26),  # Left eye, left corner
        (43.3, 32.7, -26),  # Right eye, right corner
        (-28.9, -28.9, -24.1),  # Left Mouth corner
        (28.9, -28.9, -24.1)  # Right mouth corner
    ])

    '''
    3D model eye points
    The center of the eye ball
    '''
    Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
    Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.

    '''
    camera matrix estimation
    '''
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # from 3d to 2d
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # 2d pupil location
    left_pupil = relative(points.landmark[468], frame.shape)
    # print("LEFT PUPIL", left_pupil, len(left_pupil))
    right_pupil = relative(points.landmark[473], frame.shape)

    # Transformation between image point to world point
    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

    if transformation is not None:  # if estimateAffine3D secsseded
        # project pupil image point into 3d world point
        # print("TRANSFORMATION",transformation, transformation.shape)
        pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
        # print("pupil world cord", pupil_world_cord, pupil_world_cord.shape)

        # 3D gaze point (10 is arbitrary value denoting gaze distance)
        S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10
        # print("S", S, S.shape)

        # Project a 3D gaze direction onto the image plane.
        (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)

        # print("eye pupil2D",eye_pupil2D, eye_pupil2D.shape)\

        # project 3D head pose into the image plane
        (head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                           rotation_vector,
                                           translation_vector, camera_matrix, dist_coeffs)

        # print("head pose", head_pose, head_pose.shape)

        # correct gaze for head rotation
        gaze = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)
        print("gaze",gaze)

        # print("GAZE", gaze, gaze.shape)

        # Draw gaze line into screen
        p1 = (int(left_pupil[0]), int(left_pupil[1]))
        p2 = (int(gaze[0]), int(gaze[1]))

        p3 = (150, 400)
        p4 = (550, 400)

        try:
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])


            # Function to find the
            # angle between two lines
            def findAngle(M1, M2):
                PI = 3.14159265

                # Store the tan value  of the angle
                angle = abs((M2 - M1) / (1 + M1 * M2))

                # Calculate tan inverse of the angle
                ret = atan(angle)

                # Convert the angle from
                # radian to degree
                val = (ret * 180) / PI

                # Print the result
                return round(val, 4)

            angle = findAngle(m1, m2)
            print("###", angle)
            print("dist", dist.euclidean([p1[0], p1[1]], [p2[0], p2[1]]))

            if angle < 45:
                if gaze[0] < 160 and gaze[1] > 190:
                    print("DOWN RIGHT")
                if gaze[0] < 160 and gaze[1] < 190:
                    print("UP RIGHT")
                if gaze[0] > 340 and gaze[1] > 200:
                    print("DOWN LEFT")
                if gaze[0] > 340 and gaze[1] < 200:
                    print("UP LEFT")
                if gaze[0] < 270 and gaze[0] > 200 and gaze[1] < 200:
                    print("CENTRE")
                if gaze[0] < 280 and gaze[0] > 210 and gaze[1] > 230:
                    print("DOWN CENTRE")
            else:
                # if gaze[1] > 230:
                #     print("DOWN CENTRE")
                # else:
                if angle > 60:
                    print("CENTRE")
                else:
                    print("UP CENTRE")

        except:
            pass

        # print(p1, p2)
        cv2.line(frame, p1, p2, (0, 0, 255), 2)
        # ADDED
        # if gaze[0] < 240:
        #     text = "UP CENTRE"
        # elif gaze[0] > 250:
        #     text = "DOWN CENTRE"
        # else:
        #     text = "CENTRE"
        # print(text)
        # return text
        # cv2.putText(image, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)


#         TRANSFORMATION[[-5.74923741e-01  2.84377414e-02  0.00000000e+00  1.73797105e+02]
#         [-1.61846005e-02 - 5.54015845e-01
#         0.00000000e+00
#         1.57702598e+02]
#         [4.99200237e-04  1.70881475e-02  0.00000000e+00 - 2.98555988e+01]] (3, 4)
#
#         pupilworldcord[[29.88456129]
#         [33.81099475]
#         [-26.03426766]](3, 1)
#
#         S[[37.39561289]
#         [43.80994747]
#         [95.15732336]](3, 1)
#
#         eyepupil2D[[[236.69821264 139.73780116]]](1, 1, 2)
#         head pose[[[260.89287985 194.4485064]]](1, 1, 2)
#         GAZE[236.8053327 161.28929476] (2,)
#
#
#         TRANSFORMATION [[-5.72675082e-01  2.11959336e-02  0.00000000e+00  1.72305055e+02]
#  [-1.21926414e-02 -5.69799282e-01  0.00000000e+00  1.69230402e+02]
#  [ 3.76071654e-04  1.75749742e-02  0.00000000e+00 -3.02111641e+01]] (3, 4)
# pupil world cord [[ 30.61769069]
#  [ 33.9156527 ]
#  [-26.03749575]] (3, 1)
# S [[44.7269069 ]
#  [44.85652702]
#  [95.1250425 ]] (3, 1)
# eye pupil2D [[[220.20376195 166.58403205]]] (1, 1, 2)
# head pose [[[257.78957135 215.52422821]]] (1, 1, 2)
# GAZE [218.4141906  183.05980384] (2,)

