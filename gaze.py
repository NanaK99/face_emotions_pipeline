import cv2
import numpy as np
from helpers import relative, relativeT
from scipy.spatial import distance as dist
from math import atan


def findAngle(M1, M2):
    PI = 3.14159265
    angle = abs((M2 - M1) / (1 + M1 * M2))
    ret = atan(angle)
    val = (ret * 180) / PI
    return round(val, 4)


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
    right_pupil = relative(points.landmark[473], frame.shape)

    # Transformation between image point to world point
    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

    if transformation is not None:  # if estimateAffine3D secsseded
        # project pupil image point into 3d world point
        pupil_world_cord_left = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
        pupil_world_cord_right = transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T

        # 3D gaze point (10 is arbitrary value denoting gaze distance)
        S_left = Eye_ball_center_left + (pupil_world_cord_left - Eye_ball_center_left) * 10
        S_right = Eye_ball_center_right + (pupil_world_cord_right - Eye_ball_center_right) * 10

        # Project a 3D gaze direction onto the image plane.
        (eye_pupil2D_left, _) = cv2.projectPoints((int(S_left[0]), int(S_left[1]), int(S_left[2])), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)
        (eye_pupil2D_right, _) = cv2.projectPoints((int(S_right[0]), int(S_right[1]), int(S_right[2])), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)

        # project 3D head pose into the image plane
        (head_pose_left, _) = cv2.projectPoints((int(pupil_world_cord_left[0]), int(pupil_world_cord_left[1]), int(40)),
                                           rotation_vector,
                                           translation_vector, camera_matrix, dist_coeffs)

        (head_pose_right, _) = cv2.projectPoints((int(pupil_world_cord_right[0]), int(pupil_world_cord_right[1]), int(40)),
                                           rotation_vector,
                                           translation_vector, camera_matrix, dist_coeffs)

        # correct gaze for head rotation
        gaze_left = left_pupil + (eye_pupil2D_left[0][0] - left_pupil) - (head_pose_left[0][0] - left_pupil)
        gaze_right = right_pupil + (eye_pupil2D_right[0][0] - right_pupil) - (head_pose_right[0][0] - right_pupil)

        gaze = (gaze_left + gaze_right) / 2

        # Draw gaze line into screen
        p1_left = (int(left_pupil[0]), int(left_pupil[1]))
        # p2_left = (int(gaze_left[0]), int(gaze_left[1]))

        p1_right = (int(right_pupil[0]), int(right_pupil[1]))
        # p2_right = (int(gaze_right[0]), int(gaze_right[1]))

        p2 = (int(gaze[0]), int(gaze[1]))

        p3 = (150, 400)
        p4 = (550, 400)

        try:
            m1 = (p2[1] - p1_left[1]) / (p2[0] - p1_left[0])
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])

            angle = findAngle(m1, m2)

            if angle < 45:
                if gaze[0] < 160 and gaze[1] > 190:
                    text = "DOWN RIGHT"
                if gaze[0] < 160 and gaze[1] < 190:
                    text = "UP RIGHT"
                if gaze[0] > 340 and gaze[1] > 200:
                    text = "DOWN LEFT"
                if gaze[0] > 340 and gaze[1] < 200:
                    text = "UP LEFT"
                if gaze[0] < 270 and gaze[0] > 200 and gaze[1] < 200:
                    text = "CENTRE"
                if gaze[0] < 280 and gaze[0] > 210 and gaze[1] > 230:
                    text = "DOWN CENTRE"
            else:
                if angle > 60:
                    text = "CENTRE"
                else:
                    text = "UP CENTRE"

        except:
            pass

        cv2.line(frame, p1_left, p2, (0, 0, 255), 2)
        cv2.line(frame, p1_right, p2, (0, 0, 255), 2)

        print("text", text)

        # return text


