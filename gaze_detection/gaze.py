import cv2
import numpy as np
from gaze_detection.helpers import relative, relativeT
from math import atan

from configparser import ConfigParser

config_object = ConfigParser()
config_object.read("./static/config.ini")

gaze = config_object["EYE_GAZE"]
centre_upper_limit = int(gaze["CENTRE_UPPER_LIMIT"])
centre_lower_limit = int(gaze["CENTRE_LOWER_LIMIT"])
up = int(gaze["UP"])
down = int(gaze["DOWN"])

def compute_gaze_direction(left_eye, right_eye):
    """Computes the gaze direction vector from the relative coordinates of the eye landmarks."""
    left_eye = np.array([left_eye[0], left_eye[1]])
    right_eye = np.array([right_eye[0], right_eye[1]])
    eye_center = (left_eye + right_eye) / 2
    gaze_direction = (right_eye - left_eye) / np.linalg.norm(right_eye - left_eye)
    return eye_center, gaze_direction


def findAngle(M1, M2):
    PI = 3.14159265
    angle = abs((M2 - M1) / (1 + M1 * M2))
    ret = atan(angle)
    val = (ret * 180) / PI
    return round(val, 4)


import math

def get_vector_angle(v1, v2):
    """
    Calculate the dot product of the two vectors:
    F · V = Fx * Vx + Fy * Vy + Fz * Vz

    The dot product measures the projection of one vector onto the other.
    If the dot product is positive, then the angle between the two vectors
    is acute, and the second vector is pointing in the same general direction
    as the first vector. If the dot product is negative, then the angle between
    the two vectors is obtuse, and the second vector is pointing in the opposite direction.
    If the dot product is zero, then the two vectors are orthogonal, a
    nd the second vector is perpendicular to the first vector.
    """
    print(v1)
    print(v2)
    # print(v1.shape, v2.shape)
    BC = v1 - v2
    # v1 = v1.T
    #
    # # Compute the normal vector to the plane defined by AB and AC
    # normal = np.cross(v1, v2)
    #
    # # Compute the horizontal vector perpendicular to AB
    # horizontal = np.cross(normal, v1)
    #
    # # Compute the direction of the BC vector relative to the horizontal vector
    # cos_theta = np.dot(BC, horizontal) / (np.linalg.norm(BC) * np.linalg.norm(horizontal))
    #
    # # if cos_theta > 0:
    # #     print("The direction of BC is in the same direction as the horizontal vector")
    # # else:
    # #     print("The direction of BC is in the opposite direction of the horizontal vector")
    #
    # # Compute the magnitude of the BC vector
    magnitude = np.linalg.norm(BC)
    # return cos_theta, magnitude
    # print("The magnitude of the BC vector is:", magnitude)
    return magnitude


def get_gaze_direction(gaze):
    """
    Returns the gaze direction based on the gaze vector

    Args:
        gaze: a tuple representing the gaze vector, as (x, y)

    Returns:
        A string representing the gaze direction, one of:
            "center"
            "up"
            "down"
            "left"
            "right"
            "up left"
            "up right"
            "down left"
            "down right"
    """

    # calculate the angle of the gaze vector
    # anglex = math.atan2(gaze[1], gaze[0])
    # # print("x",anglex)
    # anglex = math.degrees(anglex)
    # # angle = angle % 360
    # print("x",anglex)
    # angley = math.atan2(gaze[0], gaze[1])
    # angley = math.degrees(angley)
    # angle = angle % 360
    # print("y",angley)
    # classify the angle into one of nine categories
    """
    center:36-42
    up center:
    down center: 
    """
    # if angle >= 337.5 or angle < 22.5:
    #     return "right"
    # elif angle >= 22.5 and angle < 67.5:
    #     return "up right"
    # elif angle >= 67.5 and angle < 112.5:
    #     return "up"
    # elif angle >= 112.5 and angle < 157.5:
    #     return "up left"
    # elif angle >= 157.5 and angle < 202.5:
    #     return "left"
    # elif angle >= 202.5 and angle < 247.5:
    #     return "down left"
    # elif angle >= 247.5 and angle < 292.5:
    #     return "down"
    # elif angle >= 292.5 and angle < 337.5:
    #     return "down right"
    # else:
    #     return "center"


def gaze(frame, points):
    text = ""
    """
    The gaze function gets an image and face landmarks from mediapipe framework.
    The function draws the gaze direction into the frame.
    """

    '''
    2D image points.
    relative takes mediapipe points that is normalized to [-1, 1] and returns image points
    at (x,y) format
    '''
    # print(points.landmark[4], frame.shape)
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

    # print("@@@@",image_points)
    # print("###",image_points1)
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
        # print(pupil_world_cord_left, Eye_ball_center_left)

        # 3D gaze point (10 is arbitrary value denoting gaze distance)
        S_left = Eye_ball_center_left + (pupil_world_cord_left - Eye_ball_center_left) * 20
        S_right = Eye_ball_center_right + (pupil_world_cord_right - Eye_ball_center_right) * 20

        # Project a 3D gaze direction onto the image plane.
        (eye_pupil2D_left, _) = cv2.projectPoints((int(S_left[0]), int(S_left[1]), int(S_left[2])), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)
        (eye_pupil2D_right, _) = cv2.projectPoints((int(S_right[0]), int(S_right[1]), int(S_right[2])), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)
        # (exp, _) = cv2.projectPoints((0, 0, 0), rotation_vector,
        #                                            translation_vector, camera_matrix, dist_coeffs)

        # normal vector for the image plane
        # print(eye_pupil2D_right[0][0], type(eye_pupil2D_right[0][0]), eye_pupil2D_right[0][0].shape)
        # eye_pupil2D_right[0] += 5

        # v1 = eye_pupil2D_right[0][0] - exp[0][0]
        # v1 = np.append(v1, 0)
        #
        # v2 = eye_pupil2D_left[0][0] - exp[0][0]
        # v2 = np.append(v2, 0)
        #
        # # print(v2)
        # n = np.cross(v1, v2, axis=0)
        # # print(n)
        # n = n / np.linalg.norm(n)
        # # print("normal vector", n)

        # normal vector for the other plane
        # zero_vector = np.zeros((3, 1))
        # v_1 = S_right - zero_vector
        # v_2 = S_left - zero_vector
        # n_1 = np.cross(v_1, v_2, axis=0)
        # n_1 = n_1 / np.linalg.norm(n_1)
        # # print("##normal vector", n_1)
        #
        # angle = np.arccos(np.dot(n, n_1) / (np.linalg.norm(n) * np.linalg.norm(n_1)))
        # angle_degrees = np.degrees(angle)
        # # print(angle_degrees[0])
        # centroid = (S_right + S_left) / 2

        # Draw the plane on the fram

        # drawing the plane
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

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

        p1_left = (int(left_pupil[0]), int(left_pupil[1]))
        p1_right = (int(right_pupil[0]), int(right_pupil[1]))
        p2 = (int(gaze[0]), int(gaze[1]))

        # direction = get_gaze_direction(p2)
        # print(direction)

        # right = get_vector_angle(Eye_ball_center_right, pupil_world_cord_right)
        # left = get_vector_angle(Eye_ball_center_left, pupil_world_cord_left)
        # print("RIGHT", right)
        # print("LEFT", left)

        # normal vector for the other plane
        # print(p2)
        # v_1 = S_right - p2
        # v_2 = S_left - p2
        # print(v_1.shape, v_2.shape)
        # n_1 = np.cross(v_1, v_2, axis=0)
        # n_1 = n_1 / np.linalg.norm(n_1)
        # print("##normal vector", n_1)



        # p2x = p2[0] * 100
        # p2y = p2[1] * 100
        # gg = compute_gaze_direction(p1_left, p1_right)
        # screen_center = np.array([0, 6, 0])
        # dot_product = np.dot(np.append(gg[1], 0), np.array([0, -1, 0]))
        # theta = np.arccos(dot_product)
        # dx = 6 * np.tan(theta)
        # phi = np.arctan2(gg[1][1], gg[1][0])
        # dy = (50 / 2) * np.tan(phi)
        # look_at = screen_center + np.array([dx, dy, 0])
        # print(look_at)
        #
        # # print(gg[0], gg[1])
        # ggx = gg[1][0] * 100
        # ggy = gg[1][1] * 100

        # print(ggx, ggy)



        ### FOR MAC WEBCAM
        # try:
        #     if gaze[0] > 800:
        #         if gaze[1] < 300:
        #             text = "UP LEFT"
        #         elif gaze[1] > 300:
        #             text = "DOWN LEFT"
        #     elif gaze[0] < 570:
        #         if gaze[1] < 200:
        #             text = "UP RIGHT"
        #         elif gaze[1] > 250:
        #             text = "DOWN RIGHT"
        #     else:
        #         if gaze[1] < 200:
        #             text = "UP CENTRE"
        #         elif gaze[1] > 275 and gaze[0] < 665:
        #             text = "CENTRE"
        #         else:
        #             text = "DOWN CENTRE"
        # except:
        #     pass

        # the lastest experiment
        ### FOR Trott_Garner_Miranda_LG_2022_-_Audio2_new video
        try:
            if gaze[0] > centre_lower_limit and gaze[0] < centre_upper_limit:
                if gaze[1] < up:
                    text = "UP CENTRE"
                elif gaze[1] > down:
                    text = "DOWN CENTRE"
                else:
                    text = "STRAIGHT CENTRE"
            elif gaze[0] < centre_lower_limit:
                if gaze[1] < up:
                    text = "UP RIGHT"
                elif gaze[1] > down:
                    text = "DOWN RIGHT"
                else:
                    text = "CENTRE RIGHT"
            elif gaze[0] > centre_upper_limit:
                if gaze[1] > up:
                    text = "UP LEFT"
                elif gaze[1] > down:
                    text = "DOWN LEFT"
                else:
                    text = "CENTRE LEFT"

        except:
            pass
        # print(text)


            # if gaze[0] > centre_lower_limit and gaze[0] < centre_upper_limit:
            #     if gaze[1] < up:
            #         text = "UP CENTRE"
            #     elif gaze[1] > down:
            #         text = "DOWN CENTRE"
            #     else:
            #         text = "STRAIGHT CENTRE"
            # elif gaze[0] < centre_lower_limit:
            #     if gaze[1] < up:
            #         text = "UP RIGHT"
            #     elif gaze[1] > down:
            #         text = "DOWN RIGHT"
            #     else:
            #         text = "CENTRE RIGHT"
            # elif gaze[0] > centre_upper_limit:
            #     if gaze[1] > up:
            #         text = "UP LEFT"
            #     elif gaze[1] > down:
            #         text = "DOWN LEFT"
            #     else:
            #         text = "CENTRE LEFT"


        if len(text) != 0:
            return p1_left, p1_right, p2, text
        else:
            return p1_left, p1_right, p2, ""