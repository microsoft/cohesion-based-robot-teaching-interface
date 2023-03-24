import cv2
import mediapipe as mp
import numpy as np
import os
from . import laban_orientations

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_maximum = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_maximum, x_middle + plot_maximum])
    ax.set_ylim3d([y_middle - plot_maximum, y_middle + plot_maximum])
    ax.set_zlim3d([z_middle - plot_maximum, z_middle + plot_maximum])


def get_laban(vec_arm):
    # Calculate Laban
    laban_dic = laban_orientations.labnotation_dictionary()
    norm = -np.inf
    keys = laban_dic.keys()
    laban = None
    for key in keys:
        if np.dot(vec_arm, np.array(laban_dic[key])) > norm:
            norm = np.dot(vec_arm, laban_dic[key])
            laban = key
    return laban


def transform_to_laban(fp_image, out_dir, vizualize=False):
    # For static images:
    with mp_pose.Pose(static_image_mode=True,
                      model_complexity=1,
                      min_detection_confidence=0.5) as pose:
        image = cv2.imread(fp_image)
        image_basename = os.path.basename(fp_image)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return None

        # Draw pose landmarks on the image.
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite(os.path.join(out_dir, 'pose_' +
                    image_basename), annotated_image)

        left_wrist_position = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width,
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height,
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z * image_width])
        left_elbow_position = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width,
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height,
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z * image_width])
        left_shoulder_position = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width,
                                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height,
                                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z * image_width])
        left_hip_position = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width,
                                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height,
                                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z * image_width])
        right_shoulder_position = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width,
                                            results.pose_landmarks.landmark[
                                                mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height,
                                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z * image_width])
        right_elbow_position = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width,
                                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height,
                                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z * image_width])
        right_wrist_position = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width,
                                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height,
                                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z * image_width])
        right_hip_position = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width,
                                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height,
                                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z * image_width])

        # origin is middle of left and right hip
        origin_position = (left_hip_position + right_hip_position) / 2
        # offset all the points to the origin
        left_wrist_position -= origin_position
        left_elbow_position -= origin_position
        left_shoulder_position -= origin_position
        left_hip_position -= origin_position
        right_shoulder_position -= origin_position
        right_elbow_position -= origin_position
        right_wrist_position -= origin_position
        right_hip_position -= origin_position

        # front orientation is a vector perpendicular to the plane formed by
        # the left and right hip and left and right shoulder
        front_orientation = np.cross(
            left_shoulder_position - origin_position,
            right_shoulder_position - origin_position)
        front_orientation = front_orientation / \
            np.linalg.norm(front_orientation)
        right_orientation = right_shoulder_position - left_shoulder_position
        right_orientation = right_orientation / \
            np.linalg.norm(right_orientation)
        # up_orientation = np.cross(right_orientation, front_orientation)
        up_orientation = np.array([0, -1, 0])
        front_orientation = np.cross(up_orientation, right_orientation)

        # normalize the orientation vectors
        front_orientation = (
            front_orientation /
            np.linalg.norm(front_orientation))
        right_orientation = (
            right_orientation /
            np.linalg.norm(right_orientation))
        up_orientation = (up_orientation / np.linalg.norm(up_orientation))

        # rotate the points to the front orientation
        rotation_matrix = np.array(
            [right_orientation, up_orientation, -front_orientation])

        # rotate the points to the front orientation
        left_wrist_position = np.dot(rotation_matrix, left_wrist_position)
        left_elbow_position = np.dot(rotation_matrix, left_elbow_position)
        left_shoulder_position = np.dot(
            rotation_matrix, left_shoulder_position)
        left_hip_position = np.dot(rotation_matrix, left_hip_position)
        right_shoulder_position = np.dot(
            rotation_matrix, right_shoulder_position)
        right_elbow_position = np.dot(rotation_matrix, right_elbow_position)
        right_wrist_position = np.dot(rotation_matrix, right_wrist_position)
        right_hip_position = np.dot(rotation_matrix, right_hip_position)

        # print those positions in 3D space
        if vizualize:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.scatter(
                left_wrist_position[0],
                left_wrist_position[1],
                left_wrist_position[2],
                c='r',
                marker='o')
            ax.scatter(
                left_elbow_position[0],
                left_elbow_position[1],
                left_elbow_position[2],
                c='r',
                marker='o')
            ax.scatter(
                left_shoulder_position[0],
                left_shoulder_position[1],
                left_shoulder_position[2],
                c='r',
                marker='o')
            ax.scatter(
                left_hip_position[0],
                left_hip_position[1],
                left_hip_position[2],
                c='r',
                marker='o')
            ax.scatter(
                right_shoulder_position[0],
                right_shoulder_position[1],
                right_shoulder_position[2],
                c='b',
                marker='o')
            ax.scatter(
                right_elbow_position[0],
                right_elbow_position[1],
                right_elbow_position[2],
                c='b',
                marker='o')
            ax.scatter(
                right_wrist_position[0],
                right_wrist_position[1],
                right_wrist_position[2],
                c='b',
                marker='o')
            ax.scatter(
                right_hip_position[0],
                right_hip_position[1],
                right_hip_position[2],
                c='b',
                marker='o')

            # plot lines connecting the points
            ax.plot([left_wrist_position[0],
                    left_elbow_position[0]],
                    [left_wrist_position[1],
                    left_elbow_position[1]],
                    [left_wrist_position[2],
                    left_elbow_position[2]],
                    c='r')
            ax.plot([left_elbow_position[0],
                    left_shoulder_position[0]],
                    [left_elbow_position[1],
                    left_shoulder_position[1]],
                    [left_elbow_position[2],
                    left_shoulder_position[2]],
                    c='r')
            ax.plot([right_shoulder_position[0],
                    right_elbow_position[0]],
                    [right_shoulder_position[1],
                    right_elbow_position[1]],
                    [right_shoulder_position[2],
                    right_elbow_position[2]],
                    c='b')
            ax.plot([right_elbow_position[0],
                    right_wrist_position[0]],
                    [right_elbow_position[1],
                    right_wrist_position[1]],
                    [right_elbow_position[2],
                    right_wrist_position[2]],
                    c='b')
            ax.plot([left_shoulder_position[0],
                    right_shoulder_position[0]],
                    [left_shoulder_position[1],
                    right_shoulder_position[1]],
                    [left_shoulder_position[2],
                    right_shoulder_position[2]],
                    c='k')
            ax.plot([left_hip_position[0],
                    right_hip_position[0]],
                    [left_hip_position[1],
                    right_hip_position[1]],
                    [left_hip_position[2],
                    right_hip_position[2]],
                    c='k')
            ax.plot([left_hip_position[0],
                    left_shoulder_position[0]],
                    [left_hip_position[1],
                    left_shoulder_position[1]],
                    [left_hip_position[2],
                    left_shoulder_position[2]],
                    c='k')
            ax.plot([right_hip_position[0],
                    right_shoulder_position[0]],
                    [right_hip_position[1],
                    right_shoulder_position[1]],
                    [right_hip_position[2],
                    right_shoulder_position[2]],
                    c='k')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            set_axes_equal(ax)
            plt.show()
        # arm orientation
        vec_right_forearm = right_wrist_position - right_elbow_position
        vec_right_upper_arm = right_elbow_position - right_shoulder_position
        vec_left_forearm = left_wrist_position - left_elbow_position
        vec_left_upper_arm = left_elbow_position - left_shoulder_position

        return {'right_forearm': get_laban(vec_right_forearm),
                'right_upper_arm': get_laban(vec_right_upper_arm),
                'left_forearm': get_laban(vec_left_forearm),
                'left_upper_arm': get_laban(vec_left_upper_arm)}


if __name__ == '__main__':
    path = 'PATH_TO_MP4'
    transform_to_laban(path, '.')
