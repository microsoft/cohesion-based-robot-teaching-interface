import numpy as np
import cv2
import cv2.aruco as aruco
import os
import json


def estimate_homo_transform_matrix(img_src):
    img = img_src.copy()
    fp_cameraparam = os.path.join(
        os.getcwd(), 'settings', 'marker_setting.json')
    with open(fp_cameraparam) as f:
        data = json.load(f)
    _camera_matrix = np.array(data['camera_K']).reshape([3, 3])
    # modify camera matrix according to the image size
    H, W = img.shape[:2]
    scale = W / float(data["width"])
    _camera_matrix[0, 0] = _camera_matrix[0, 0] * scale
    _camera_matrix[0, 2] = _camera_matrix[0, 2] * scale
    _camera_matrix[1, 1] = _camera_matrix[1, 1] * scale
    _camera_matrix[1, 2] = _camera_matrix[1, 2] * scale
    _dist_coef = np.array(data['camera_D'])
    key = getattr(aruco, data['marker_type'])
    marker_size = data['marker_size']
    marker_id = data['marker_id']

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    arucodict = cv2.aruco.getPredefinedDictionary(key)
    param =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucodict, param)
    bboxs, ids, _ = detector.detectMarkers(gray)
    # OLD API before 4.7.0
    #varucodict = aruco.Dictionary_get(key)
    #param = aruco.DetectorParameters_create()
    #bboxs, ids, _ = aruco.detectMarkers(gray, arucodict, parameters=param)
    aruco.drawDetectedMarkers(img, bboxs)
    if len(bboxs) != 0:
        for bbox, id in zip(bboxs, ids):
            if int(id) == marker_id:
                # pose estimation
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    bbox, marker_size, _camera_matrix, _dist_coef)
                cv2.drawFrameAxes(img, _camera_matrix,
                                  _dist_coef, rvecs, tvecs, 0.05)
                R, _ = cv2.Rodrigues(rvecs[0])
                T = np.array(tvecs[0][0])
                rot_mat_4x4 = np.zeros((4, 4))
                rot_mat_4x4[:3, :3] = R
                rot_mat_4x4[:3, 3] = T
                rot_mat_4x4[3, 3] = 1
                # inverse
                rot_mat_4x4_marker_to_camera = np.linalg.inv(rot_mat_4x4)
                return rot_mat_4x4_marker_to_camera, img
    return None, img_src


if __name__ == '__main__':
    fp_data = "PATH_TO_MP4"
    # read frame from video
    cap = cv2.VideoCapture(fp_data)
    ret, frame = cap.read()
    if ret:
        rot_mat_4x4_marker_to_camera, frame = estimate_homo_transform_matrix(
            frame)
        cv2.imshow('Image', frame)
        cv2.waitKey(0)
    cap.release()
