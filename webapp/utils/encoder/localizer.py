from argparse import _CountAction
from importlib.resources import path
import json
from operator import contains
import cv2
import requests
from typing import Optional, Tuple
import numpy as np
import os
import tempfile
import open3d
import shutil

def create_pointcloud(img, depth, cameramodel=None): 
    rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(
        open3d.geometry.Image(img), open3d.geometry.Image(depth),
        depth_trunc=3.0,
        # depth_scale=1.0,
        convert_rgb_to_intensity=False)
    if cameramodel is not None:
        intrinsic = open3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            cameramodel['width'],
            cameramodel['height'],
            cameramodel['fx'],
            cameramodel['fy'],
            cameramodel['cx'],
            cameramodel['cy'])
        print(intrinsic.intrinsic_matrix)
        print(intrinsic.width)
        print(intrinsic.height)
    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic,
        project_valid_depth_only=False)
    return pcd


def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


def landmark_finder(fp_img, output_path):
    import cv2
    import mediapipe as mp
    import copy

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        image = cv2.imread(fp_img)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        annotated_image = image.copy()
        if not results.multi_hand_landmarks:
            return None
        scores = [x.classification[0].score for x in results.multi_handedness]
        # argmax returns the index of the max value in the list
        max_score_index = scores.index(max(scores))
        hand_landmarks = results.multi_hand_landmarks[max_score_index]
        #print('hand_landmarks:', hand_landmarks)
        image_height, image_width, _ = image.shape
        print(
            f'Index finger tip coordinates: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})')
        contact_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
        contact_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
        contact_point = {"x": int(contact_x), "y": int(contact_y)}
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imwrite(output_path, annotated_image)
        return contact_point


def upload_data(
        upload_file_depth,
        upload_file_rgb,
        upload_json_roi,
        upload_json_camera_param,
        upload_json_contact_point=None):
    url = 'http://DOMAIN:PORT/position_extraction'
    headers = {'accept': 'application/json'}
    if upload_json_contact_point is None:
        data = {
            'upload_file_depth': open(
                upload_file_depth, 'rb'), 'upload_file_rgb': open(
                upload_file_rgb, 'rb'), 'upload_json_roi': open(
                upload_json_roi, 'rb'), 'upload_json_camera_param': open(
                    upload_json_camera_param, 'rb')}
    else:
        data = {'upload_file_depth': open(upload_file_depth, 'rb'),
                'upload_file_rgb': open(upload_file_rgb, 'rb'),
                'upload_json_roi': open(upload_json_roi, 'rb'),
                'upload_json_camera_param': open(upload_json_camera_param, 'rb'),
                'upload_json_contact_point': open(upload_json_contact_point, 'rb')}
    response = requests.post(url, headers=headers,
                             files=data)
    #data = response.data()
    return response


def run(frame_img, frame_depth, loc_json, output_dir, dir_name):
    debug = False
    with tempfile.TemporaryDirectory() as output_dir_tmp:
        fp_tmp_json = os.path.join(output_dir_tmp, 'loc_tmp.json')
        fp_tmp_img = os.path.join(output_dir_tmp, 'loc_tmp.png')
        fp_tmp_depth = os.path.join(output_dir_tmp, 'loc_tmp.npy')
        fp_tmp_out = os.path.join(output_dir_tmp, 'loc_tmp.zip')

        fp_tmp_cameraparam = os.path.join(output_dir_tmp, 'cameramodel.json')
        fp_cameraparam = os.path.join(
            os.getcwd(), 'settings', 'cameramodel.json')
        H, W = frame_img.shape[:2]
        with open(fp_cameraparam) as json_file:
            data = json.load(json_file)
            scale = W / float(data["width"])
            data["width"] = W
            data["height"] = H
            data["fx"] = data["fx"] * scale
            data["fy"] = data["fy"] * scale
            data["cx"] = data["cx"] * scale
            data["cy"] = data["cy"] * scale
            # print(data)
        if debug:
            rgb_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            open3d_cloud = create_pointcloud(rgb_img, frame_depth, data)
            import open3d as o3d
            o3d.io.write_point_cloud(output_dir+'\\debug.ply', open3d_cloud)
        with open(fp_tmp_cameraparam, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        # save data
        cv2.imwrite(fp_tmp_img, frame_img)
        np.save(fp_tmp_depth, frame_depth)
        with open(fp_tmp_json, 'w') as outfile:
            json.dump(loc_json, outfile, indent=4)

        response = upload_data(
            fp_tmp_depth,
            fp_tmp_img,
            fp_tmp_json,
            fp_tmp_cameraparam)
        data = response.content
        with open(fp_tmp_out, 'wb') as s:
            s.write(data)

        out_dir = os.path.join(output_dir, 'localization', dir_name)
        try:
            shutil.unpack_archive(fp_tmp_out, out_dir)
        except BaseException:
            print('unpack failed (localizer.py)')
            print(data)
        return os.path.join(
            out_dir, 'position.json'), os.path.join(
            out_dir, 'pcd_output.ply')


def run_find_contactweb(frame_img, frame_depth, hand_frame_img, loc_json, output_dir, dir_name):
    with tempfile.TemporaryDirectory() as output_dir_tmp:
        fp_tmp_json = os.path.join(output_dir_tmp, 'loc_tmp.json')
        fp_tmp_img = os.path.join(output_dir_tmp, 'loc_tmp.png')
        fp_tmp_img_crop = os.path.join(output_dir_tmp, 'loc_tmp_crop.png')
        fp_tmp_depth = os.path.join(output_dir_tmp, 'loc_tmp.npy')
        fp_tmp_out = os.path.join(output_dir_tmp, 'loc_tmp.zip')
        fp_tmp_contactpoint = os.path.join(
            output_dir_tmp, 'loc_tmp_contactpoint.json')

        fp_tmp_cameraparam = os.path.join(output_dir_tmp, 'cameramodel.json')
        fp_cameraparam = os.path.join(
            os.getcwd(), 'settings', 'cameramodel.json')
        H, W = frame_img.shape[:2]
        with open(fp_cameraparam) as json_file:
            data = json.load(json_file)
            scale = W / float(data["width"])
            data["width"] = W
            data["height"] = H
            data["fx"] = data["fx"] * scale
            data["fy"] = data["fy"] * scale
            data["cx"] = data["cx"] * scale
            data["cy"] = data["cy"] * scale
            # print(data)
        with open(fp_tmp_cameraparam, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        # save data
        cv2.imwrite(fp_tmp_img, frame_img)
        np.save(fp_tmp_depth, frame_depth)
        with open(fp_tmp_json, 'w') as outfile:
            json.dump(loc_json, outfile, indent=4)

        fp_out = os.path.join(output_dir, 'hand_landmark.jpg')
        frame_crop = hand_frame_img[
            loc_json['top']:loc_json['bottom'],
            loc_json['left']:loc_json['right']]
        cv2.imwrite(fp_tmp_img_crop, frame_crop)

        contact_point = landmark_finder(fp_tmp_img_crop, fp_out)
        print(contact_point)
        if contact_point is None:
            return None, None
        with open(fp_tmp_contactpoint, 'w') as outfile:
            json.dump(contact_point, outfile, indent=4)
        response = upload_data(
            fp_tmp_depth,
            fp_tmp_img,
            fp_tmp_json,
            fp_tmp_cameraparam,
            upload_json_contact_point=fp_tmp_contactpoint)

        data = response.content
        with open(fp_tmp_out, 'wb') as s:
            s.write(data)

        import shutil
        out_dir = os.path.join(output_dir, 'localization', dir_name)
        # shutil.unpack_archive(fp_tmp_out, out_dir)
        try:
            shutil.unpack_archive(fp_tmp_out, out_dir)
        except BaseException:
            print('unpack failed (localizer.py)')
            print(data)
        return os.path.join(
            out_dir, 'position.json'), os.path.join(
            out_dir, 'pcd_output.ply')
