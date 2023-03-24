import numpy as np
import cv2
from typing import Optional, Tuple
import json
import open3d as o3d
import os
import copy
from . import hand_detection
from . import object_detection
from . import localizer
from . import grasptype_recognizer
from . import armarker_localizer
from . import trajection_fitting
from . import frame_to_laban


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


class task_daemon:
    def __init__(
            self,
            task,
            verbal_input,
            object_name,
            fp_mp4,
            fp_depth_npy,
            output_dir,
            hand_laterality='unknown',
            id='default'):
        self.hand_laterality = hand_laterality
        self.task = task
        with open(os.path.join(os.getcwd(), 'utils', 'encoder', 'task_models', self.task + '.json')) as f:
            self.taskmodel_json = json.load(f)
        self.verbal_input = verbal_input
        self.fp_mp4 = fp_mp4
        self.fp_depth_npy = fp_depth_npy
        self.id = id + '_' + task
        self.output_dir = output_dir
        self.object_name = object_name
        # create dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        cap = cv2.VideoCapture(str(self.fp_mp4))
        ret, _ = cap.read()
        if ret:
            # video info
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.segment_timings_frame = (0, self.video_length)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.segment_timings_frame[0])
            _, self.first_frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.segment_timings_frame[1] - 1)
            _, self.last_frame = cap.read()
            # save images
            self.fp_first_frame = os.path.join(
                self.output_dir, 'first_frame.png')
            self.fp_last_frame = os.path.join(
                self.output_dir, 'last_frame.png')
            cv2.imwrite(self.fp_first_frame, self.first_frame)
            cv2.imwrite(self.fp_last_frame, self.last_frame)
        else:
            print("failed to open video")
            return
        # try to find ar marker
        rot_mat_4x4_marker_to_camera, frame = armarker_localizer.estimate_homo_transform_matrix(
            self.first_frame)
        if rot_mat_4x4_marker_to_camera is None:
            print('AR marker was not found. 3D points will not be transformed')
            self.rot_mat_4x4_marker_to_camera = None
        else:
            self.rot_mat_4x4_marker_to_camera = rot_mat_4x4_marker_to_camera
            cv2.imwrite(
                os.path.join(
                    self.output_dir,
                    'ar_marker_first_frame.png'),
                frame)
            # save as npy
            np.save(
                os.path.join(
                    self.output_dir,
                    'rot_mat_4x4_marker_to_camera.npy'),
                self.rot_mat_4x4_marker_to_camera)
        print(fp_depth_npy)
        try:
            depth_array = np.load(fp_depth_npy)
            depth_array = depth_array['arr_0']
            self.first_frame_depth = depth_array[self.segment_timings_frame[0]]
            self.last_frame_depth = depth_array[min(
                self.segment_timings_frame[1], len(depth_array)-1)]
        except:
            import pdb;pdb.set_trace()
        # need to reshape the depth to be exactly the same as the video
        self.first_frame_depth = cv2.resize(
            self.first_frame_depth,
            (self.width,
             self.height),
            interpolation=cv2.INTER_NEAREST)
        self.last_frame_depth = cv2.resize(
            self.last_frame_depth,
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST)
        # save depth as npy
        np.save(
            os.path.join(
                self.output_dir,
                'first_frame_depth.npy'),
            self.first_frame_depth)
        np.save(
            os.path.join(
                self.output_dir,
                'last_frame_depth.npy'),
            self.last_frame_depth)

        # information about the manipulating hand
        self.pos_hand_first = None  # xyz
        self.pos_hand_last = None  # xyz
        self.loc_hand_first = None  # bbox
        self.loc_hand_last = None  # bbox
        self.fp_first_frame_hand = None
        self.fp_last_frame_hand = None
        self.fp_loc_hand_first = None
        self.fp_loc_hand_last = None
        self.fp_3dmodel_hand_first = None
        self.fp_3dmodel_hand_last = None
        self.pos_hand_trajectory = None  # used only for PTG3 and PTG5
        self.rotation_axis = None  # used only for PTG5

        # information about the object
        #self.object_name = None
        self.object_attribute = None
        self.pos_object_first = None  # xyz
        self.pos_object_last = None  # xyz
        self.loc_object_first = None  # bbox
        self.loc_object_last = None  # bbox
        self.fp_first_frame_object = None
        self.fp_last_frame_object = None
        self.fp_loc_object_first = None
        self.fp_loc_object_last = None
        self.fp_3dmodel_object_first = None
        self.fp_3dmodel_object_last = None

        # other information
        self.hand_direction = None
        self.first_laban = None
        self.last_laban = None

        if hand_laterality == 'unknown':  # recognizie hand laterality
            # find object
            (self.loc_object_first,
                self.loc_object_last,
                self.fp_first_frame_object,
                self.fp_last_frame_object) = self.extract_object()
            handloc_right, handloc_left = self.extract_hands(self.last_frame)
            # check the hand laterality based on the distance to the object
            dist_left = self.loc_distance(
                self.loc_object_first, handloc_left)
            dist_right = self.loc_distance(
                self.loc_object_first, handloc_right)
            if dist_left > dist_right:
                self.hand_laterality = 'right'
            elif dist_left < dist_right:
                self.hand_laterality = 'left'
            else:
                self.hand_laterality = 'unknown'
        else:  # normal encoding
            print("finding hand locations in 2D: " + self.id)
            (self.loc_hand_first,
                self.loc_hand_last,
                self.fp_first_frame_hand,
                self.fp_last_frame_hand) = self.extract_hands_with_laterality()
            if len(self.loc_hand_first) > 0:
                fp_tmp_json = os.path.join(
                    self.output_dir,
                    'first_frame_hand_loc.json')
                with open(fp_tmp_json, 'w') as outfile:
                    json.dump(self.loc_hand_first, outfile, indent=4)
            if len(self.loc_hand_last) > 0:
                fp_tmp_json = os.path.join(
                    self.output_dir,
                    'last_frame_hand_loc.json')
                with open(fp_tmp_json, 'w') as outfile:
                    json.dump(self.loc_hand_last, outfile, indent=4)
            print("finding hand locations in 3D: " + self.id)
            (self.pos_hand_first,
                self.pos_hand_last,
                self.fp_loc_hand_first,
                self.fp_loc_hand_last,
                self.fp_3dmodel_hand_first,
                self.fp_3dmodel_hand_last) = self.localization_hand()
            # data transform (hand)
            self.pos_hand_first = self.transform(self.pos_hand_first)
            self.pos_hand_last = self.transform(self.pos_hand_last)
            self.hand_direction = self.set_handdirection()
            print("Getting laban: " + self.id)
            self.first_laban, self.last_laban = self.extract_laban()
            print("finding objects in 2D: " + self.id)
            (self.loc_object_first,
                self.loc_object_last,
                self.fp_first_frame_object,
                self.fp_last_frame_object) = self.extract_object()
            if self.task == "GRASP":
                (self.pos_object_first, self.fp_3dmodel_object_first,
                    self.fp_loc_object_first) = self.localization_object(
                    self.first_frame,
                    self.first_frame_depth,
                    self.loc_object_first,
                    self.object_name)

                self.pos_object_first = self.transform(self.pos_object_first)
                print('Object position: {}'.format(self.pos_object_first))

    def get_grasptype(self):
        if self.fp_first_frame_hand is not None:
            object_name = 'none'
            if self.object_name is not None and self.object_name != "UNKNOWN":
                object_name = self.object_name
            return grasptype_recognizer.run_image(
                self.fp_last_frame_hand, object_name)
        else:
            return None

    def show_rgb_images(self):
        # horizontal concatenation
        show_frame = np.concatenate(
            (self.first_frame, self.last_frame), axis=1)
        cv2.imshow("first_and_last_frame", show_frame)
        cv2.waitKey(0)

    def show_depth_images(self):
        # horizontal concatenation
        frame_d_f = colorize(self.first_frame_depth, (None, 5000))
        frame_d_l = colorize(self.last_frame_depth, (None, 5000))
        show_frame = np.concatenate((frame_d_f, frame_d_l), axis=1)
        cv2.imshow("first_and_last_frame(depth)", show_frame)
        cv2.waitKey(0)

    def loc_distance(self, loc_1, loc_2):
        if loc_1 is None or loc_2 is None:
            return np.Inf
        else:
            y_1 = loc_1['top'] + 0.5 * loc_1['bottom']
            x_1 = loc_1['left'] + 0.5 * loc_1['right']
            y_2 = loc_2['top'] + 0.5 * loc_2['bottom']
            x_2 = loc_2['left'] + 0.5 * loc_2['right']
            return np.sqrt((y_1 - y_2)**2 + (x_1 - x_2)**2)

    def extract_hands(self, frame):
        json_data = hand_detection.run_image(
            frame)
        handloc_right = json_data['location_righthand']
        handloc_left = json_data['location_lefthand']
        if len(handloc_right) == 0:
            handloc_right = None
        if len(handloc_left) == 0:
            handloc_left = None
        return handloc_right, handloc_left

    def extract_hands_with_laterality(self):
        handloc_first = None
        handloc_last = None
        fp_first_frame_hand = None
        fp_last_frame_hand = None
        json_data_first = hand_detection.run_image(
            self.first_frame)
        json_data_last = hand_detection.run_image(
            self.last_frame)

        if self.hand_laterality == 'right':
            handloc_first = json_data_first['location_righthand']
            handloc_last = json_data_last['location_righthand']
        else:
            handloc_first = json_data_first['location_lefthand']
            handloc_last = json_data_last['location_lefthand']
        # crop hand images
        if len(handloc_first) > 0:
            frame_crop_first_frame = self.first_frame[
                handloc_first['top']:handloc_first['bottom'],
                handloc_first['left']:handloc_first['right']]
            fp_first_frame_hand = os.path.join(
                self.output_dir, 'first_frame_hand.png')
            cv2.imwrite(fp_first_frame_hand, frame_crop_first_frame)
        if len(handloc_last) > 0:
            frame_crop_last_frame = self.last_frame[
                handloc_last['top']:handloc_last['bottom'],
                handloc_last['left']:handloc_last['right']]
            fp_last_frame_hand = os.path.join(
                self.output_dir, 'last_frame_hand.png')
            cv2.imwrite(fp_last_frame_hand, frame_crop_last_frame)
        return handloc_first, handloc_last, fp_first_frame_hand, fp_last_frame_hand

    def extract_allhands_with_laterality(self, skip_frame=5):
        time_focus = np.arange(
            self.segment_timings_frame[0] /
            float(
                self.fps),
            self.segment_timings_frame[1] /
            float(
                self.fps),
            skip_frame /
            float(
                self.fps)).tolist()
        frame_focus = np.arange(
            self.segment_timings_frame[0],
            self.segment_timings_frame[1],
            skip_frame).tolist()
        json_send = {}
        json_send['time_focus'] = time_focus
        json_data = hand_detection.run_allframe(
            self.fp_mp4, json_send, self.output_dir)
        cap = cv2.VideoCapture(str(self.fp_mp4))
        depth_array = np.load(self.fp_depth_npy)
        loc_xyz_list = []
        for i, loc in enumerate(json_data):
            if self.hand_laterality == 'right':
                handloc = loc['location_righthand']
            else:
                handloc = loc['location_lefthand']
            # crop hand images
            if len(handloc) > 0:
                frame_crop_first_frame = self.first_frame[
                    handloc['top']:handloc['bottom'],
                    handloc['left']:handloc['right']]
                tmp_focus_frame = frame_focus[i]
                cap.set(cv2.CAP_PROP_POS_FRAMES, tmp_focus_frame)
                _, frame = cap.read()
                frame_depth = depth_array[tmp_focus_frame]
                # need to reshape the depth to be exactly the same as the video
                frame_depth = cv2.resize(
                    frame_depth,
                    (self.width,
                     self.height),
                    interpolation=cv2.INTER_NEAREST)
                frame = cv2.resize(
                    frame,
                    (self.width, self.height),
                    interpolation=cv2.INTER_NEAREST)
                fp_loc, _ = localizer.run(
                    frame, frame_depth, handloc, self.output_dir, "hand_" + str(i))
                if fp_loc is not None:
                    with open(fp_loc) as f:
                        json_data = json.load(f)
                        loc_xyz = json_data['roi_position']
                        loc_xyz_list.append(self.transform(loc_xyz))
        np_loc_xyz_list = np.array(loc_xyz_list)
        np_loc_xyz_list = np.reshape(np_loc_xyz_list, (-1, 3))
        np.savetxt(
            os.path.join(
                self.output_dir,
                "trajectory.csv"),
            np_loc_xyz_list,
            delimiter=',')
        return loc_xyz_list

    def extract_handpos_from_bone(self, fp_bone, mode='first', laterality='right'):
        # For static images:
        bone_array = np.load(fp_bone)
        bone_array = bone_array['arr_0']
        if mode == 'first':
            idx = 0
            target = bone_array[idx]
            while np.any(np.isnan(target)):
                idx = idx + 1
                target = bone_array[idx]
                if idx >= len(bone_array):
                    return None
        elif mode == 'last':
            idx = -1
            target = bone_array[idx]
            while np.any(np.isnan(target)):
                idx = idx - 1
                target = bone_array[idx]
                if abs(idx) >= len(bone_array):
                    return None
        left_wrist_position = target[7][:3]/1000
        right_wrist_position = target[14][:3]/1000
        if laterality == 'right':
            return right_wrist_position.tolist(), idx, len(bone_array)
        if laterality == 'left':
            return left_wrist_position.tolist(), idx, len(bone_array)

    def _extract_pointcloud(self):
        loc = {}
        loc['top'] = 0
        loc['bottom'] = self.height - 1
        loc['left'] = 0
        loc['right'] = self.width - 1
        _, fp_first = localizer.run(
            self.first_frame, self.first_frame_depth, loc, self.output_dir, "first_frame")
        _, fp_last = localizer.run(
            self.last_frame, self.last_frame_depth, loc, self.output_dir, "last_frame")
        return fp_first, fp_last

    def extract_laban(self):
        first_laban = frame_to_laban.transform_to_laban(
            self.fp_first_frame, self.output_dir)
        last_laban = frame_to_laban.transform_to_laban(
            self.fp_last_frame, self.output_dir)
        return first_laban, last_laban

    def extract_object(self):
        objectloc_first = None
        objectloc_last = None
        fp_first_frame_object = None
        fp_last_frame_object = None
        json_data_first = object_detection.run_image(
            self.first_frame, self.object_name)
        json_data_last = object_detection.run_image(
            self.last_frame, self.object_name)

        objectloc_first = json_data_first['location']
        objectloc_last = json_data_last['location']
        print(json_data_first['found_objects'])

        # crop object images
        if len(objectloc_first) > 0:
            frame_crop_first_frame = self.first_frame[
                objectloc_first['top']:objectloc_first['bottom'],
                objectloc_first['left']:objectloc_first['right']]
            fp_first_frame_object = os.path.join(
                self.output_dir, 'first_frame_object.png')
            cv2.imwrite(fp_first_frame_object, frame_crop_first_frame)
        else:
            objectloc_first = None
        if len(objectloc_last) > 0:
            frame_crop_last_frame = self.last_frame[
                objectloc_last['top']:objectloc_last['bottom'],
                objectloc_last['left']:objectloc_last['right']]
            fp_last_frame_object = os.path.join(
                self.output_dir, 'last_frame_object.png')
            cv2.imwrite(fp_last_frame_object, frame_crop_last_frame)
        else:
            objectloc_last = None
        return objectloc_first, objectloc_last, fp_first_frame_object, fp_last_frame_object

    def localization_hand(self):
        loc_hand_first = None
        loc_hand_last = None
        fp_loc_hand_first = None
        fp_loc_hand_last = None
        fp_3dmodel_hand_first = None
        fp_3dmodel_hand_last = None
        if len(self.loc_hand_first) > 0:
            fp_loc_hand_first, fp_3dmodel_hand_first = localizer.run(
                self.first_frame, self.first_frame_depth, self.loc_hand_first, self.output_dir, "hand_first")
        if len(self.loc_hand_last) > 0:
            fp_loc_hand_last, fp_3dmodel_hand_last = localizer.run(
                self.last_frame, self.last_frame_depth, self.loc_hand_last, self.output_dir, "hand_last")
        # load json
        if fp_loc_hand_first is not None:
            with open(fp_loc_hand_first) as f:
                json_data = json.load(f)
                loc_hand_first = json_data['roi_position']
        if fp_loc_hand_last is not None:
            with open(fp_loc_hand_last) as f:
                json_data = json.load(f)
                loc_hand_last = json_data['roi_position']
        return loc_hand_first, loc_hand_last, fp_loc_hand_first, fp_loc_hand_last, fp_3dmodel_hand_first, fp_3dmodel_hand_last

    def localization_object(self, frame, depth, loc, object_name):
        pos_object = None
        fp_loc_object = None
        fp_3dmodel_object = None
        fp_loc_object, fp_3dmodel_object = localizer.run(
            frame, depth, loc, self.output_dir, object_name)
        if fp_loc_object is not None and os.path.exists(fp_loc_object):
            with open(fp_loc_object) as f:
                json_data = json.load(f)
                pos_object = json_data['roi_position']
        return pos_object, fp_3dmodel_object, fp_loc_object

    def localization_contactweb(self, frame, depth, hand_frame, loc):
        pos_object = None
        fp_loc_object = None
        fp_3dmodel_object = None
        fp_loc_object, fp_3dmodel_object = localizer.run_find_contactweb(
            frame, depth, hand_frame, loc, self.output_dir, "contactweb")
        # load json
        if fp_loc_object is not None:
            with open(fp_loc_object) as f:
                json_data = json.load(f)
                pos_object = json_data['roi_position']
        return pos_object, fp_3dmodel_object, fp_loc_object

    def transform(self, xyz_array):
        # transform the coordinates to the original image
        if self.rot_mat_4x4_marker_to_camera is not None and xyz_array is not None:
            # copy the array
            tmp_xyz_array = copy.deepcopy(xyz_array)
            tmp_xyz_array.append(1)
            tmp_xyz_array_ret = np.dot(
                self.rot_mat_4x4_marker_to_camera, tmp_xyz_array)
            return tmp_xyz_array_ret[0:3].tolist()
        else:
            return xyz_array

    def set_handdirection(self):
        hand_direction = None
        if self.pos_hand_last is not None and self.pos_hand_first is not None:
            # subtraction as numpy array
            hand_direction = list(
                np.array(
                    self.pos_hand_last) -
                np.array(
                    self.pos_hand_first))
        return hand_direction

    def show_pointcloud(self):
        pcd_list = []
        pcd = o3d.io.read_point_cloud(self.fp_3dmodel_hand_first)
        pcd.remove_non_finite_points()
        pcd.uniform_down_sample(7)
        # transform the coordinates to the original image
        if self.rot_mat_4x4_marker_to_camera is not None:
            pcd.transform(self.rot_mat_4x4_marker_to_camera)
        pcd_list.append(pcd)
        pcd = o3d.io.read_point_cloud(self.fp_3dmodel_hand_last)
        pcd.remove_non_finite_points()
        pcd.uniform_down_sample(7)
        pcd_list.append(pcd)
        o3d.visualization.draw_geometries(pcd_list)

    def set_skillparameters(self):
        skill_params = [
            i for i in self.taskmodel_json.keys() if not i.startswith('_')]
        for param_name in skill_params:
            if "hand_laterality" in param_name:
                if self.taskmodel_json[param_name]["value"] is None:
                    print("filling hand laterality...")
                    self.taskmodel_json[param_name]["value"] = self.hand_laterality
            if "detach_direction" in param_name or "approach_direction" in param_name or "depart_direction" in param_name or "attach_direction" in param_name:
                if self.taskmodel_json[param_name]["value"] is None:
                    print("filling direction...")
                    self.taskmodel_json[param_name]["value"] = self.hand_direction
                    if self.rot_mat_4x4_marker_to_camera is None:
                        self.taskmodel_json[param_name]["coordinate_system"] = "camera"
                else:
                    self.hand_direction = self.taskmodel_json[param_name]["value"]
            if "start_position" in param_name:
                if self.taskmodel_json[param_name]["value"] is None:
                    print("filling start_position...")
                    self.taskmodel_json[param_name]["value"] = self.pos_hand_first
                    if self.rot_mat_4x4_marker_to_camera is None:
                        self.taskmodel_json[param_name]["coordinate_system"] = "camera"
                else:
                    self.pos_hand_first = self.taskmodel_json[param_name]["value"]
            if "end_position" in param_name:
                if self.taskmodel_json[param_name]["value"] is None:
                    print("filling end_position...")
                    self.taskmodel_json[param_name]["value"] = self.pos_hand_last
                    if self.rot_mat_4x4_marker_to_camera is None:
                        self.taskmodel_json[param_name]["coordinate_system"] = "camera"
                else:
                    self.pos_hand_last = self.taskmodel_json[param_name]["value"]
            if "start_laban" in param_name:
                if self.taskmodel_json[param_name]["value"] is None:
                    print("filling start_laban...")
                    self.taskmodel_json[param_name]["value"] = self.first_laban
                else:
                    self.first_laban = self.taskmodel_json[param_name]["value"]
            if "end_laban" in param_name:
                if self.taskmodel_json[param_name]["value"] is None:
                    print("filling end_laban...")
                    self.taskmodel_json[param_name]["value"] = self.last_laban
                else:
                    self.last_laban = self.taskmodel_json[param_name]["value"]
            if "target_object_name" in param_name:
                if self.taskmodel_json[param_name]["value"] is None:
                    print("filling target_object_name...")
                    self.taskmodel_json[param_name]["value"] = self.object_name
                else:
                    self.object_name = self.taskmodel_json[param_name]["value"]
            if "target_object_attribute" in param_name:
                if self.taskmodel_json[param_name]["value"] is None:
                    print("filling target_object_attribute...")
                    self.taskmodel_json[param_name]["value"] = self.object_attribute
                else:
                    self.object_attribute = self.taskmodel_json[param_name]["value"]
            if "grasp_type" in param_name:
                if self.taskmodel_json[param_name]["value"] is None:
                    print("filling grasp_type...")
                    self.taskmodel_json[param_name]["value"] = self.get_grasptype(
                    )

            if "rotation_axis" in param_name:
                if self.taskmodel_json[param_name]["value"] is None:
                    print("filling rotation parameters...")
                    if self.taskmodel_json["hand_trajectory"]["value"] is None:
                        print("filling hand_trajectory...")
                        if self.pos_hand_trajectory is None:
                            self.pos_hand_trajectory = self.extract_allhands_with_laterality()
                        self.taskmodel_json["hand_trajectory"]["value"] = self.pos_hand_trajectory
                        self.taskmodel_json["hand_trajectory"]["filled_by_daemon"] = True
                    else:
                        self.pos_hand_trajectory = self.taskmodel_json["hand_trajectory"]["value"]
                    loc_xyz_list = np.array(self.pos_hand_trajectory)

                    # TODO detect rotation axis direction. Use [0,0,1] as
                    # default
                    rotation_axis = [0, 0, 1]
                    self.taskmodel_json["rotation_axis"]["value"] = rotation_axis
                    self.taskmodel_json["rotation_axis"]["filled_by_daemon"] = True

                    if rotation_axis == [0, 0, 1]:
                        xi = loc_xyz_list[:, 0]
                        yi = loc_xyz_list[:, 1]
                        a, b, r = trajection_fitting.circle(xi, yi)
                        rotation_center_position = [float(a), float(
                            b), float(np.median(loc_xyz_list[:, 2]))]
                    else:
                        pass  # TODO vertical rotation
                    self.taskmodel_json["rotation_center_position"]["value"] = rotation_center_position
                    self.taskmodel_json["rotation_center_position"]["filled_by_daemon"] = True
                    self.taskmodel_json["rotation_radius"]["value"] = float(r)
                    self.taskmodel_json["rotation_radius"]["filled_by_daemon"] = True

            # set flag
            if self.taskmodel_json[param_name]["value"] is not None:
                self.taskmodel_json[param_name]["filled_by_daemon"] = True

    def dump_json(self):
        # save json
        with open(os.path.join(self.output_dir, 'taskmodel.json'), 'w') as f:
            json.dump(self.taskmodel_json, f, indent=4)

    def load_dumped_json(self):
        # open json
        with open(os.path.join(self.output_dir, 'taskmodel.json'), 'r') as f:
            return json.load(f)
