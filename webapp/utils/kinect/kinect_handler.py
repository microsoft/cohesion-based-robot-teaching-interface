import os
import cv2
from typing import Optional, Tuple
import numpy as np
import time
import copy
from utils.encoder import armarker_localizer
import time
import threading
from pyk4a import Config, ImageFormat, PyK4A


def draw_rectangles(frame, predicts):
    frame_render = frame.copy()
    if predicts is not None:
        for item in predicts:
            loc = item["location"]
            x_min = int(loc["left"] * frame.shape[1])
            x_max = int(loc["width"] * frame.shape[1]) + x_min
            y_min = int(loc["top"] * frame.shape[0])
            y_max = int(loc["height"] * frame.shape[0]) + y_min
            frame_render = cv2.rectangle(frame_render, (x_min, y_min), (x_max, y_max), (255, 0, 0), 4)
            cv2.putText(frame_render, item["name"].lower(), (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame_render


def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image


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


class kinect_handler:
    def __init__(self, size=(640, 360)):
        deviceid = 0
        self.imageformat = ImageFormat.COLOR_MJPG
        print(f"Starting device #{deviceid}")
        config = Config(color_format=self.imageformat)
        self.device = PyK4A(config=config, device_id=deviceid)
        self.device.start()

        self.recording = False
        self.thread_capture = None
        self.thread_record = None
        self.predicts = None
        self.centerrect_width = 0.2
        self.centerrect_height = 0.5
        self.size = size
        self.W_show = size[0]
        self.H_show = size[1]

        self.color_image = None
        self.depth_image = None
        self.is_capturing = None

    def start_capturing(self):
        if self.thread_capture is None:
            self.is_capturing = True
            self.thread_capture = threading.Thread(target=self.capture_images)
            self.thread_capture.start()
        else:
            print('thread is active')

    def stop_capturing(self):
        if self.thread_capture is not None and self.thread_capture.is_alive():
            self.is_capturing = False
            self.thread_capture = None
        else:
            print('thread is not alive')

    def get_frame(self, concat='vertical', frameguide=False):
        color_image = copy.deepcopy(self.color_image)
        _, color_image = armarker_localizer.estimate_homo_transform_matrix(
            color_image)
        resized = cv2.resize(
            color_image, (int(self.W_show), int(self.H_show)),
            interpolation=cv2.INTER_NEAREST)
        depth_image = copy.deepcopy(self.depth_image)
        resized_d = cv2.resize(
            depth_image, (int(self.W_show), int(self.H_show)),
            interpolation=cv2.INTER_NEAREST)
        resized_d_color = colorize(resized_d, (None, 5000))
        if frameguide:
            width = self.centerrect_width * resized.shape[1]
            height = self.centerrect_height * resized.shape[0]
            x_min = int(resized.shape[1]/2 - width/2)
            x_max = int(resized.shape[1]/2 + width/2)
            y_min = int(resized.shape[0]/2 - height/2)
            y_max = int(resized.shape[0]/2 + height/2)
            #import pdb;pdb.set_trace()
            resized = cv2.rectangle(resized, (x_min, y_min), (x_max, y_max), (0, 0, 255), 4)
        if concat == 'vertical':
            return cv2.vconcat([resized, resized_d_color])
        elif concat == 'horizontal':
            return cv2.hconcat([resized, resized_d_color])

    def save_image(self, dirname):
        import uuid
        import json
        savedir = os.path.join('..', 'image_database', dirname)
        print(savedir)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        tmp_name = str(uuid.uuid4())

        color_image = copy.deepcopy(self.color_image)
        depth_image = copy.deepcopy(self.depth_image)
        resized = cv2.resize(
            color_image, (int(self.W_show), int(self.H_show)),
            interpolation=cv2.INTER_NEAREST)

        frame_d = depth_image
        resized_d = cv2.resize(
            frame_d, (int(self.W_show), int(self.H_show)),
            interpolation=cv2.INTER_NEAREST)

        width = self.centerrect_width * resized.shape[1]
        height = self.centerrect_height * resized.shape[0]
        x_min = int(resized.shape[1]/2 - width/2)
        x_max = int(resized.shape[1]/2 + width/2)
        y_min = int(resized.shape[0]/2 - height/2)
        y_max = int(resized.shape[0]/2 + height/2)
        rec = {'top': y_min, 'bottom': y_max, 'left': x_min, 'right': x_max}
        with open(os.path.join(savedir, tmp_name+'.json'), 'w') as f:
            json.dump(rec, f, indent=4)
        cv2.imwrite(os.path.join(savedir, tmp_name+'.jpg'), resized)
        np.save(os.path.join(savedir, tmp_name+'.npy'), resized_d)
        return 'saved'

    def capture_images(self):
        i = 0
        cv2.namedWindow('Preview', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('Preview', -30, 0)
        while True:
            if self.is_capturing == False:
                break
            capture = self.device.get_capture()
            if capture.color is not None:
                self.color_image = convert_to_bgra_if_required(
                    self.imageformat, capture.color)
                self.color_image = copy.deepcopy(self.color_image)
            else:
                continue
            if capture.transformed_depth is not None:
                self.depth_image = copy.deepcopy(capture.transformed_depth)
            else:
                continue
            show_frame = self.get_frame()
            cv2.imshow('Preview', show_frame)
            if cv2.waitKey(1) == ord('q'):
                break
            i = i+1
            time.sleep(1.0/30)

    def start_recording(self, fp_dir, fp_base='sample', size=(1280, 720), save_fps=5):
        args = (fp_dir, fp_base, size, save_fps)
        if not os.path.exists(fp_dir):
            os.makedirs(fp_dir)
        self.recording = True
        self.thread_record = threading.Thread(target=self.record, args=args)
        self.thread_record.start()

    def stop_recording(self):
        if self.thread_record is not None and self.thread_record.is_alive():
            self.recording = False
        else:
            print('thread is not alive')
    # continue doing stuff

    def record(self, fp_dir, fp_base, size=(1280, 720), save_fps=0):
        fp_out_mp4 = None
        fp_out_depth_npy = None
        color_frame_count = 0
        depth_frame_count = 0
        if save_fps > 0:
            fs = save_fps
        else:
            fs = 30
        W_save, H_save = size[0], size[1]
        depth_stack = []
        fp_out_mp4 = os.path.join(fp_dir, fp_base + ".mp4")
        fp_out_depth_npy = os.path.join(fp_dir, fp_base + "_depth.npz")

        videowriter = cv2.VideoWriter(
            fp_out_mp4, cv2.VideoWriter_fourcc(
                *"mp4v"), fs, (W_save, H_save))
        fp_out_depth_mp4 = os.path.join(fp_dir, fp_base + "_depth.mp4")
        videowriter_d = cv2.VideoWriter(
            fp_out_depth_mp4, cv2.VideoWriter_fourcc(
                *"mp4v"), fs, (W_save, H_save))
        # get current time in ms
        past_time = int(round(time.time() * 1000))
        past_time_save = past_time

        while True:
            if self.recording:
                # get current time in ms
                current_time = int(round(time.time() * 1000))
                time_diff = current_time - past_time_save
                if time_diff > int(1000 * (1.0 / fs))*0.9:  # multiply 0.9 to adjust fps delay
                    past_time_save = current_time
                    color_image = copy.deepcopy(self.color_image)
                    depth_image = copy.deepcopy(self.depth_image)
                    resized = cv2.resize(
                        color_image, (int(W_save), int(H_save)),
                        interpolation=cv2.INTER_NEAREST)

                    frame_d = depth_image
                    resized_d = cv2.resize(
                        frame_d, (int(W_save), int(H_save)),
                        interpolation=cv2.INTER_NEAREST)
                    resized_d_color = colorize(resized_d, (None, 5000))
                    videowriter.write(resized)
                    color_frame_count += 1
                    depth_stack.append(resized_d)
                    videowriter_d.write(resized_d_color)
                    depth_frame_count += 1
            else:
                break
        print(f"{color_frame_count} frames written (RGB).")
        print(f"{depth_frame_count} frames written (D).")
        videowriter.release()
        videowriter_d.release()
        #import pdb;pdb.set_trace()
        if len(depth_stack) > 0:
            depth_data = np.stack(depth_stack)
            np.savez_compressed(fp_out_depth_npy, depth_data)
        ret = {
            "fp_out_mp4": fp_out_mp4,
            "fp_out_depth_npy": fp_out_depth_npy}
        return ret
