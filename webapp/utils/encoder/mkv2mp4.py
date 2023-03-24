from pyk4a import PyK4APlayback, ImageFormat
import cv2
from typing import Optional, Tuple
import numpy as np


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


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


def extract(
        playback: PyK4APlayback,
        fp: str,
        fs=30,
        output_dir='./',
        scale=1.0):
    # write the mp4
    color_frame_count = 0
    depth_frame_count = 0
    depth_stack = []
    import os
    filename = os.path.basename(fp)
    fp_out_mp4 = os.path.join(output_dir, filename.replace(".mkv", ".mp4"))
    fp_out_depth_mp4 = os.path.join(
        output_dir, filename.replace(
            ".mkv", "_depth.mp4"))
    fp_out_depth_npy = os.path.join(
        output_dir, filename.replace(
            ".mkv", "_depth.npy"))
    fp_out_depth_npz = os.path.join(
        output_dir, filename.replace(
            ".mkv", "_depth.npz"))
    try:
        while True:
            try:
                capture = playback.get_next_capture()
                if (capture.color is not None) and (
                        capture.transformed_depth is not None):
                    frame = convert_to_bgra_if_required(
                        playback.configuration["color_format"], capture.color)
                    frame_d = colorize(capture.transformed_depth, (None, 5000))
                    H, W = frame.shape[:2]
                    H = int(H * scale)
                    W = int(W * scale)

                    H_d, W_d = frame_d.shape[:2]
                    H_d = int(H_d * scale)
                    W_d = int(W_d * scale)
                    videowriter = cv2.VideoWriter(
                        fp_out_mp4, cv2.VideoWriter_fourcc(
                            *"mp4v"), fs, (W, H))
                    videowriter_d = cv2.VideoWriter(
                        fp_out_depth_mp4, cv2.VideoWriter_fourcc(
                            *"mp4v"), fs, (W_d, H_d))
                    # check if H == H_d and W == W_d
                    if H != H_d or W != W_d:
                        # runtime error
                        raise RuntimeError
                    frame = cv2.resize(
                        frame, (W, H), interpolation=cv2.INTER_NEAREST)
                    frame_d = cv2.resize(
                        frame_d, (W_d, H_d), interpolation=cv2.INTER_NEAREST)
                    videowriter.write(frame)
                    videowriter_d.write(frame_d)
                    depth_stack.append(
                        cv2.resize(
                            capture.transformed_depth,
                            (W_d,
                             H_d),
                            interpolation=cv2.INTER_NEAREST))
                    color_frame_count += 1
                    depth_frame_count += 1
                    break
            except EOFError:
                break
        while True:
            try:
                capture = playback.get_next_capture()
                if capture.color is not None:
                    frame = convert_to_bgra_if_required(
                        playback.configuration["color_format"], capture.color)
                    frame = cv2.resize(
                        frame, (W, H), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Color", frame)
                    videowriter.write(frame)
                    color_frame_count += 1
                if capture.transformed_depth is not None:
                    frame_d = colorize(capture.transformed_depth, (None, 5000))
                    frame_d = cv2.resize(
                        frame_d, (W_d, H_d), interpolation=cv2.INTER_NEAREST)
                    depth_stack.append(
                        cv2.resize(
                            capture.transformed_depth,
                            (W_d,
                             H_d),
                            interpolation=cv2.INTER_NEAREST))
                    cv2.imshow("Depth", frame_d)
                    videowriter_d.write(frame_d)
                    depth_frame_count += 1
                key = cv2.waitKey(10)
                if key != -1:
                    break
            except EOFError:
                break
        videowriter.release()
        videowriter_d.release()
        depth_data = np.stack(depth_stack)
        # save ndarray
        np.save(fp_out_depth_npy, depth_data)
        #np.savez_compressed(fp_out_depth_npz, depth_data)
    except EOFError:
        pass
    cv2.destroyAllWindows()
    print(f"Color frames: {color_frame_count}")
    print(f"Depth frames: {depth_frame_count}")
    return fp_out_mp4, fp_out_depth_mp4, fp_out_depth_npy


def run(fp, output_dir, offset=0.0, scale=1.0):
    # print(fp)
    playback = PyK4APlayback(fp)
    playback.open()
    info(playback)

    if offset != 0.0:
        playback.seek(int(offset * 1000000))
    fp_mp4, fp_depth_mp4, fp_depth_npy = extract(
        playback, fp, fs=30, output_dir=output_dir, scale=scale)

    playback.close()
    return fp_mp4, fp_depth_mp4, fp_depth_npy
