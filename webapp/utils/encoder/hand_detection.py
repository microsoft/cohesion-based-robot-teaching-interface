from ftplib import all_errors
import json
from time import time
from unicodedata import name
import cv2
import requests
import os
import tempfile
import cv2

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)


def upload_data(mp4path, jsonpath):
    url = 'http://DOMAIN:PORT/hand_localization'
    headers = {'accept': 'application/json'}
    data = {
        'upload_file': open(
            mp4path, 'rb'), 'upload_json': open(
            jsonpath, 'rb')}
    response = requests.post(url, headers=headers,
                             files=data)
    #data = response.data()
    return response


def upload_data_image(fp_tmp_img):
    url = 'http://DOMAIN:PORT/hand_localization_image'
    headers = {'accept': 'application/json'}
    data = {'upload_file': open(fp_tmp_img, 'rb')}
    response = requests.post(url, headers=headers,
                             files=data)
    #data = response.data()
    return response


def run(time_focus, mp4path, output_dir):
    import os
    fp_tmp_json = os.path.join(output_dir, 'tmp.json')
    fp_tmp_out = os.path.join(output_dir, 'tmp.zip')
    json_send = {}
    json_send['time_focus'] = time_focus
    # save the result
    with open(fp_tmp_json, 'w') as outfile:
        json.dump(json_send, outfile, indent=4)
    # ask server to convert image to depth
    response = upload_data(mp4path, fp_tmp_json)
    data = response.content
    # save the result
    with open(fp_tmp_out, 'wb') as s:
        s.write(data)

    print(fp_tmp_out)
    import shutil
    try:
        shutil.unpack_archive(
            fp_tmp_out, os.path.join(
                output_dir, 'hand_detection'))
        from glob import glob
        import zipfile
        zip_f = zipfile.ZipFile(fp_tmp_out, 'r')
        lst = zip_f.namelist()
        relative_path = ""
        for item in lst:
            if "hand_detection.json" in item:
                relative_path = item
        return os.path.normpath(
            os.path.join(
                output_dir,
                'hand_detection',
                relative_path))
    except BaseException:
        print('unpack failed (hand_detection.py)')
    # find the result file


def run_allframe(mp4path, json_send, output_dir):
    with tempfile.TemporaryDirectory() as output_dir_tmp:
        import os
        fp_tmp_json = os.path.join(output_dir_tmp, 'tmp.json')
        fp_tmp_out = os.path.join(output_dir, 'tmp.zip')
        # save the result
        with open(fp_tmp_json, 'w') as outfile:
            json.dump(json_send, outfile, indent=4)
        # ask server to convert image to depth
        response = upload_data(mp4path, fp_tmp_json)
        data = response.content
        # save the result
        with open(fp_tmp_out, 'wb') as s:
            s.write(data)
        print(fp_tmp_out)
        import shutil
        try:
            shutil.unpack_archive(
                fp_tmp_out, os.path.join(
                    output_dir, 'hand_allframe'))
            import zipfile
            zip_f = zipfile.ZipFile(fp_tmp_out, 'r')
            lst = zip_f.namelist()
            relative_path = ""
            for item in lst:
                if "hand_detection.json" in item:
                    relative_path = item
            fp_json = os.path.join(
                output_dir,
                'hand_allframe',
                relative_path)
            with open(fp_json) as json_file:
                data = json.load(json_file)
                return data
        except BaseException:
            print('unpack failed (hand_detection.py)')
        # find the result file


def run_image(frame_img):
    with tempfile.TemporaryDirectory() as output_dir:
        fp_tmp_img = os.path.join(output_dir, 'tmp.png')
        cv2.imwrite(fp_tmp_img, frame_img)

        fp_tmp_out = os.path.join(output_dir, 'tmp.json')
        response = upload_data_image(fp_tmp_img)
        data = response.content
        with open(fp_tmp_out, 'wb') as s:
            s.write(data)
        with open(fp_tmp_out) as json_file:
            data = json.load(json_file)
            return data
