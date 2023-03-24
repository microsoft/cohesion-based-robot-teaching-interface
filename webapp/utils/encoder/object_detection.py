import json
import cv2
import requests
import os
import tempfile


def upload_data_image(fp_tmp_img, object_name):
    url = 'http://DOMAIN:PORT/object_localization_image'
    headers = {'accept': 'application/json'}
    files = {'upload_file': open(fp_tmp_img, 'rb')}
    values = {'object_name': object_name}
    response = requests.post(url, headers=headers,
                             files=files, data=values)
    return response


def run_image(frame_img, object_name):
    with tempfile.TemporaryDirectory() as output_dir_tmp:
        fp_tmp_img = os.path.join(output_dir_tmp, 'tmp.png')
        cv2.imwrite(fp_tmp_img, frame_img)

        fp_tmp_out = os.path.join(output_dir_tmp, 'tmp.json')
        response = upload_data_image(fp_tmp_img, object_name)
        data = response.content
        with open(fp_tmp_out, 'wb') as s:
            s.write(data)
        with open(fp_tmp_out) as json_file:
            data = json.load(json_file)
            return data
