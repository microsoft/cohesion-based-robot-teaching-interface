import json
import requests
import os


def upload_data(fp_audio_json, fp_segmentation):
    url = 'http://DOMAIN:PORT/audio_split_and_speech_recognition'
    headers = {'accept': 'application/json'}
    data = {'upload_file': open(fp_audio_json, 'rb'),
            'upload_json': open(fp_segmentation, 'rb')}
    response = requests.post(url, headers=headers,
                             files=data)
    return response


def run(fp_audio_json, fp_segmentation, output_dir):
    fp_tmp_zip = os.path.join(output_dir, 'speech_recognized.zip')
    response = upload_data(fp_audio_json, fp_segmentation)
    data = response.content
    with open(fp_tmp_zip, 'wb') as s:
        s.write(data)
    import shutil
    try:
        shutil.unpack_archive(fp_tmp_zip, fp_tmp_zip.replace('.zip', ''))
        from glob import glob
        files = []
        for dir, _, _ in os.walk(
            os.path.join(
                fp_tmp_zip.replace(
                '.zip', ''))):
            files.extend(glob(os.path.join(dir, "transcript.json")))
        return os.path.normpath(files[0])
    except BaseException:
        print('unpack failed')
