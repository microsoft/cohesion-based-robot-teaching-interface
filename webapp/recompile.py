import json
import asyncio
import os.path as osp
import os
import PySimpleGUI as sg
import nest_asyncio
import pickle
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import utils.encoder.task_compiler as task_compiler
nest_asyncio.apply()


def get_file_info():
    layout = [
        [
            sg.FolderBrowse(
                font='Helvetica 14'), sg.Text(
                "Folder name", font='Helvetica 14'), sg.InputText(
                font='Helvetica 14')], [
            sg.Submit(
                key="submit", font='Helvetica 14'), sg.Cancel(
                "Exit", font='Helvetica 14')]]

    window = sg.Window("file selection", layout, location=(800, 400))

    while True:
        event, values = window.read(timeout=100)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        elif event == 'submit':
            if values[0] == "":
                sg.popup("Enter file information")
                continue
            else:
                fp_dir = values[0]
                break
    window.close()
    return fp_dir


with open('../secrets.json') as f:
    credentials = json.load(f)
blob_url = credentials["blob_url"]


def load_functions():
    with open('./function_database/functions.json', 'r') as f:
        return json.loads(f.read())


def save_data():
    global data_save
    data = load_functions()
    data.append(data_save)
    with open('./function_database/functions.json', 'w') as f:
        json.dump(data, f, indent=4)


def compile_task(
        task,
        verbal_input,
        object_name,
        fp_mp4,
        fp_depth_npy,
        output_dir_daemon,
        hand_laterality,
        id):
    daemon = task_compiler.task_daemon(
        task,
        verbal_input,
        object_name,
        fp_mp4,
        fp_depth_npy,
        output_dir_daemon,
        hand_laterality=hand_laterality,
        id=id)
    tmp = id + '_' + task

    daemon.set_skillparameters()
    print(f'Setting skill paramers:{tmp}: done')
    daemon.dump_json()
    return daemon


async def run_daemon(loop,
                     task_list,
                     verbal_input_list,
                     object_name,
                     fp_mp4_list,
                     fp_depth_npy_list,
                     output_dir_daemon,
                     hand_laterality,
                     id_list):
    sem = asyncio.Semaphore(10)
    async def run_request(task, verbal_input, object_name, fp_mp4,
                          fp_depth_npy, output_dir_daemon, hand_laterality, id):
        async with sem:
            return await loop.run_in_executor(None, compile_task,
                                              task,
                                              verbal_input,
                                              object_name,
                                              fp_mp4,
                                              fp_depth_npy,
                                              output_dir_daemon,
                                              hand_laterality,
                                              id)
    damon_list = [
        run_request(
            task_list[i],
            verbal_input_list[i],
            object_name,
            fp_mp4_list[i],
            fp_depth_npy_list[i],
            output_dir_daemon[i],
            hand_laterality,
            id_list[i]) for i in range(
            len(task_list))]
    return await asyncio.gather(*damon_list)


async def compile(task_list, verbal_input_list, object_name, output_dir):
    hand_laterality = 'right'
    output_dir_daemon = output_dir
    output_dir_daemon_list = [osp.join(output_dir, str(
        i) + '_' + fp_base) for i, fp_base in enumerate(task_list)]
    verbal_input_list = verbal_input_list
    object_name = object_name
    fp_mp4_list = [osp.join(output_dir_daemon, str(
        i) + '_' + fp_base, str(i) + '_' + fp_base + ".mp4") for i, fp_base in enumerate(task_list)]
    fp_depth_npy_list = [osp.join(output_dir_daemon, str(i) + '_' + fp_base, str(i) + '_' + fp_base + "_depth.npz")
                         for i, fp_base in enumerate(task_list)]
    id_list = [str(i) for i in range(len(task_list))]
    loop = asyncio.get_event_loop()
    print('running daemon')
    daemons = loop.run_until_complete(
        run_daemon(
            loop,
            task_list,
            verbal_input_list,
            object_name,
            fp_mp4_list,
            fp_depth_npy_list,
            output_dir_daemon_list,
            hand_laterality,
            id_list))

    fp_daemon = os.path.join(output_dir_daemon, "daemons.pkl")
    with open(fp_daemon, 'wb') as f:
        pickle.dump(daemons, f)
    dump_json(output_dir)
    print('done')


def dump_json(output_dir):
    fp_daemon = os.path.join(output_dir, "daemons.pkl")
    with open(fp_daemon, 'rb') as f:
        daemons = pickle.load(f)
    task_models = []
    for daemon in daemons:
        task_models.append(daemon.taskmodel_json)
    task_models_save = []
    for i, item in enumerate(task_models):
        if i > 0 and item["_task"] == "GRASP":
            item_pre = task_models[i - 1]
            item["prepre_grasp_position"]["value"] = item_pre["start_position"]["value"]
    for item in task_models:
        task_models_save.append(item)

    task_models_save.append({"_task": "END"})

    # save the task sequence
    task_models_save_json = {}
    task_models_save_json["version"] = "1.0"
    task_models_save_json["task_models"] = task_models_save
    fp_task_sequence = os.path.join(output_dir, "task_models.json")
    with open(fp_task_sequence, 'w') as f:
        json.dump(task_models_save_json, f, indent=4)


async def interface():
    data = load_functions()
    _output_dir = get_file_info()
    task_list_with_order = [name for name in os.listdir(
        _output_dir) if os.path.isdir(os.path.join(_output_dir, name))]
    task_list = [item.split('_')[1] for item in task_list_with_order]
    _function_teach = None
    for item in data:
        if item['task_cohesion']['task_sequence'] == task_list and item['task_cohesion']['object_name'] == _output_dir.split('/')[-1].split('_')[0]:
            _function_teach = item
    if _function_teach is None:
        raise Exception('no task cohesion found')
    await compile(_function_teach['task_cohesion']['task_sequence'],
                  _function_teach['task_cohesion']['step_instructions'],
                  _function_teach['task_cohesion']['object_name'], _output_dir)

    # upload to blob
    print('uploading to blob storage')
    default_credential = DefaultAzureCredential(
        exclude_environment_credentials=True, exclude_shared_token_cache_credential=True)
    blob_service_client = BlobServiceClient(
        blob_url, credential=default_credential)
    container_name = 'storage'
    filename = 'latest.json'
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=filename)
    with open(file=osp.join(_output_dir, 'task_models.json'), mode="rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print('done!!!')

if __name__ == '__main__':
    asyncio.run(interface())
