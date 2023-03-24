import nest_asyncio
nest_asyncio.apply()
from fastapi.responses import HTMLResponse, StreamingResponse
import time
from utils.kinect import kinect_handler
import cv2
import utils.encoder.task_compiler as task_compiler
import json
from utils.azure_services import luis
from utils.azure_services import speech_synthesizer
import numpy as np
import re
import unicodedata
import os.path as osp
import os
import pickle
import uuid
import copy
from typing import List
import fastapi
from fastapi import WebSocket, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import asyncio

class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)

    async def broadcast(self, data: str):
        for connection in self.connections:
            await connection.send_text(data)


app = fastapi.FastAPI()
nlp = pipeline("feature-extraction")
manager = ConnectionManager()
kh = kinect_handler.kinect_handler(size=(320, 180))
kh.start_capturing()
threshold_taskrecognition = 0.7  # 0.2
with open('../secrets.json') as f:
    credentials = json.load(f)
blob_url = credentials["blob_url"]
luis_handler_intent = luis.luis(credentials["intent_recognizer"])
luis_handler_function = luis.luis(credentials["function_recognizer"])
luis_handler_task = luis.luis(credentials["task_recognizer"])
speech_synthesizer_azure = speech_synthesizer.speech_synthesizer(
    credentials["speech_synthesizer"], speech_synthesis_voice_name="en-US-TonyNeural")
templates = Jinja2Templates(directory="templates")
state_list = ['Idle', 'RegisterOperation', 'StartTeaching',
              'RegisterOperation_finalize', 'RegisterObject']
state_idx = 0
waiting_user_confirmation = False
state_idx_candidate = None
register_started = False
teaching_started = False
extract_sentence = False
data_save = {}
task_sequence = []
instruction_sequence = []
function_teach = None
function_candidate = None
function_idx = None
output_dir_root = '..\\out'
thread_compile = None
preview_mode = 'preview'
registerobject_name = None
registerobject_mode = 'init'


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})


@app.get('/favicon.ico')
async def favicon():
    file_name = "favicon.ico"
    file_path = os.path.join(app.root_path, "static", file_name)
    return FileResponse(path=file_path)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(sent):
    s = unicodeToAscii(sent.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\.", r"", s)
    return s


def sentence2vec(sentence):
    vec = np.array(nlp(normalizeString(sentence))[0][0])
    return vec


def load_functions():
    with open('./function_database/functions.json', 'r') as f:
        return json.loads(f.read())


def save_data():
    global data_save
    data = load_functions()
    data.append(data_save)
    with open('./function_database/functions.json', 'w') as f:
        json.dump(data, f, indent=4)


def cosdis(feature_1, feature_2):
    return np.dot(feature_1, feature_2)/(np.linalg.norm(feature_1)*np.linalg.norm(feature_2))


def get_function_candidate(input):
    functions = load_functions()
    vec_input = sentence2vec(input)
    dist = -1
    candidate = None
    sentence = None
    for i in functions:
        for sent in i['verbal_commands']:
            tmp_dist = cosdis(vec_input, sentence2vec(sent))
            if tmp_dist > dist:
                dist = tmp_dist
                candidate = i
                sentence = sent
    return candidate, sentence


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


async def notify(message):
    global state_idx
    await manager.broadcast(f";ROBOT_TALKING_START")
    speech_synthesizer_azure.synthesize_speech(message.split(';')[0])
    await manager.broadcast(f";ROBOT_TALKING_FINISH")
    await manager.broadcast(f"Robot [{state_list[state_idx]}]: " + message)


async def interface(user_input):
    global state_idx
    global waiting_user_confirmation
    global state_idx_candidate
    global register_started
    global task_sequence
    global instruction_sequence
    global data_save
    user_input = user_input.replace('.', '').lower()
    if user_input == 'reset':
        state_idx = 0
        return "Reset to the idling mode. What can I do for you today?"
    if state_list[state_idx] == "Idle":
        query = user_input
        intent, _ = luis_handler_intent.analyze_input(query)
        print(intent)
        if waiting_user_confirmation:
            if intent == 'Confirm':
                state_idx = state_idx_candidate
                waiting_user_confirmation = False
            elif intent == 'Cancel':
                waiting_user_confirmation = False
                return "Hello. What can I do for you today?"
            else:
                return "I'm sorry, I don't understand. Please rephrase."
        else:
            if intent == 'RegisterOperation':
                state_idx_candidate = 1
                waiting_user_confirmation = True
                return f"It seems that your intention is: {intent}. Is that correct?"
            elif intent == 'StartTeaching':
                state_idx_candidate = 2
                waiting_user_confirmation = True
                return f"It seems that your intention is: {intent}. Is that correct?"
            elif intent == 'RegisterObject':
                state_idx_candidate = 4
                waiting_user_confirmation = True
                return f"It seems that your intention is: {intent}. Is that correct?"
            else:
                return f"I'm sorry, I don't understand. Please rephrase."

    if state_list[state_idx] == "RegisterOperation":

        if register_started == False:
            register_started = True
            task_sequence = []
            instruction_sequence = []
            return "Please give the instruction step-by-step. When teaching finishes, input 'finish'"
        else:
            intent, _ = luis_handler_intent.analyze_input(user_input)
            if waiting_user_confirmation:  # check the final sequence
                if intent == 'Confirm':
                    string = ''
                    data_save["task_cohesion"] = {}
                    data_save["task_cohesion"]["task_sequence"] = task_sequence
                    data_save["task_cohesion"]["step_instructions"] = instruction_sequence
                    for i in task_sequence:
                        string += f"<{i}>"
                    waiting_user_confirmation = False
                    state_idx = 3
                    task_sequence = []
                    instruction_sequence = []
                    register_started = False
                    return f"Registered the sequence.\n What is the object name?"
                elif intent == 'Cancel':
                    waiting_user_confirmation = False
                    state_idx = 0
                    task_sequence = []
                    instruction_sequence = []
                    register_started = False
                    return f"Cancelling registration. What can I do for you today?"
                else:
                    return f"I'm sorry, I don't understand. Please rephrase."
            elif user_input == 'finish':
                if len(task_sequence) == 0:
                    register_started = False
                    state_idx = 0
                    task_sequence = []
                    instruction_sequence = []
                    return f"No task in the sequence. Cancelling registration. What can I do for you today?"
                else:
                    string = ''
                    for i in task_sequence:
                        string += f"<{i}>"
                    waiting_user_confirmation = True
                    return f"Registering a sequence: {string}. Is that correct?"
            elif user_input == 'cancel':
                register_started = False
                state_idx = 0
                task_sequence = []
                instruction_sequence = []
                return "Cancel requested. What can I do for you today?"
            else:
                # task recognition
                label, confidence = luis_handler_task.analyze_input(user_input)
                if confidence < threshold_taskrecognition:
                    return "I'm sorry, I don't understand. Please rephrase."
                else:
                    task_sequence.append(label)
                    instruction_sequence.append(user_input)
                    return f"the task is <{label}>"

    if state_list[state_idx] == "RegisterOperation_finalize":
        print(data_save)
        if "object_name" not in data_save["task_cohesion"]:
            data_save["task_cohesion"]["object_name"] = user_input
            return f"Registered the objectname:{user_input}. How do you describe the task sequence in a sentence?"
        if "verbal_commands" not in data_save:
            data_save["verbal_commands"] = [user_input]
            save_data()
            data_save = {}
            state_idx = 0
            return f"Registered the sentence:{user_input}. What can I do for you today?"

    if state_list[state_idx] == "StartTeaching":
        global teaching_started
        global extract_sentence
        global function_teach
        global function_candidate
        global function_idx
        global output_dir_root
        global output_dir
        global thread_compile
        if teaching_started == False:
            teaching_started = True
            extract_sentence = True
            return "What should I do?"
        elif extract_sentence:
            function_candidate, sentence = get_function_candidate(user_input)
            waiting_user_confirmation = True
            extract_sentence = False
            return f"Found a closest function: '{sentence}'. Is that correct?"

        if waiting_user_confirmation:
            intent, _ = luis_handler_intent.analyze_input(user_input)
            if intent == 'Confirm':
                waiting_user_confirmation = False
                function_idx = 0
                function_teach = function_candidate
            elif intent == 'Cancel':
                waiting_user_confirmation = False
                state_idx = 0
                teaching_started = False
                return "Please register the cohesion. Reset to the idling mode. What can I do for you today?"
            else:
                return "I'm sorry, I don't understand. Please rephrase."

        if user_input == "[button-kinect-stoprecording]":
            kh.stop_recording()
            function_idx += 1
            await notify("Recording stopped")

        if function_idx >= len(function_teach['task_cohesion']['task_sequence']):
            print(function_teach['task_cohesion']['task_sequence'])
            await notify("Compiling task model...")
            await compile(function_teach['task_cohesion']['task_sequence'], function_teach['task_cohesion']['step_instructions'], function_teach['task_cohesion']['object_name'], output_dir)
            print('uploading to blob storage')
            default_credential = DefaultAzureCredential(
                exclude_environment_credentials=True, exclude_shared_token_cache_credential=True)
            # Create the BlobServiceClient object
            blob_service_client = BlobServiceClient(
                blob_url, credential=default_credential)
            container_name = 'storage'
            blob_client = blob_service_client.get_blob_client(
                container=container_name, blob='task_models.json')
            with open(file=osp.join(output_dir, 'task_models.json'), mode="rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            state_idx = 0
            teaching_started = False
            return "Compile Done. What can I do for you today?"
        else:
            if function_idx == 0:
                # Prepare a new output directory
                output_dir = osp.join(
                    output_dir_root, function_teach["task_cohesion"]['object_name']+'_'+str(uuid.uuid4()))
                instruction = function_teach['task_cohesion']['step_instructions'][function_idx]
                task = function_teach['task_cohesion']['task_sequence'][function_idx]
                kh.start_recording(osp.join(output_dir, str(
                    function_idx) + '_' + task),  fp_base=str(function_idx) + '_' + task, save_fps=10)
                return f"Please do step-by-step:{instruction}. Recording started."
            else:
                instruction = function_teach['task_cohesion']['step_instructions'][function_idx]
                task = function_teach['task_cohesion']['task_sequence'][function_idx]
                kh.start_recording(osp.join(output_dir, str(
                    function_idx) + '_' + task),  fp_base=str(function_idx) + '_' + task, save_fps=10)
                return f"Please do the next:{instruction}. Recording started."

    if state_list[state_idx] == "RegisterObject":  # teach
        global preview_mode
        global registerobject_name
        global registerobject_mode  # 'ask_object', 'capture_image'
        preview_mode = 'registerobject'

        if registerobject_mode == 'init':
            registerobject_mode = 'ask_object'
            return 'Enter the object name'
        if registerobject_mode == 'ask_object':
            registerobject_name = user_input
            registerobject_mode = 'capture_image'
            return f"Start registering: {registerobject_name}. Please capture the image in the center of the camera;REGISTER_OBJECT"
        if registerobject_mode == 'capture_image':
            if user_input == '[button-capture]':
                kh.save_image(registerobject_name)
                return 'Capture button pressed;REGISTER_OBJECT'
            if user_input == '[button-capture-finish]':
                state_idx = 0
                registerobject_mode = 'init'
                registerobject_name = None
                preview_mode = 'preview'
                return 'Capture mode finished. What can I do for you today?'
        else:
            return 'Please capture the image in the center of the camera;REGISTER_OBJECT'


@app.websocket("/ws/user")
async def websocket_endpoint(websocket: WebSocket):
    print('socket came')
    await manager.connect(websocket)
    while True:
        print('waiting input...')
        data = await websocket.receive_text()
        if not data.startswith('['):  # "[command]" is an internal command
            await manager.broadcast(f"User: {data}")
        agent_return = await interface(data)
        if agent_return is not None:
            # inhibit the recognition and speech
            await manager.broadcast(f";ROBOT_TALKING_START")
            speech_synthesizer_azure.synthesize_speech(
                agent_return.split(';')[0])
            time.sleep(1)
            await manager.broadcast(f";ROBOT_TALKING_FINISH")
            if ';' not in agent_return:
                agent_return = agent_return+";"
            print('sending...')
            await manager.broadcast(f"Robot [{state_list[state_idx]}]: {agent_return}")
        else:
            pass


def gen_frames():  # generate frame by frame from camera
    global preview_mode
    last_frame = 255*np.ones(shape=[5, 640, 3], dtype=np.uint8)
    while True:
        try:
            if preview_mode == 'preview':
                frame = kh.get_frame(concat='horizontal')
            if preview_mode == 'registerobject':
                frame = kh.get_frame(concat='horizontal', frameguide=True)
            if frame is None:
                print('empty frame came')
                frame = last_frame
        except:
            frame = last_frame
        last_frame = copy.deepcopy(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        # print(len(frame))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.get('/video_feed', response_class=HTMLResponse)
async def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return StreamingResponse(gen_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')
