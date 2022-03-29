import serial.tools.list_ports
import time
import sys, threading
from Adafruit_IO import MQTTClient
import urllib.request
import cv2
import numpy as np
import torch
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import  non_max_suppression, scale_coords
from utils.plots import plot_one_box
import base64
import json

# callback functions
# Adafruit client utilities
def connected(client):
    print("Connect successfully...")
    for feed_id in AIO_FEED_ID:
        client.subscribe(AIO_FEED_ID[feed_id])

def subscribe(client , userdata , mid , granted_qos):
    print("Subcribe successfully...")

def disconnected(client):
    print("Disconnecting...")
    sys.exit (1)

def message(client , feed_id , payload):
    # feed_id = feed_id.split('/')[-1]
    print(f"Receive data from {feed_id}: {payload}")
    # Initial state:
    global init
    global state
    if init:
        if feed_id == AIO_FEED_ID['buzzer']:
            state['buzzer'] = int(payload)
        elif feed_id == AIO_FEED_ID['human']:
            state['human'] = int(payload)
        elif feed_id == AIO_FEED_ID['relay']:
            state['relay'] = int(payload)
        elif feed_id == AIO_FEED_ID['door']:
            state['door'] = int(payload)
        return

    # only send data when the data is different to the state
    if feed_id == AIO_FEED_ID['buzzer'] and int(payload) != state['buzzer']:
        state['buzzer'] = int(payload)
        encoded = (feed_id+":"+ str(payload) + "#").encode()
        if isMircrobitConnected:
            print(f"Sending data to sensor: {encoded}")
            ser.write(encoded)

    if feed_id == AIO_FEED_ID['relay'] and int(payload) != state['relay']:
    # if feed_id == AIO_FEED_ID['relay']:
        state['relay'] = int(payload)
        encoded = (feed_id+":"+ str(payload) + "#").encode()
        if isMircrobitConnected:
            print(f"Sending data to sensor: {encoded}")
            ser.write(encoded)


# utilities functions for serial data read
def getPort():
    ports = serial.tools.list_ports.comports()
    N = len(ports)
    commPort = "None"
    for i in range(0, N):
        port = ports[i]
        strPort = str(port)
        print(strPort)
        if "USB Serial Device" in strPort:
            splitPort = strPort.split(" ")
            commPort = (splitPort[0])
    return commPort


def processData(data):
    global timer_event
    data = data.replace("!", "")
    data = data.replace("#", "")
    splitData = data.split(":")
    if len(splitData) != 3 or splitData[2] == "":
        return
    # touch the button, only accept the data when the state is changed
    if splitData[1] == "BUZZER" and int(splitData[2]) != state['buzzer']:
        print(f"Receive buzzer sensor data: {splitData}")
        state['buzzer'] = 0
        client.publish(AIO_FEED_ID['buzzer'], splitData[2])
    if splitData[1] == "DOOR" and int(splitData[2]) != state['door']:
        print(f"Receive door sensor data: {splitData}")
        # If the door is closed, stop the timer thread
        if splitData[2] == "0" and timer_event is not None:
            timer_event.set()
        state['door'] = int(splitData[2])
        client.publish(AIO_FEED_ID['door'], splitData[2])


def readSerial():
    bytesToRead = ser.inWaiting()
    if (bytesToRead > 0):
        global mess
        mess = mess + ser.read(bytesToRead).decode("UTF-8")
        while ("#" in mess) and ("!" in mess):
            start = mess.find("!")
            end = mess.find("#")
            processData(mess[start:end + 1])
            if (end == len(mess)):
                mess = ""
            else:
                mess = mess[end+1:]


# camera utility
def encodeImage64(frame, ext='.jpg'):
    # resize and encode image to base 64
    frame = cv2.resize(frame, (360,270))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, img_arr = cv2.imencode(ext, frame)
    img_base64 = base64.b64decode(img_arr.tobytes())
    return img_base64

def timer():
    # print("Thread time")
    global timer_event
    # wait 15 seconds or until the flag is set to True
    print("Start counting...")
    timer_event.wait(15)
    # if the flag is set to true, this mean the door is close before timeout
    if timer_event.isSet():
        pass
    # If timeout and nobody closes the door, ring the buzzer and turn off the relay to announce closing the door
    else:
        print("Turning on the buzzer and switch off the light ...")
        if isMircrobitConnected and state['buzzer'] == 0:
            print(f"Sending data to sensor: bbc-buzzer:1#")
            ser.write("bbc-buzzer:1#".encode())
            #  update the current state
            state["buzzer"] = 1
            client.publish(AIO_FEED_ID['buzzer'], "1")
        if isMicrobitConnected and state['relay'] == 1:
            print(f"Sending data to sensor: bbc-relay:0#")
            ser.write("bbc-relay:0#".encode())
            state["relay"] = 0
            client.publish(AIO_FEED_ID['relay'], 0)

def listenToCamera(img_id, img_url):
    time.sleep(10)
    img_resp = urllib.request.urlopen(img_url)
    imgnp = np.array(bytearray(img_resp.read()),dtype=np.uint8)
    frame = cv2.imdecode(imgnp, -1)
    size = frame.shape
    print(f'Frame received: {size}')
    # frame = cv2.imread('people2.jpg')
    if frame is not None:
        n_people, processed_img = processOneFrame(frame,img_id)
        print("Number of people detected:", n_people)
        if n_people != state['human']:
            # publish human detection result to Adafruit
            client.publish(AIO_FEED_ID['human'], int(n_people))

        # every 30 seconds, update the frame to adafruit
        if img_id % 3 == 0:
            client.publish(AIO_FEED_ID['frame'], encodeImage64(frame))

    global timer_thread
    global timer_event
    # if there is at least one people, turn on the relay (turn on the light system)
    if n_people > 0 and state["relay"] == 0:
        print('There is people. Turning light on ...')
        ser.write('bbc-relay:1#'.encode())
        state["relay"] = 1
        client.publish(AIO_FEED_ID['relay'], '1')

    # if there is people and the door is opened, start the timer thread
    elif n_people == 0 and state["door"] == 1:
        if timer_thread is None or timer_thread.is_alive() == False:
            timer_event.clear()
            timer_thread = threading.Thread(target=timer)
            timer_thread.start()

# def listenToCamera(img_id, img_url):
#     loop_cnt = 0
#     timer_thread = None
#     print('Start receiving frame')
#     while True:
#         # print('Thread camera')
#         img_resp = urllib.request.urlopen(img_url)
#         imgnp = np.array(bytearray(img_resp.read()),dtype=np.uint8)
#         frame = cv2.imdecode(imgnp, -1)
#         # frame = cv2.imread('people2.jpg')
#         if frame is not None:
#             n_people, processed_img = processOneFrame(frame,img_id)
#             print("Number of people:", n_people)

#             # publish human detection result to Adafruit
#             client.publish(AIO_FEED_ID['human'], int(n_people))
#             # every 30 seconds, update the frame to adafruit
#             # if loop_cnt % 3 == 0:
#                 # client.publish(AIO_FEED_ID['frame'], encodeImage64(frame))

#         global timer_event
#         # if there is at least one people, turn on the relay (turn on the light)
#         if n_people > 0 and state["relay"] == 0:
#             print('There is people. Turning light on ...')
#             ser.write('bbc-relay:1#'.encode())
#             state["relay"] = 1
#             client.publish(AIO_FEED_ID['relay'], 1)
#         # if there is people, check the door condition
#         elif n_people == 0 and state["door"] == 1:
#             if timer_thread is None or timer_thread.is_alive() == False:
#                 timer_event.clear()
#                 timer_thread = threading.Thread(target=timer)
#                 timer_thread.start()
#         # read a frame every 10 seconds
#         time.sleep(10)
#         # update loop count
#         loop_cnt += 1


# Model related functions
def processOneFrame(img0, cnt):  # a numpy array read by cv2 framework
    imgRaw = img0.copy()
    img = letterbox(imgRaw, img_size, stride=int(model.stride.max()))[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=True)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)[0]
    pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()

    for *xyxy, conf, cls in pred:
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, imgRaw, label=label, color=colors[int(cls)], line_thickness=3)
    
    cv2.imwrite(f'./images/{cnt}.png', imgRaw)
    NoPeople = len(pred)
    return NoPeople, imgRaw

def fetchInit():
    global client
    client.receive('bbc-door')
    client.receive('bbc-relay')
    client.receive('bbc-buzzer')
    client.receive('bbc-human')

def logState():
    global state
    for feed_id in state:
        print(f'{feed_id}: {state[feed_id]}')

if __name__ == "__main__":
    # Client info
    with open("adafruit_key.json", "r") as f:
        api_key = json.load(f)
    
    AIO_FEED_ID = {"human": "bbc-human",
                   "frame": "bbc-cam",
                   "relay": "bbc-relay",
                   "buzzer": "bbc-buzzer",
                   "door": "bbc-door"
                   }

    AIO_USERNAME = str(api_key['Username'])
    AIO_KEY = str(api_key['Key'])

    # shared variable:
    mutex = threading.Lock()
    # State dict of devices
    init = True
    state = {}

    # Establish MQTT connections:
    client = MQTTClient(AIO_USERNAME , AIO_KEY)
    client.on_connect = connected
    client.on_disconnect = disconnected
    client.on_message = message
    client.on_subscribe = subscribe
    client.connect()
    client.loop_background()

    time.sleep(5)
    # fetch initial data and store to state dict
    fetchInit()

    # wait until all data has been fetched
    while len(state) != 4:
        continue
    init = False
    print('Initial state:')
    logState()

    # Connect to serial port
    serialPort = getPort()
    isMircrobitConnected = False
    if serialPort != "None":
        isMircrobitConnected = True
        ser = serial.Serial(port=serialPort, baudrate=115200)
    print(f'Connected to serial port: {serialPort}')
    # print(isMircrobitConnected)
    mess = ""

    # Initalize YOLO model and utilities
    weights='crowdhuman_yolov5m.pt' # modify your weight here
    device='cpu'
    conf_thres = 0.25
    iou_thres = 0.45
    model = attempt_load(weights, map_location=device)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # img_size=600,800
    img_size = 640,640
    
    # timer event
    timer_event = threading.Event()
    timer_thread = None

    # Camera set up and start thread for reading camera frames
    url = 'http://192.168.137.117/cam-hi.jpg'
    image_id = 0
    # camera_thread = threading.Thread(target=listenToCamera, args= (image_id, url))
    # camera_thread.start()

    # start the main loop: reading serial data every one second
    
    camera_thread = None
    while True:
        # read serial data
        if isMircrobitConnected == True:
            # print("Reading Sensor Data...")
            readSerial()
        # Reading camera
        if camera_thread is None or camera_thread.is_alive() == False:
            threading.Thread(target=listenToCamera, args=(image_id, url)).run()
            image_id += 1

        # time.sleep(2)
