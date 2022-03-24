import serial.tools.list_ports
import time
import sys, threading
from Adafruit_IO import MQTTClient
import urllib.request
import cv2
import numpy as np
import torch
from numpy import random
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import  non_max_suppression, scale_coords
from yolov5.utils.plots import plot_one_box
import base64


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
    feed_id = feed_id.split('/')[-1]
    print(f"Receive data from {feed_id}: {payload}")

    # only send data when the data is different to the state
    if feed_id == AIO_FEED_ID['buzzer'] and int(payload) != state['buzzer']:
        state['buzzer'] = int(payload)
        encoded = (feed_id+":"+ str(payload) + "#").encode()
        print(f"Sending data to sensor: {encoded}")

    if feed_id == AIO_FEED_ID['door'] and int(payload) != state['door']:
        state['door'] = int(payload)
        encoded = (feed_id+":"+ str(payload) + "#").encode()
        print(f"Sending data to sensor: {encoded}")

# utilities functions for serial data read
def getPort():
    ports = serial.tools.list_ports.comports()
    N = len(ports)
    commPort = "None"
    for i in range(0, N):
        port = ports[i]
        strPort = str(port)
        if "USB Serial Device" in strPort:
            splitPort = strPort.split(" ")
            commPort = (splitPort[0])
    return commPort


def processData(data):
    global timer_event
    data = data.replace("!", "")
    data = data.replace("#", "")
    splitData = data.split(":")
    print(f"Receive sensor infor: {splitData}")
    # touch the button
    if splitData[1] == "BUZZER":
        state['buzzer'] = 0
        client.publish(AIO_FEED_ID['buzzer'], splitData[2])
    if splitData[1] == "DOOR":
        # If the door is closed, stop the timer thread
        if splitData[2] == "0" and timer_event is not None:
            timer_event.set()
        state['door'] = int(splitData[2])
        client.publish(AIO_FEED_ID['door'], splitData[2])
    # if splitData[1] == "DOORSTAT":
    #     client.publish(AIO_FEED_ID['doorStat'], splitData[2])

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
def encodeImage64(frame, ext='jpg'):
    _, img_arr = cv2.imencode(ext, frame)
    img_base64 = base64.b64decode(img_arr.tobytes())
    return img_base64

def timer():
    global timer_event
    # wait 15 seconds or until the flag is set to True
    timer_event.wait(15)
    # if the flag is set to true, this mean the door is close before timeout
    if timer_event.isSet():
        pass
    # If timeout and nobody closes the door, ring the buzzer and turn off the relay to announce closing the door
    else:
        print("Turning on the buzzer and switch off the light ...")
        ser.write("bbc-buzzer:1#".encode())
        state["buzzer"] = 1
        client.publish(AIO_FEED_ID['buzzer'], 1)
        if state['relay'] == 1:
            ser.write("bbc-relay:0#".encode())
            state["relay"] = 0
            client.publish(AIO_FEED_ID['relay'], 0)

def listenToCamera(img_id, img_url):
    loop_cnt = 0
    timer_thread = None
    while True:
        print('Start receiving frame')
        img_resp = urllib.request.urlopen(img_url)
        imgnp = np.array(bytearray(img_resp.read()),dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)
        if frame is not None:
            n_people, processed_img = processOneFrame(frame,img_id)
            print("Number of people:", n_people)

            # publish human detection result to Adafruit
            client.publish(AIO_FEED_ID['human'], int(n_people))
            # every 30 seconds, update the frame to adafruit
            if loop_cnt % 3 == 0:
                client.publish(AIO_FEED_ID['frame'], encodeImage64(frame))

        global timer_event
        # if there is at least one people, turn on the relay (turn on the light)
        if n_people > 0 and state["relay"] == 0:
            ser.write('bbc-relay:1#'.encode())
            state["relay"] = 1
            client.publish(AIO_FEED_ID['relay'], 1)
        # if there is people, check the door condition
        elif n_people == 0 and state["door"] == 1:
            if timer_thread is None or timer_thread.isAlive() == False:
                timer_event.clear()
                timer_thread = threading.Thread(target=timer)
                timer_thread.run()
        # read a frame every 10 seconds
        time.sleep(10)
        # update loop count
        loop_cnt += 1


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
    # print(f'./images/{cnt}.png')
    cv2.imwrite(f'./images/{cnt}.png', imgRaw)
    NoPeople = len(pred)
    return NoPeople, imgRaw



if __name__ == "__main__":
    # Client info
    AIO_FEED_ID = {"human": "bbc-human",
                   "frame": "bbc-cam",
                   "relay": "lntloc/feeds/bbc-relay",
                   "buzzer": "lntloc/feeds/bbc-buzzer",
                   "door": "lntloc/feeds/bbc-door",
                   "doorStat": "lntloc/feeds/bbc-doorStat"
                   }
    AIO_USERNAME = "KanNan312"
    AIO_KEY = "aio_FshX02nlj7KAZfaQRTvweeVHqE4X"

    # Establish MQTT connections:
    client = MQTTClient(AIO_USERNAME , AIO_KEY)
    client.on_connect = connected
    client.on_disconnect = disconnected
    client.on_message = message
    client.on_subscribe = subscribe
    client.connect()
    client.loop_background()

    serialPort = getPort()
    isMircrobitConnected = False
    if serialPort is not 'None':
        isMircrobitConnected = True
        ser = serial.Serial( port=serialPort, baudrate=115200)
    mess = ""

    # Initalize YOLO model and utilities
    weights='yolov5/best.pt' # modify your weight here
    device='cpu'
    conf_thres = 0.25
    iou_thres = 0.45
    model = attempt_load(weights, map_location=device)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    img_size=600,800

    # timer event
    timer_event = threading.Event()

    # Camera set up and start thread for reading camera frames
    url = 'http://192.168.137.223/cam-hi.jpg'
    image_id = 0
    threading.Thread(target=listenToCamera, args= (image_id, url)).run()

    # initial state of the devices:
    state = {
        "door": 0,
        "relay": 0,
        "buzzer": 0
    }
    # start the main loop: reading serial data every one second
    while True:
        # read serial data
        if isMircrobitConnected:
            readSerial()
        time.sleep(1)

'''
Problem remained:
    1/ Cannot process locally (all the command (switch off, on buzzer) needs to send to adafruit and control on adafruit
    2/  
'''