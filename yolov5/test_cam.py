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
    print(feed_id)
    print(f"Receive data from {feed_id}: {payload}")



# encode to base64
def encodeImage(frame, ext='.jpg'):
    frame = cv2.resize(frame, (360,270))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(frame.shape)
    print(frame.dtype)
    _, im_arr = cv2.imencode(ext, frame)
    im_base64 = base64.b64encode(im_arr.tobytes())
    return im_base64


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
    while True:
        time.sleep(30)
        frame = cv2.imread('people.jpg')
        client.publish(AIO_FEED_ID['frame'], encodeImage(frame))
        print(encodeImage(frame))
