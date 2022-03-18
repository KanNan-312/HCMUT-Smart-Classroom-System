import serial.tools.list_ports
import time
import sys
from Adafruit_IO import MQTTClient

AIO_FEED_ID = ["bbc-relay", 'bbc-buzzer', 'bbc-door', 'bbc-doorStat']
AIO_USERNAME = ""
AIO_KEY = ""

def connected(client):
    print("Ket noi thanh cong...")
    for i in AIO_FEED_ID: client.subscribe(i)

def subscribe(client , userdata , mid , granted_qos):
    print("Subcribe thanh cong...")

def disconnected(client):
    print("Ngat ket noi...")
    sys.exit (1)

def message(client , feed_id , payload):
    print(feed_id)
    print("Nhan du lieu: " + payload)
    print((feed_id+":"+ str(payload) + "#").encode())
    ser.write((feed_id+":"+ str(payload) + "#").encode())


client = MQTTClient(AIO_USERNAME , AIO_KEY)
client.on_connect = connected
client.on_disconnect = disconnected
client.on_message = message
client.on_subscribe = subscribe
client.connect()
client.loop_background()

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

ser = serial.Serial( port=getPort(), baudrate=115200)

mess = ""
def processData(data):
    data = data.replace("!", "")
    data = data.replace("#", "")
    splitData = data.split(":")
    print(splitData)
    if splitData[1] == "BUZZER":
        client.publish("bbc-buzzer", splitData[2])
    if splitData[1] == "DOOR":
        client.publish("bbc-door", splitData[2])
    if splitData[1] == "DOORSTAT":
        client.publish("bbc-doorStat", splitData[2])

mess = ""
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

while True:
    readSerial()
    time.sleep(1)