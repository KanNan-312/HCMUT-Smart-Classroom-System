import threading
import time

def Task():
    event.wait(10)
    if event.isSet():
        print('Set')
    else:
        print('Time out')
    print('Sub thread')

def Caller():
    print('Calling')
    # threading.Thread(target=Task).start()
    cnt = 0
    while cnt < 100000000:
        cnt+=1
    print('Calling done')

event = threading.Event()
caller_thread = None
cnt = 1
while True:
    if caller_thread is None or not caller_thread.is_alive():
        caller_thread = threading.Thread(target=Caller)
        caller_thread.start()
    print('Main thread')
    time.sleep(2)