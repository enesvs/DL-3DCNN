
import keyboard

import socket
socket.gethostbyname(socket.gethostname())
print(socket.gethostbyname(socket.gethostname()))


import socket

UDP_IP = "192.168.1.20"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))


import datetime
now = datetime.datetime.now()

file = open("./HandGestureDataCollection/data_{}-{}-{}___{}-{}.txt".format(now.year, now.month, now.day, now.hour, now.minute, now.second),"w")

data = "SensorNumber -- SampleNumber -- AccX AccY AccZ Gx Gy Gz"
file.write(data + "\n\n")

while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    data = data.decode()
    print("received message: {}\t\t type: {}".format(data, type(data)))
    file.write(data + "\n")
    if keyboard.is_pressed('q'):  # if key 'q' is pressed
        print('You Pressed A Key!')
        break  # finishing the loop

file.close()