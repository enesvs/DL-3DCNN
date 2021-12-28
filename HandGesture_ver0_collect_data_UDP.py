
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

file = open("./HandGestureDataCollection/data_{}-{}-{}___{}-{}-{}.txt".format(now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond),"w")

data = "SampleNumber -- SensorNumber -- AccX AccY AccZ Gx Gy Gz"
file.write(data + "\n\n")

all_sensor_data = []

while True:
    data, addr = sock.recvfrom(8192) # buffer size is 1024 bytes
    # data = data.decode()
    str_data = data.hex()
    print("\n-----------------received message: {}\t\t type: {}".format(str_data, type(str_data))) # 14byte=28char gelmiş olmalı.

    AccX_L = (str_data[0:2])
    AccX_H = (str_data[2:4])
    AccY_L = (str_data[4:6])
    AccY_H = (str_data[6:8])
    AccZ_L = (str_data[8:10])
    AccZ_H = (str_data[10:12])
    AccX = int(AccX_H + AccX_L, 16)
    AccY = int(AccY_H + AccY_L, 16)
    AccZ = int(AccZ_H + AccZ_L, 16)

    GyrX_L = (str_data[12:14])
    GyrX_H = (str_data[14:16])
    GyrY_L = (str_data[16:18])
    GyrY_H = (str_data[18:20])
    GyrZ_L = (str_data[20:22])
    GyrZ_H = (str_data[22:24])
    GyrX = int(GyrX_H + GyrX_L, 16)
    GyrY = int(GyrY_H + GyrY_L, 16)
    GyrZ = int(GyrZ_H + GyrZ_L, 16)

    sampleNumber_L = (str_data[24:26])
    sampleNumber_H = (str_data[26:28])
    sampleNumber = int(sampleNumber_H + sampleNumber_L, 16)

    sensorNumber_L = (str_data[28:30])
    sensorNumber = int("00" + sensorNumber_L, 16)


    print("sampleNumber: {}\t sensorNumber: {}".format(sampleNumber, sensorNumber))
    # print("AccX_L:{} AccX_H:{}\t AccY_L:{} AccY_H:{}\t AccZ_L:{} AccZ_H:{}\t".format(AccX_L, AccX_H, AccY_L, AccY_H, AccZ_L, AccZ_H), end="")
    # print("GyrX_L:{} GyrX_H:{}\t GyrY_L:{} GyrY_H:{}\t GyrZ_L:{} GyrZ_H:{}\t".format(GyrX_L, GyrX_H, GyrY_L, GyrY_H, GyrZ_L, GyrZ_H))
    print("AccX:{} AccY:{} AccZ:{}\t\tGyrX:{} GyrY:{} GyrZ:{}".format(AccX, AccY, AccZ, GyrX, GyrY, GyrZ))



    file.write(str(sampleNumber) +"\t"+ str(sensorNumber) +"\t\t"+ str(AccX) +"\t"+ str(AccY) +"\t"+ str(AccZ) +"\t\t"+ str(GyrX) +"\t"+ str(GyrY) +"\t"+ str(GyrZ) + "\n")


    if keyboard.is_pressed('q') or (sampleNumber == 200 -1 and sensorNumber == 7-1):
        print('You Pressed A Key!')
        break  # finishing the loop

file.close()