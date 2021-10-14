import socket
socket.gethostbyname(socket.gethostname())
print(socket.gethostbyname(socket.gethostname()))


import socket

TCP_IP = "192.168.1.20"
TCP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP
sock.bind((TCP_IP, TCP_PORT))

sock.listen(1)

conn, addr = sock.accept()
print('Connection address:', addr)

while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    print("received message: {} from {}".format(data, addr))