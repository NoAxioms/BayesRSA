import socket

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(('192.168.0.244', 8089))
# clientsocket.connect(('138.16.161.180', 9879))
# running = True
for i in range(10):
	clientsocket.send(bytes('hello', 'UTF-8'))