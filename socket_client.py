import socket

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(('138.16.161.180', 9879))
clientsocket.send(bytes('hello', 'UTF-8'))