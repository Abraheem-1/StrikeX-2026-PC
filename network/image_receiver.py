import socket


class ImageReceiver:
    def __init__(self, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("0.0.0.0", port))

    def receive_frame(self):
        data, _ = self.socket.recvfrom(65535)
        return data