import socket
import json

PI_IP = "192.168.137.7"
PI_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

packet = {
    "target_id": 1,
    "error_x": 150,
    "error_y": -42,
    "distance_mm": 3000
}

sock.sendto(
    json.dumps(packet).encode(),
    (PI_IP, PI_PORT)
)

print("Packet sent")