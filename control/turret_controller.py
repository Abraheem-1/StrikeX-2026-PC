import socket
import json


class TurretController:

    PI_IP = "192.168.137.7"
    PI_PORT = 5005

    def __init__(self):
        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

    def calculate_error(
        self,
        target,
        frame_width,
        frame_height
    ):

        if target is None:
            return None, None

        frame_center_x = (
            frame_width / 2
        )

        frame_center_y = (
            frame_height / 2
        )

        error_x = (
            target.center_x
            - frame_center_x
        )

        error_y = (
            target.center_y
            - frame_center_y
        )

        return (
            error_x,
            error_y
        )

    def send_to_pi(
        self,
        target,
        error_x,
        error_y
    ):

        if target is None:
            return

        packet = {
            "target_id":   int(target.track_id),
            "error_x":     float(error_x),
            "error_y":     float(error_y),
            "distance_mm": float(target.distance_mm)
        }

        self.sock.sendto(
            json.dumps(packet).encode(),
            (self.PI_IP, self.PI_PORT)
        )