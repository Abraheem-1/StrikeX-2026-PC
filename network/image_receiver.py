import socket
import struct

from network.protocol import HEADER_FORMAT
from network.frame_reassembler import FrameReassembler


class ImageReceiver:

    def __init__(self, port):

        self.socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        self.socket.bind(("0.0.0.0", port))

        self.reassembler = FrameReassembler()

    def receive_frame(self):

        while True:

            packet, _ = self.socket.recvfrom(65535)

            header = packet[:9]
            payload = packet[9:]

            (
                camera_id,
                frame_id,
                packet_index,
                packet_count
            ) = struct.unpack(
                HEADER_FORMAT,
                header
            )

            jpeg_bytes = self.reassembler.add_packet(
                frame_id,
                packet_index,
                packet_count,
                payload
            )

            if jpeg_bytes is not None:
                return jpeg_bytes