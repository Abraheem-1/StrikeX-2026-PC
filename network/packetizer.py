import struct

from network.protocol import (
    HEADER_FORMAT,
    PAYLOAD_SIZE
)


class Packetizer:

    @staticmethod
    def packetize(camera_id, frame_id, jpeg_bytes):

        packet_count = (
            len(jpeg_bytes) + PAYLOAD_SIZE - 1
        ) // PAYLOAD_SIZE

        packets = []

        for packet_index in range(packet_count):

            start = packet_index * PAYLOAD_SIZE
            end = start + PAYLOAD_SIZE

            payload = jpeg_bytes[start:end]

            header = struct.pack(
                HEADER_FORMAT,
                camera_id,
                frame_id,
                packet_index,
                packet_count
            )

            packets.append(header + payload)

        return packets