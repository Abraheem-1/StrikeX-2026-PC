from network.packetizer import Packetizer
from network.frame_reassembler import FrameReassembler

data = bytes(5000)

packets = Packetizer.packetize(
    camera_id=0,
    frame_id=1,
    jpeg_bytes=data
)

reassembler = FrameReassembler()

result = None

for packet in packets:

    payload = packet[9:]

    result = reassembler.add_packet(
        frame_id=1,
        packet_index=packets.index(packet),
        packet_count=len(packets),
        payload=payload
    )

print(len(result))