from network.packetizer import Packetizer

test_data = bytes(5000)

packets = Packetizer.packetize(
    camera_id=0,
    frame_id=1,
    jpeg_bytes=test_data
)

print(f"Number of packets: {len(packets)}")