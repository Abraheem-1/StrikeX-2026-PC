class FrameReassembler:

    def __init__(self):
        self.frames = {}

    def add_packet(
        self,
        frame_id,
        packet_index,
        packet_count,
        payload
    ):

        if frame_id not in self.frames:

            self.frames[frame_id] = {
                "packet_count": packet_count,
                "packets": {}
            }

        self.frames[frame_id]["packets"][packet_index] = payload

        frame = self.frames[frame_id]

        if len(frame["packets"]) == frame["packet_count"]:

            jpeg_bytes = b"".join(
                frame["packets"][i]
                for i in range(packet_count)
            )

            del self.frames[frame_id]

            return jpeg_bytes

        return None