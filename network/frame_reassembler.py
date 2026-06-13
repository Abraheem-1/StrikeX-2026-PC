import time


class FrameReassembler:

    FRAME_TIMEOUT = 2.0  # seconds before incomplete frame is discarded

    def __init__(self):
        self.frames = {}

    def add_packet(
        self,
        frame_id,
        packet_index,
        packet_count,
        payload
    ):
        now = time.time()

        # ── Cleanup stale incomplete frames ──────────────────────
        stale = [
            fid for fid, f in self.frames.items()
            if now - f["timestamp"] > self.FRAME_TIMEOUT
        ]
        for fid in stale:
            del self.frames[fid]

        # ── Add packet to frame buffer ───────────────────────────
        if frame_id not in self.frames:
            self.frames[frame_id] = {
                "packet_count": packet_count,
                "packets": {},
                "timestamp": now
            }

        self.frames[frame_id]["packets"][packet_index] = payload

        # ── Check if frame is complete ───────────────────────────
        frame = self.frames[frame_id]

        if len(frame["packets"]) == frame["packet_count"]:

            jpeg_bytes = b"".join(
                frame["packets"][i]
                for i in range(packet_count)
            )

            del self.frames[frame_id]

            return frame_id, jpeg_bytes

        return None