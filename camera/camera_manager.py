from camera.camera_stream import CameraStream


class CameraManager:

    def __init__(self):

        self.cameras = {
            0: CameraStream(7000),
            1: CameraStream(7001)
        }

    def start(self):

        for camera in self.cameras.values():
            camera.start()

    def get_frame(self, camera_id):

        return self.cameras[camera_id].get_frame()

    def get_fps(self, camera_id):

        return self.cameras[camera_id].get_fps()

    def get_dropped_frames(self, camera_id):

        return self.cameras[camera_id].get_dropped_frames()

    def get_jpeg_size(self, camera_id):

        return self.cameras[camera_id].get_jpeg_size()