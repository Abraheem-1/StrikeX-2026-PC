from ultralytics import YOLO

from detection.detection import Detection


class YoloDetector:

    def __init__(
        self,
        model_path,
        device="cpu"
    ):

        self.model = YOLO(model_path)

        self.device = device

    def detect(self, image):

        results = self.model(
            image,
            device=self.device,
            verbose=False
        )

        detections = []

        r = results[0]

        if r.boxes is None:
            return detections

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy().astype(int)

        names = r.names

        for (
            box,
            confidence,
            class_id
        ) in zip(
            xyxy,
            confs,
            clses
        ):

            x1, y1, x2, y2 = box

            detections.append(
                Detection(
                    class_name=names[class_id],
                    confidence=float(confidence),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2)
                )
            )

        return detections