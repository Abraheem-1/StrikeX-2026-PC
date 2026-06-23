from ultralytics import YOLO

from detection.detection import Detection


class YoloDetector:

    def __init__(
        self,
        model_path,
        device="cpu",
        imgsz=640,
        nms_iou=0.45
    ):
        self.model = YOLO(model_path)

        # IMPORTANT: pass device="cuda" (or 0) from main.py if you have an
        # NVIDIA GPU. Inference at 174 ms is the signature of CPU inference and
        # is the root of your whole latency chain — moving to GPU typically
        # drops it to ~10-30 ms, which shrinks the 260 ms pipeline latency,
        # eases the pitch/yaw lead, AND makes tracking far more stable
        # (boxes overlap more frame-to-frame, fewer ID splits).
        self.device = device

        # imgsz: match what the model was trained at. Lowering it (e.g. 416)
        # speeds inference at some accuracy cost — useful on CPU.
        self.imgsz = imgsz

        # NMS IoU threshold. LOWER = more aggressive suppression of overlapping
        # boxes. Default ultralytics value (0.7) let multiple boxes survive on
        # one drone (your "Targets: 3"). 0.45 merges them to one box per object.
        self.nms_iou = nms_iou

    def detect(self, image):

        results = self.model(
            image,
            conf=0.70,
            iou=self.nms_iou,        # aggressive NMS -> one box per drone
            imgsz=self.imgsz,
            device=self.device,
            # half=True,             # uncomment on GPU for a free speed-up
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

        for box, confidence, class_id in zip(xyxy, confs, clses):

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