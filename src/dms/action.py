"""Task-trained MobileNet and YOLO11 action adapters."""

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

ALLOWED_ACTIONS = {"normal", "phone", "drink", "eat", "smoke"}


@dataclass(frozen=True)
class ActionDetection:
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


class ActionSmoother:
    """Majority vote over fresh model outputs; reset immediately on unusable input."""

    def __init__(self, window: int = 5, consensus: float = .6, min_samples: int = 3):
        if window <= 0 or not 0 < consensus <= 1 or not 0 < min_samples <= window:
            raise ValueError("Require positive window, consensus in (0,1], and min_samples <= window")
        self.window = window
        self.consensus = consensus
        self.min_samples = min_samples
        self._labels = deque(maxlen=window)
        self._scores = deque(maxlen=window)

    def reset(self):
        self._labels.clear()
        self._scores.clear()

    def update(self, label: str, scores: dict[str, float]):
        if label != "unknown" and label not in ALLOWED_ACTIONS:
            raise ValueError("Unsupported action label")
        self._labels.append(label)
        self._scores.append(dict(scores))
        return self.current()

    def current(self):
        if not self._labels:
            return "unknown", {}
        counts = Counter(self._labels)
        label, votes = counts.most_common(1)[0]
        if (len(self._labels) < self.min_samples or label == "unknown"
                or votes / len(self._labels) < self.consensus):
            label = "unknown"
        keys = sorted({key for row in self._scores for key in row})
        averaged = {key: sum(row.get(key, 0.) for row in self._scores) / len(self._scores)
                    for key in keys}
        return label, averaged


class MobileNetAction:
    requires_face = True

    def __init__(self, checkpoint: Path, threshold: float = .7):
        import torch
        from torchvision.models import mobilenet_v3_small
        from torchvision import transforms

        self.torch = torch
        torch.set_num_threads(4)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.classes = payload["classes"]
        if (payload.get("architecture") != "mobilenet_v3_small"
                or payload.get("trained_steps", 0) < 1 or payload.get("smoke_only", False)):
            raise ValueError("Require a task-trained MobileNetV3-Small checkpoint, not smoke/random weights")
        if (len(self.classes) < 2 or len(set(self.classes)) != len(self.classes)
                or not set(self.classes).issubset(ALLOWED_ACTIONS)):
            raise ValueError("Unsupported action class mapping")
        if payload.get("preprocessing") != "resize256_center224_imagenet":
            raise ValueError("Unsupported action preprocessing")
        self.threshold = threshold
        self.model = mobilenet_v3_small(weights=None)
        self.model.classifier[-1] = torch.nn.Linear(self.model.classifier[-1].in_features,
                                                  len(self.classes))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def predict(self, frame):
        import cv2
        from PIL import Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with self.torch.inference_mode():
            scores = self.model(self.transform(image).unsqueeze(0)).softmax(dim=1)[0].tolist()
        mapping = dict(zip(self.classes, scores))
        index = max(range(len(scores)), key=scores.__getitem__)
        return (self.classes[index] if scores[index] >= self.threshold else "unknown"), mapping


class YOLOAction:
    """Map AI Hub phone-use/call detections to the common DMS phone action."""

    requires_face = False
    SUPPORTED_CLASSES = {"phone_use", "phone_call"}

    def __init__(self, checkpoint: Path, threshold: float = .5, device: str = "auto"):
        from ultralytics import YOLO

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not 0 < threshold <= 1:
            raise ValueError("YOLO action threshold must be in (0,1]")
        self.threshold = threshold
        self.device = device
        self.model = YOLO(str(checkpoint))
        names = set(self.model.names.values())
        if names != self.SUPPORTED_CLASSES:
            raise ValueError(f"Expected YOLO classes {sorted(self.SUPPORTED_CLASSES)}, got {sorted(names)}")
        self.last_detections: list[ActionDetection] = []

    @staticmethod
    def aggregate(detections: list[ActionDetection], threshold: float):
        scores = {name: 0. for name in YOLOAction.SUPPORTED_CLASSES}
        for detection in detections:
            if detection.label in scores:
                scores[detection.label] = max(scores[detection.label], detection.confidence)
        phone_score = max(scores.values())
        common_scores = {**scores, "phone": phone_score, "normal": 1. - phone_score}
        return ("phone" if phone_score >= threshold else "normal"), common_scores

    def predict(self, frame):
        result = self.model.predict(
            source=frame, conf=self.threshold, verbose=False,
            device=None if self.device == "auto" else self.device,
        )[0]
        detections = []
        if result.boxes is not None:
            for class_id, confidence, coords in zip(
                    result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
                label = str(result.names[int(class_id)])
                detections.append(ActionDetection(label, float(confidence), tuple(map(float, coords))))
        self.last_detections = detections
        return self.aggregate(detections, self.threshold)


def load_action_model(checkpoint: Path, backend: str, threshold: float | None, device: str = "auto"):
    if backend == "auto":
        backend = "yolo11" if "yolo" in checkpoint.name.lower() else "mobilenet"
    if backend == "yolo11":
        return YOLOAction(checkpoint, .25 if threshold is None else threshold, device)
    if backend == "mobilenet":
        return MobileNetAction(checkpoint, .7 if threshold is None else threshold)
    raise ValueError(f"Unsupported action backend: {backend}")
