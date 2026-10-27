"""Optional task-trained MobileNetV3-Small adapter; no built-in action weights."""

from collections import Counter, deque
from pathlib import Path

ALLOWED_ACTIONS = {"normal", "phone", "drink", "eat", "smoke"}


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
