"""Optional task-trained MobileNetV3-Small adapter; no built-in action weights."""

from pathlib import Path

ALLOWED_ACTIONS = {"normal", "phone", "drink", "eat", "smoke"}


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
