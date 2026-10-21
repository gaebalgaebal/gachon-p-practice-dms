"""Train an optional MobileNetV3-Small using a driver-disjoint image manifest."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

from validate_manifest import validate
from dms.action import ALLOWED_ACTIONS
from dms.metrics import classification_report


def train(args):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    validate(args.manifest)
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(not row.get("file_path", "").strip() for row in rows):
        raise ValueError("Action manifest additionally requires file_path for every row")
    classes = sorted({row['label'] for row in rows if row['split'] == 'train'})
    if len(classes) < 2 or not set(classes).issubset(ALLOWED_ACTIONS):
        raise ValueError("Need at least two train classes from normal/phone/drink/eat/smoke")
    if any(row['label'] not in classes for row in rows):
        raise ValueError("Validation/test class is missing from train")
    for split in ('train', 'val', 'test'):
        if {row['label'] for row in rows if row['split'] == split} != set(classes):
            raise ValueError(f"Every class must appear in {split}; revise the subject split")
    if args.output.exists():
        raise ValueError("Checkpoint output already exists; choose a new output")
    for row in rows:
        path = Path(row['file_path'])
        row['resolved_path'] = path if path.is_absolute() else args.manifest.parent / path
        if not row['resolved_path'].is_file():
            raise FileNotFoundError(row['resolved_path'])
    # Prevent renamed copies of one image leaking across splits.
    hashes = {}
    for row in rows:
        digest = hashlib.sha256(row['resolved_path'].read_bytes()).hexdigest()
        if digest in hashes and hashes[digest] != row['split']:
            raise ValueError("Identical image content appears across splits")
        hashes[digest] = row['split']
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    eval_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                         transforms.ToTensor(), normalize])
    train_transform = transforms.Compose([
        transforms.Resize(256), transforms.RandomResizedCrop(224, scale=(.8, 1.)),
        transforms.ColorJitter(brightness=.2, contrast=.2), transforms.ToTensor(), normalize,
    ])
    class Images(Dataset):
        def __init__(self, split):
            self.rows = [row for row in rows if row['split'] == split]
            self.transform = train_transform if split == 'train' else eval_transform
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            row = self.rows[index]
            with Image.open(row['resolved_path']) as image:
                tensor = self.transform(image.convert('RGB'))
            return tensor, classes.index(row['label'])
    datasets = {split: Images(split) for split in ('train', 'val', 'test')}
    if len(datasets['train']) < 2:
        raise ValueError("Training requires at least two images")
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size,
                                 shuffle=split == 'train', num_workers=0)
               for split, dataset in datasets.items()}
    model = mobilenet_v3_small(weights=None if args.from_scratch else
                               MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Freeze feature statistics too; small student datasets should not corrupt BatchNorm.
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    steps, best_f1, history = 0, -1., []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def evaluate(split):
        model.eval()
        actual, predicted = [], []
        with torch.inference_mode():
            for images, labels in loaders[split]:
                guess = model(images).argmax(dim=1)
                actual.extend(classes[i] for i in labels.tolist())
                predicted.extend(classes[i] for i in guess.tolist())
        return classification_report(actual, predicted, classes)
    for epoch in range(args.epochs):
        model.train()
        model.features.eval()
        total_loss = 0.
        for images, labels in loaders['train']:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            steps += 1
            total_loss += loss.item() * len(labels)
        report = evaluate('val')
        history.append({'epoch': epoch + 1, 'train_loss': total_loss / len(datasets['train']),
                        'val': report})
        print(f"epoch={epoch + 1} val_macro_f1={report['macro_f1']:.4f}")
        if report['macro_f1'] > best_f1:
            best_f1 = report['macro_f1']
            torch.save({'architecture': 'mobilenet_v3_small', 'classes': classes,
                        'state_dict': model.state_dict(), 'trained_steps': steps,
                        'preprocessing': 'resize256_center224_imagenet',
                        'smoke_only': args.smoke_only, 'seed': args.seed,
                        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest()},
                       args.output)
    payload = torch.load(args.output, map_location='cpu', weights_only=True)
    model.load_state_dict(payload['state_dict'])
    test = evaluate('test')
    results = {'history': history, 'test': test, 'seed': args.seed,
               'torch': str(torch.__version__), 'epochs': args.epochs, 'batch_size': args.batch_size,
               'learning_rate': args.learning_rate, 'from_scratch': args.from_scratch,
               'smoke_only': args.smoke_only, 'classes': classes,
               'samples': {split: len(data) for split, data in datasets.items()},
               'checkpoint_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
               'manifest_sha256': payload['manifest_sha256']}
    args.output.with_suffix('.metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f"Saved {args.output}. Test macro F1={test['macro_f1']:.4f}; inspect data/conditions before use.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--smoke-only', action='store_true', help='Mark synthetic test weights unusable in live demo')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) <= 0 or not 0 < args.learning_rate < 1:
        parser.error('epochs/batch-size/threads must be positive and learning-rate in (0,1)')
    try:
        train(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f'Action training error: {exc}\n')


if __name__ == '__main__':
    main()
