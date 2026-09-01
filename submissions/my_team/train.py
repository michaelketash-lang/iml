from pathlib import Path
import random
from torch.utils.data import DataLoader, Subset, ConcatDataset
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from base_model import ImageNetSubset
from model import ModelArchitecture

##################################################################
# Constants:
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "dataset"
OUTPUT = Path("weights.joblib")

IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 30

LR = 0.001

IMAGENET_MEAN = (0.485,0.456,0.406)
IMAGENET_STD = (0.229,0.224,0.225)

# Fraction of each augmentation folder mixed into training; the rest stays
# held out for the aug_loader monitoring metric. black_white/salt_pepper get
# a bigger share since get_train_transforms() has no analog for them, while
# color_jitter already overlaps with the ColorJitter transform.
AUG_TRAIN_FRACTIONS = {
    "augmentations/black_white": 0.5,
    "augmentations/salt_pepper": 0.5,
    "augmentations/color_jitter": 0.2,
}
AUG_SPLIT_SEED = 42
#################################################################


def get_train_transforms():
    """creating dynamic data augmentation pipeline that applies random visual changes
    for example cropping ,flipping and color shifts for each image during training ."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """create a pipeline that simply resize and normalize the images without
     applying random augmentatinos."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def stratified_indices(dataset, take_frac, seed):
    """Per-class shuffle+split (same idea as split_data.py), returns (take_idx, rest_idx)."""
    rng = random.Random(seed)
    by_class = {}
    for i, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(i)

    take_idx, rest_idx = [], []
    for label, idxs in by_class.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_take = int(len(idxs) * take_frac)
        take_idx.extend(idxs[:n_take])
        rest_idx.extend(idxs[n_take:])

    return take_idx, rest_idx


def build_base_datasets():
    """Loads the clean train and validation ImageNetSubset datasets with their transforms."""
    train_dataset = ImageNetSubset(root=DATA_ROOT, split="train",
                                   transform=get_train_transforms())
    val_dataset = ImageNetSubset(root=DATA_ROOT, split="validation",
                                 transform=get_val_transforms())
    return train_dataset, val_dataset


def build_augmentation_splits():
    """Splits each augmentation folder per class: part goes into training (with
    train-time transforms), the rest stays held out for the aug_loader monitoring
    metric (with clean eval transforms), so that metric isn't just measuring memorization."""
    train_aug_parts = []
    held_aug_parts = []

    for aug_split, train_frac in AUG_TRAIN_FRACTIONS.items():
        aug_train_view = ImageNetSubset(root=DATA_ROOT, split=aug_split,
                                        transform=get_train_transforms())
        aug_val_view = ImageNetSubset(root=DATA_ROOT, split=aug_split,
                                      transform=get_val_transforms())

        take_idx, rest_idx = stratified_indices(aug_train_view, train_frac, AUG_SPLIT_SEED)

        train_aug_parts.append(Subset(aug_train_view, take_idx))
        held_aug_parts.append(Subset(aug_val_view, rest_idx))

    return train_aug_parts, held_aug_parts


def get_data_loaders():
    """Builds the train, validation, and combined augmentation pipelines."""
    train_dataset, val_dataset = build_base_datasets()
    train_aug_parts, held_aug_parts = build_augmentation_splits()

    train_dataset = ConcatDataset([train_dataset] + train_aug_parts)
    combined_aug_dataset = ConcatDataset(held_aug_parts)

    # wrapping datasets in dataloaders to handle batching ,shuffling only train data.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    aug_loader = DataLoader(combined_aug_dataset, batch_size=BATCH_SIZE,
                            shuffle=False)

    return train_loader, val_loader, aug_loader


def get_device():
    """Returning the Optimal Hardware device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Executes a single training epoch and updates model weights."""
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")


def evaluate_model(model, val_loader, device) -> float:
    """Runs the model on  validation set and returns the acc percentage.
       Comparing the model's predictions to true labels"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def save_weights(model, output_path: Path):
    """Safely moves the model to CPU and saves the state_dict."""
    print("Saving model weights:")
    model = model.cpu()
    joblib.dump(model.state_dict(), output_path)
    print(f"Saved trained weights to {output_path}")


def build_model_and_optimizer(device):
    """Initializes the model, loss function, optimizer, and LR scheduler for training."""
    model = ModelArchitecture().to(device)
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                                gamma=0.5)
    return model, criteria, optimizer, scheduler


def run_training_loop(model, train_loader, val_loader, aug_loader, criteria,
                      optimizer, scheduler, device):
    """Runs all training epochs, printing clean and augmented accuracy after each one."""
    for epoch in range(1, EPOCHS + 1):
        # train
        train_one_epoch(model, train_loader, criteria, optimizer, device,
                        epoch)

        # check on existing data
        val_accur = evaluate_model(model, val_loader, device)

        # check on augmentation
        aug_accur = evaluate_model(model, aug_loader, device)

        print(
            f"-> Epoch {epoch} | Val Acc: {val_accur:.2f}% | Aug Acc: {aug_accur:.2f}%\n")
        scheduler.step()


def main():
    device = get_device()
    print(f"Training on device: {device}")

    print("LOADING DAta:")
    train_loader, val_loader, aug_loader = get_data_loaders()

    print("Initialize model:")
    model, criteria, optimizer, scheduler = build_model_and_optimizer(device)

    print("Start training loop:")
    run_training_loop(model, train_loader, val_loader, aug_loader, criteria,
                      optimizer, scheduler, device)

    save_weights(model, OUTPUT)


if __name__ == "__main__":
    main()
