from pathlib import Path
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

SEED = 42
LR = 0.001
TRAINING_FACTOR = 0.8

IMAGENET_MEAN = (0.485,0.456,0.406)
IMAGENET_STD = (0.229,0.224,0.225)
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
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_data_loaders():
    """Builds the train, validation, and combined augmentation pipelines."""
    # loading training and validation sets with the specific transforms
    train_dataset = ImageNetSubset(root=DATA_ROOT, split="train",
                                   transform=get_train_transforms())
    val_dataset = ImageNetSubset(root=DATA_ROOT, split="validation",
                                 transform=get_val_transforms())
    # load stress test images with clean validation transforms
    aug_bw = ImageNetSubset(root=DATA_ROOT, split="augmentations/black_white",
                            transform=get_val_transforms())
    aug_cj = ImageNetSubset(root=DATA_ROOT, split="augmentations/color_jitter",
                            transform=get_val_transforms())
    aug_sp = ImageNetSubset(root=DATA_ROOT, split="augmentations/salt_pepper",
                            transform=get_val_transforms())
    # merge all of three into one big dataset for evaluation
    combined_aug_dataset = ConcatDataset([aug_bw, aug_cj, aug_sp])
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


def main():
    device = get_device()
    print(f"Training on device: {device}")

    print("LOADING DAta:")
    train_loader, val_loader, aug_loader = get_data_loaders()

    print("Initialize model:")
    model = ModelArchitecture().to(device)
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                                gamma=0.5)

    print("Start training loop:")
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

    save_weights(model, OUTPUT)


if __name__ == "__main__":
    main()
