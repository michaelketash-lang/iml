from pathlib import Path

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from base_model import ImageNetSubset
from model import ModelArchitecture


# Anchors the path exactly 3 folders up from train.py (my_team -> submissions -> project -> dataset)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "dataset"
OUTPUT = Path("weights.joblib")

IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 15
SEED = 42
LR = 0.001
TRAINING_FACTOR = 0.8

IMAGENET_MEAN = (0.485,0.456,0.406)
IMAGENET_STD = (0.229,0.224,0.225)

def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor(),transforms.Normalize(mean = IMAGENET_MEAN,std= IMAGENET_STD),

    ])
    full_dataset = ImageNetSubset(root = DATA_ROOT,split="train",transform= transform)

    total_samples = len(full_dataset)
    train_size = int(total_samples * TRAINING_FACTOR)
    val_size = total_samples - train_size

    generator = torch.Generator().manual_seed(SEED)
    train_subset,val_subset = random_split(full_dataset,[train_size,val_size],generator=generator)
    train_loader = DataLoader(train_subset,batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_subset,batch_size=BATCH_SIZE,shuffle=False)
    return train_loader,val_loader


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
    """Runs the model on the validation set and returns the accuracy percentage."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def save_weights(model, output_path: Path):
    """Safely moves the model to CPU and saves the state_dict."""
    print("Saving model weights...")
    model = model.cpu()
    joblib.dump(model.state_dict(), output_path)
    print(f"Saved trained weights to {output_path}")


def main():
    """
    Full training pipeline.

    This script must create weights.joblib.
    """
    device = get_device()
    print(f"Training in device: {device}")

    print("LOADS THE DATA...")
    train_loader,val_loader=get_data_loaders()

    print("initialize the best model in the world")
    model = ModelArchitecture().to(device)
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=LR)

    print("Strat train loop.......")
    for epoch in range(1,EPOCHS+ 1):
        train_one_epoch(model,train_loader,criteria,optimizer,device,epoch)
        val_accur = evaluate_model(model,val_loader,device)
        print(f"-> Epoch {epoch} Validation Accuracy: {val_accur:.2f}%\n")
    save_weights(model,OUTPUT)
    # TODO: load dataset (you might want to use ImageNetSubset)
    # TODO: create your model

    # TODO: save trained model weights to weights.joblib



if __name__ == "__main__":
    main()