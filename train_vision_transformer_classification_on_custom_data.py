
import pathlib
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


use_own_data = True #@param {type:"boolean"}

if use_own_data:
  #Get the current working directory
  current_dir = pathlib.Path(os.getcwd())
    
  #Combine with the 'decades' folder
  data_dir = current_dir / 'decades'

  if os.path.isdir(data_dir):
    print("Data folder loaded!")
  else:
    print("Error: Data folder not found.")


import torchvision
from torchvision import datasets, transforms, models
import torch
import numpy as np


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Resize(224),
                                       transforms.ToTensor(),
# These normalizations are a good idea, but are omitted here just because they
# make it a bit more complicated to demonstrate the model's output.
                                      #  transforms.Normalize([0.485, 0.456, 0.406],
                                      #                       [0.229, 0.224, 0.225])
                                       ])

test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      # transforms.Normalize([0.485, 0.456, 0.406],
                                      #                   [0.229, 0.224, 0.225])
                                      ])
def load_split_train_valid_test(datadir,
                          valid_size = .15,
                          test_size = .10,
                          train_transforms = train_transforms,
                          test_transforms = test_transforms):

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    valid_data = datasets.ImageFolder(datadir, transform=test_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)

    # Get total number of samples
    num_total = len(train_data)
    indices = list(range(num_total))
    np.random.shuffle(indices)

    # Calculate split indices
    test_split = int(np.floor(test_size * num_total))
    valid_split = int(np.floor((test_size + valid_size) * num_total))

    # Split indices for each subset
    test_idx = indices[:test_split]
    valid_idx = indices[test_split:valid_split]
    train_idx = indices[valid_split:]
    from torch.utils.data.sampler import SubsetRandomSampler
    # Create samplers for each subset
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Define DataLoaders for each subset
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, sampler=valid_sampler, batch_size=1)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=32)

    return trainloader, validloader, testloader

train_loader, eval_loader, test_loader = load_split_train_valid_test(data_dir)

# Configuration
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ViT Feature Extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")


# Load Pretrained ViT Model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=10
)
model.to(DEVICE)

# Optimizer and Loss Function
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = CrossEntropyLoss()

# Training Function
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Forward pass
        outputs = model(pixel_values=inputs).logits
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

# Validation Function
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(pixel_values=inputs).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

# Training Loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, eval_loader, criterion)

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), "finetuned_vit.pth")
print("Model saved as finetuned_vit.pth")