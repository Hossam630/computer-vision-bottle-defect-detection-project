# src/training/trainer.py

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, dataloaders, dataset_sizes, config):
    """Train the model using the provided dataloaders and config.

    Args:
        model: a PyTorch model instance.
        dataloaders: dict with 'train' and 'val' DataLoader objects.
        dataset_sizes: dict with dataset sizes for train and val.
        config: config dict containing training hyperparameters.

    Returns:
        The best model state loaded into the model.
    """

    # Select the device: CUDA if available and requested, otherwise CPU.
    device = torch.device(config["device"])
    model = model.to(device)

    # Loss function for classification.
    criterion = nn.CrossEntropyLoss()

    # Optimizer only updates trainable parameters (usually the final classifier layer).
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"]
    )

    # Keep track of the best validation accuracy and corresponding weights.
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    # Ensure checkpoint directory exists, and prepare best model filepath.
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    best_model_path = os.path.join(
        config["checkpoint_dir"], f"best_{config['model_name']}.pth"
    )

    # Training loop over epochs.
    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)

        # Each epoch has a training phase and a validation phase.
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Enable training mode (dropout, batchnorm, etc.).
            else:
                model.eval()   # Enable evaluation mode.

            running_loss = 0.0
            running_corrects = 0

            # Iterate over all batches in this phase.
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # Only compute gradients during training.
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        # Backpropagate and update weights.
                        loss.backward()
                        optimizer.step()

                # Accumulate loss and correct predictions for the epoch.
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            # Compute average metrics for the epoch phase.
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            # If validation accuracy improves, save the model weights.
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, best_model_path)
                print(f"Saved new best model → {best_model_path}")

    # Load best validation model weights before returning.
    model.load_state_dict(best_weights)
    return model
