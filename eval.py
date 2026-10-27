"""Validation logic adapted from Lab 02.

No gradient computation or optimizer updates are done during validation.
"""
import torch


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += batch_size
    if total == 0:
        raise RuntimeError('Validation DataLoader has zero samples')
    return running_loss / total, 100.0 * correct / total
