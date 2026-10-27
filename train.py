"""Tiny ImageNet training, adapted from Lab 02 and organized for Lab 03."""
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import torch
from torch import nn
from dataset.tiny_imagenet import make_loaders
from eval import validate
from models.customnet import CustomNet
from utils.visualization import plot_history


def train_one_epoch(model, loader, criterion, optimizer, device,
                    epoch, print_every=100):
    model.train()
    loss_total, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        count = targets.size(0)
        loss_total += loss.item() * count
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += count
        if print_every and (batch_idx % print_every == 0 or batch_idx == len(loader)):
            print(f'Epoch {epoch}: batch {batch_idx}/{len(loader)}', flush=True)
    if total == 0:
        raise RuntimeError('Training DataLoader has zero samples')
    return loss_total / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train Lab 02 CNN on Tiny ImageNet')
    parser.add_argument('--data-root', default='data/tiny-imagenet-200')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Lab 02 uses 224; use 64 for faster training')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--print-every', type=int, default=100)
    parser.add_argument('--wandb', action='store_true',
                        help='Log loss/accuracy to Weights & Biases; requires wandb login')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    train_loader, val_loader = make_loaders(args.data_root, args.image_size,
                                            args.batch_size, args.num_workers)
    print(f'Train images: {len(train_loader.dataset)}, '
          f'validation images: {len(val_loader.dataset)}')
    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum)
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "aml-lab03"),
            config=vars(args)
        )
        
    history = {key: [] for key in ('train_loss', 'train_accuracy',
                                   'val_loss', 'val_accuracy')}
    Path('checkpoints').mkdir(exist_ok=True)
    best_accuracy = -1.0
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch, args.print_every)
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            metrics = dict(train_loss=train_loss, train_accuracy=train_accuracy,
                           val_loss=val_loss, val_accuracy=val_accuracy)
            for key, value in metrics.items():
                history[key].append(value)
            print(f'Epoch {epoch}/{args.epochs}: '
                  f'train loss {train_loss:.4f}, acc {train_accuracy:.2f}% | '
                  f'val loss {val_loss:.4f}, acc {val_accuracy:.2f}%')
            if run is not None:
                run.log(metrics, step=epoch)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), 'checkpoints/best_model.pt')
        fig = plot_history(history, 'training_curves.png')
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f'Best validation accuracy: {best_accuracy:.2f}%')
    finally:
        if run is not None:
            run.finish()


if __name__ == '__main__':
    main()
