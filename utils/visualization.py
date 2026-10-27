"""Visualization functions adapted from Lab 01 Exercise 02."""
import matplotlib.pyplot as plt
import numpy as np
from dataset.tiny_imagenet import MEAN, STD


def denormalize(image):
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(image * np.array(STD) + np.array(MEAN), 0, 1)


def show_class_examples(loader, n=10):
    """Show up to n examples, each from a distinct class."""
    n = min(n, len(loader.dataset.classes))
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    axes = axes.ravel()
    seen = set()
    for images, labels in loader:
        for image, label in zip(images, labels):
            idx = int(label)
            if idx not in seen:
                ax = axes[len(seen)]
                ax.imshow(denormalize(image))
                ax.set_title(f'Class: {loader.dataset.classes[idx]}')
                ax.axis('off')
                seen.add(idx)
            if len(seen) == n:
                break
        if len(seen) == n:
            break
    for ax in axes[len(seen):]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def plot_history(history, output_path='training_curves.png'):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    for ax, name, label in zip(axes, ('loss', 'accuracy'), ('Loss', 'Accuracy (%)')):
        ax.plot(epochs, history['train_' + name], label='Training')
        ax.plot(epochs, history['val_' + name], label='Validation')
        ax.set(xlabel='Epoch', ylabel=label, title=label)
        ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160)
    return fig
