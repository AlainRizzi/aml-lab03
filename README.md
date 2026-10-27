# Advanced Machine Learning — Lab 03

This project is based on the Lab 03 GitHub project skeleton and organizes the
original code from **Lab 01 (Exercises 01 and 02)** and **Lab 02** into reusable
Python modules.

## Project contents

```text
aml-lab03/
├── lab01/                  # Original Lab 01 notebooks, unchanged
├── lab02/                  # Original Lab 02 notebook, unchanged
├── models/customnet.py     # Lab 02 three-convolution CNN
├── dataset/tiny_imagenet.py  # Tiny ImageNet dataset and DataLoaders
├── utils/visualization.py  # Lab 01 image visualization + learning curves
├── train.py                # Training loop (SGD, checkpointing, optional W&B)
├── eval.py                 # Validation loop
├── checkpoints/            # Saved weights (ignored by Git)
├── data/                   # Datasets (ignored by Git)
└── requirements.txt
```

Lab 01 Exercise 01 introduces tensors, CUDA, seeds, and other PyTorch
fundamentals, so the original notebook is retained rather than converted
into a training module. Lab 01 Exercise 02 supplies dataset preparation
and visualization; Lab 02 supplies model/training/evaluation code.

## Run in Google Colab

After pushing this project, clone your own repository in Colab:

```python
!git clone https://github.com/YOUR_USERNAME/aml-lab03.git
%cd aml-lab03
!pip install -r requirements.txt
```

Download the dataset **inside Colab** and extract it below `data/`:

```python
!mkdir -p data
!wget -nc -P data https://cs231n.stanford.edu/tiny-imagenet-200.zip
!unzip -n data/tiny-imagenet-200.zip -d data/
```

The loader reads both the original Tiny ImageNet validation layout
(`val/images` + `val_annotations.txt`) and the Lab 02 rearranged
`val/<class>/` layout **without changing the dataset**. The unlabeled
official Tiny ImageNet test set is not used to calculate accuracy.

Run with your Lab 02 defaults (224x224, batch size 32, 10 epochs):

```python
!python train.py --data-root data/tiny-imagenet-200
```

Or train faster with the original Tiny ImageNet 64x64 dimensions:

```python
!python train.py --data-root data/tiny-imagenet-200 --image-size 64
```

Training displays progress every 100 batches, prints train/validation
statistics after each epoch, saves the best model weights in
`checkpoints/best_model.pt`, and plots `training_curves.png`.

## Optional experiment tracking (Weights & Biases)

In Colab, authenticate using your own W&B account, then use:

```python
!wandb login
!python train.py --image-size 64 --wandb
```

Do not commit datasets, model checkpoints, credentials, or `.env` files
to GitHub. Dataset files stay on the machine where you run training.

## Push changes to your fork

From this repository directory in a terminal:

```bash
git status
git add .
git commit -m "Populate Lab 3 with Lab 1 and Lab 2 code"
git push origin master
```

The `master` branch is used by this repository's existing Git configuration.
