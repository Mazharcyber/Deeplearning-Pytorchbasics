# Deeplearning-PytorchBasics

This repository demonstrates basic deep learning techniques using **PyTorch**, focusing on **logistic regression** applied to the **MNIST dataset** for digit classification.

## Contents

- **`logistic_regression.ipynb`**: Jupyter notebook implementing logistic regression on MNIST.
- **`logistic_regression.py`**: Python script with the same functionality as the notebook.
- **`README.md`**: Documentation for this project.

## Setup Instructions

### Step 1: Install Dependencies
pip install torch torchvision matplotlib


please make sure the below
To download the MNIST dataset, run the following code in a Jupyter notebook:
from torchvision import datasets
dataset = datasets.MNIST(root='data/', download=True)
