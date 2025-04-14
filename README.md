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

__for the pytroch basics please follow the below__

📚 Recommended Learning Order for Notebooks
To help you understand the basics of Deep Learning with PyTorch, please follow the notebooks in the sequence below. Each notebook builds upon the concepts introduced in the previous one.

🔰 Step-by-Step Learning Path:

1. **`pytorch_tesnor_mazharwrite.ipynb`**  
   → Introduction to PyTorch tensors, basic operations, and how to manipulate them.

2. **`Autograd.ipynb`**  
   → Understanding automatic differentiation, gradients, and how PyTorch handles backpropagation.

3. **`pytorch_training_pipeline.ipynb`**  
   → Building a manual training pipeline using only PyTorch tensors and functions.

4. **`pytorch_nn_module.ipynb`**  
   → Implementing models using `nn.Module`, making code modular and cleaner.

5. **`pytorch_training_pipeline_using_nn_module.ipynb`**  
   → Building a complete training loop using `nn.Module`, loss functions, and optimizers.

6. **`dataset_and_dataloader_demo.ipynb`**  
   → Using `Dataset` and `DataLoader` for handling data in batches efficiently.

7. **`mazhar_esitpytorch_training_pipeline_using_dataset_and_dataloader.ipynb`**  
   → A full training pipeline using `Dataset`, `DataLoader`, `nn.Module`, and optimizers.

8. **`training_neural_network_gpu.ipynb`**  
   → Training a neural network on MNIST using GPU acceleration, with a full training-validation loop.

9. **`ann_fashion_mnist_pytorch_gpu.ipynb`**  
   → A real-world implementation of an Artificial Neural Network (ANN) trained on Fashion MNIST, using GPU acceleration.

10. **`Building_Simple_FFNN.ipynb`**  
    → Learn to build and understand a simple Feed-Forward Neural Network from scratch.

