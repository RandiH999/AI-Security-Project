# AI Security Project – MNIST Adversarial Attack

## Overview
This project demonstrates adversarial attacks on a neural network trained on MNIST and evaluates defense strategies.

## Model
Simple neural network trained using PyTorch on MNIST dataset.

## Attack
Implemented Fast Gradient Sign Method (FGSM) to generate adversarial examples.

## Results
- Clean accuracy: ~97%
- Attack accuracy: drops to ~0–50%
- Defense improves robustness significantly

## Defense Method
Adversarial training using FGSM-generated examples.

## Tools
- PyTorch
- NumPy
- Matplotlib
