import torch

def fgsm_attack(image, epsilon, data_grad):
    return torch.clamp(image + epsilon * data_grad.sign(), 0, 1)
