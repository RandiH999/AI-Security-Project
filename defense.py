import torch

def train_adversarial(model, train_loader, optimizer, criterion, epsilon, device):
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        data_grad = images.grad.data
        adv_images = torch.clamp(images + epsilon * data_grad.sign(), 0, 1)

        outputs_adv = model(adv_images)
        loss_adv = criterion(outputs_adv, labels)

        optimizer.zero_grad()
        loss_adv.backward()
        optimizer.step()
