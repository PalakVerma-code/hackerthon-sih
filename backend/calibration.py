import torch
from torch import (
    nn,
    optim,
)


def fit_temperature(model, val_loader, device) -> float:
    model.eval()
    logits_list, labels_list = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            logits_list.append(logits.cpu())
            labels_list.append(labels)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    nll_criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=1)
