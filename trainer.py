
from typing import Callable, List
import torch
import torch.utils.data as data
from metrics import Metric

class UNETTrainer:
    def __init__(self, model: torch.nn.Module,
                 loss: Callable,
                 optimizer: torch.optim.Optimizer,
                 use_cuda=True):
        self.loss = loss
        self.use_cuda = use_cuda
        self.optimizer = optimizer
        self.predicted_masks = []
        self.true_masks = []

        if use_cuda:
            self.model = model.to(device="cuda:0")

    def seperateTarget(self, input, target):
        b, c, h, w = input.size()
        separated_masks = torch.empty(b, c, h, w)

        separated_masks = separated_masks.cuda()

        for i in range(b):
            for class_idx in range(c):
                mask = target[i] == class_idx
                separated_masks[i, class_idx, :, :] = mask.int()
        return separated_masks

    def fit(self, train_data_loader: data.DataLoader,
            epoch: int):
        avg_loss = 0.
        self.model.training = True
        for e in range(epoch):
            print(f"Start epoch {e+1}/{epoch}")
            n_batch = 0
            for i, (input_image, true_mask) in enumerate(train_data_loader):
                # Reset previous gradients
                self.optimizer.zero_grad()

                # Move data to cuda is necessary:
                if self.use_cuda:
                    input_image = input_image.cuda()
                    true_mask = self.seperateTarget(input_image.cuda(), true_mask.cuda())

                # Make forward
                pred_mask = self.model.forward(input_image)
                loss = self.loss(true_mask,
                                 pred_mask)
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()
                avg_loss += loss.item()
                n_batch += 1

                print(
                    f"\r{i+1}/{len(train_data_loader)}: loss = {loss / n_batch}", end='')
            print()

        return avg_loss

    def create_masks(self, val_data_loader: data.DataLoader):
        with torch.no_grad():
            for i, (x, attr) in enumerate(val_data_loader):
                if self.use_cuda:
                    x = x.cuda()
                    attr = attr.cuda()
                predicted_mask_batched = self.model(x)
                self.predicted_masks.append(predicted_mask_batched)
                self.true_masks.append(attr)

    def eval(self, val_data_loader: data.DataLoader,
            metrics: List[Metric]) -> torch.Tensor:
        scores = torch.zeros(len(metrics), dtype=torch.float32)
        names = []
        self.create_masks(val_data_loader)
        with torch.no_grad():
            for i, m in enumerate(metrics):
                scores[i] = m.compute_metric(self.true_masks, self.predicted_masks)
                names.append(m.name)

        return scores, names

class VAETrainer:
    def __init__(self, model: torch.nn.Module,
                 loss: Callable,
                 optimizer: torch.optim.Optimizer,
                 use_cuda=True):
        self.loss = loss
        self.use_cuda = use_cuda
        self.optimizer = optimizer
        self.predicted_masks = []
        self.true_masks = []

        if use_cuda:
            self.model = model.to(device="cuda:0")

    def fit(self, train_data_loader: data.DataLoader,
            epoch: int):
        avg_loss = 0.
        self.model.training = True
        for e in range(epoch):
            print(f"Start epoch {e+1}/{epoch}")
            n_batch = 0
            for i, (input_image, true_mask) in enumerate(train_data_loader):
                # Reset previous gradients
                self.optimizer.zero_grad()

                # Move data to cuda is necessary:
                if self.use_cuda:
                    input_image = input_image.cuda()
                    true_mask = true_mask.cuda()

                pred_mask = self.model.forward(input_image)
                loss = self.loss(true_mask, 
                                pred_mask)

                loss.backward()

                # Adjust learning weights
                self.optimizer.step()
                avg_loss += loss.item()
                n_batch += 1

                print(f"\r{i+1}/{len(train_data_loader)}: loss = {loss / n_batch}", end='')
            print()

        return avg_loss
    
    def create_masks(self, val_data_loader: data.DataLoader):
        with torch.no_grad():
            for i, (x, attr) in enumerate(val_data_loader):
                if self.use_cuda:
                    x = x.cuda()
                    attr = attr.cuda()
                predicted_mask_batched = self.model(x)
                self.predicted_masks.append(predicted_mask_batched)
                self.true_masks.append(attr)


    def eval(self, val_data_loader: data.DataLoader,
            metrics: List[Metric]) -> torch.Tensor:
        scores = torch.zeros(len(metrics), dtype=torch.float32)
        names = []
        self.create_masks(val_data_loader)
        with torch.no_grad():
            for i, m in enumerate(metrics):
                scores[i] = m.compute_metric(self.true_masks, self.predicted_masks)
                names.append(m.name)

        return scores, names
        
