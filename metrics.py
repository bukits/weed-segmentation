
import torch
from typing import List
from abc import ABC, abstractmethod

class Metric(ABC):
    """Abstract metric class to evaluate generative models.
    """
    def __init__(self, name: str, use_cuda: bool = True) -> None:
        self.name = name
        self.use_cuda = use_cuda
    
    def compute_metric(self, true_masks_batched, predicted_masks_batched) -> torch.Tensor:
        cumulative_score = 0.
        n_batch = 0.
        with torch.no_grad():
            for i, true_mask_batch in enumerate(true_masks_batched):
                predicted_mask_batch = predicted_masks_batched[i]

                out = self.batch_compute([predicted_mask_batch, true_mask_batch])
                cumulative_score += out.sum(dim=0)
                n_batch += 1.

            res = out / n_batch

        return res

    @abstractmethod
    def batch_compute(self, inp: List[torch.Tensor], model: torch.nn.Module):
        raise NotImplementedError

class CustomIoUMetric(Metric):
    def __init__(self, use_cuda: bool = True):
        super().__init__("Custom IoU Metric", use_cuda=use_cuda)
        self.name = "IOU Metric"
    
    def calculate_iou(self, pred_mask, target_mask, threshold=0.5):
        pred_mask = (pred_mask > threshold).float()
        target_mask = (target_mask > threshold).float()

        intersection = torch.sum(pred_mask * target_mask)
        union = torch.sum(pred_mask) + torch.sum(target_mask) - intersection

        epsilon = 1e-6
        iou = (intersection + epsilon) / (union + epsilon)

        return iou.item()
    
    def batch_compute(self, inp: List[torch.Tensor]):
        pred_mask, target_mask = inp

        pred_mask = torch.argmax(pred_mask, dim=1)
        iou_score = self.calculate_iou(pred_mask, target_mask, threshold=0.5)

        return torch.tensor(iou_score)
    
class MeanPixelAccuracy:
    def mean_iou(pred, target):
        pred = torch.argmax(pred, dim=1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.item()

    def compute_metric(self, dataloader, model):
        total_accuracy = 0
        total_samples = 0
        for inputs, targets in dataloader:
            predictions = model(inputs)
            accuracy = self.mean_pixel_accuracy(predictions, targets)
            total_accuracy += accuracy * inputs.size(0)
            total_samples += inputs.size(0)
        return total_accuracy / total_samples

class MeanIoU:
    def mean_iou(pred, target):
        pred = torch.argmax(pred, dim=1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.item()

    def compute_metric(self, dataloader, model):
        total_iou = 0
        total_samples = 0
        for inputs, targets in dataloader:
            predictions = model(inputs)
            iou = self.mean_iou(predictions, targets)
            total_iou += iou * inputs.size(0)
            total_samples += inputs.size(0)
        return total_iou / total_samples

class DiceCoefficient:
    def dice_coefficient(pred, target):
        pred = torch.argmax(pred, dim=1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
        return dice.item()

    def compute_metric(self, dataloader, model):
        total_dice = 0
        total_samples = 0
        for inputs, targets in dataloader:
            predictions = model(inputs)
            dice = dice_coefficient(predictions, targets)
            total_dice += dice * inputs.size(0)
            total_samples += inputs.size(0)
        return total_dice / total_samples
