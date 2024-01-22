
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

class IntersectionOverUnion(Metric):
    def __init__(self, use_cuda: bool = True):
        super().__init__("Custom IoU Metric", use_cuda=use_cuda)
        self.name = "Intersection Over Union Metric"
    
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
    
class MeanIntersectionOverUnion(IntersectionOverUnion):
    def __init__(self, use_cuda: bool = True):
        super().__init__("Mean IoU", use_cuda=use_cuda)
        self.name = "Mean Intersection Over Union"

    def batch_compute(self, inp: List[torch.Tensor]):
        pred_mask, target_mask = inp

        num_classes = pred_mask.size(1)
        iou_scores = []

        for class_idx in range(num_classes):
            pred_mask_class = pred_mask[:, class_idx, :, :]
            target_mask_class = (target_mask == class_idx).float()

            iou_class = self.calculate_iou(pred_mask_class, target_mask_class, threshold=0.5)
            iou_scores.append(iou_class)

        mean_iou = torch.tensor(iou_scores).mean().item()

        return mean_iou
    
class PixelAccuracy(Metric):
    def __init__(self, use_cuda: bool = True):
        super().__init__("Pixel Accuracy", use_cuda=use_cuda)
        self.name = "Pixel Accuracy"

    def calculate_pixel_accuracy(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        correct_pixels = torch.sum(pred_flat == target_flat).item()

        total_pixels = target_flat.numel()

        accuracy = correct_pixels / total_pixels

        return accuracy

    def batch_compute(self, inp: List[torch.Tensor]):
        pred_mask, target_mask = inp
        pred_mask = torch.argmax(pred_mask, dim=1)
        pixel_accuracy = self.calculate_pixel_accuracy(pred_mask, target_mask)

        return torch.tensor(pixel_accuracy)

class MeanPixelAccuracy(PixelAccuracy):
    def __init__(self, use_cuda: bool = True):
        super().__init__("Mean Pixel Accuracy", use_cuda=use_cuda)
        self.name = "Mean Pixel Accuracy"

    def batch_compute(self, inp: List[torch.Tensor]):
        pred_mask, target_mask = inp

        num_classes = pred_mask.size(1)
        pixel_accuracies = []

        for class_idx in range(num_classes):
            pred_mask_class = pred_mask[:, class_idx, :, :]
            target_mask_class = (target_mask == class_idx).float()

            class_accuracy = self.calculate_pixel_accuracy(pred_mask_class, target_mask_class)
            pixel_accuracies.append(class_accuracy)

        mean_accuracy = torch.tensor(pixel_accuracies).mean().item()

        return torch.tensor(mean_accuracy)

class DiceCoefficient(Metric):
    def __init__(self, use_cuda: bool = True):
        super().__init__("Dice Coefficient", use_cuda=use_cuda)
        self.name = "Dice Coefficient"

    def calculate_dice_coefficient(self, pred_mask, target_mask, threshold=0.5):
        pred_mask = (pred_mask > threshold).float()
        target_mask = (target_mask > threshold).float()

        intersection = torch.sum(pred_mask * target_mask)
        total_predicted = torch.sum(pred_mask)
        total_target = torch.sum(target_mask)

        dice_coefficient = (2 * intersection) / (total_predicted + total_target + 1e-6)

        return dice_coefficient.item()

    def batch_compute(self, inp: List[torch.Tensor]):
        pred_mask, target_mask = inp

        pred_mask = torch.argmax(pred_mask, dim=1)
        dice_coefficient = self.calculate_dice_coefficient(pred_mask, target_mask, threshold=0.5)

        return torch.tensor(dice_coefficient)
