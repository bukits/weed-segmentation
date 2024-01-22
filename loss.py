
import torch
import torch.nn as nn

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_loss_func = nn.CrossEntropyLoss(reduction='sum')
    def forward(self, ground_truth_mask, pred_mask):
        seg_loss = self.seg_loss_func(pred_mask, ground_truth_mask)
        return seg_loss