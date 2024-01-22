import torch.nn as nn
import torch.nn.functional as F

class UNetLoss(nn.Module):
    def _init_(self, weight=None, size_average=True):
        super(UNetLoss, self)._init_()

    def forward(self, inputs, targets, smooth=1):
        
        BCE = F.binary_cross_entropy_with_logits(inputs, targets)
        
        inputs = F.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return BCE + dice_loss

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_loss_func = nn.CrossEntropyLoss(reduction='sum')
    def forward(self, ground_truth_mask, pred_mask):
        seg_loss = self.seg_loss_func(pred_mask, ground_truth_mask)
        return seg_loss