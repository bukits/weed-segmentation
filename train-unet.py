
import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from dataset import CropSegmentationDataset
from unet import UNet
from vae import VariationalAutoEncoder
from loss import UNetLoss
from trainer import UNETTrainer
import numpy as np
from metrics import CustomIoUMetric

transform_images = Compose([
    Resize((512, 512)),  # Image of size 64x64 for faster training
    lambda z: torch.from_numpy(np.array(z, copy=True)).to(dtype=torch.float32) / 255  # Normalize between 0 and 1
])

transform_masks = Compose([
    Resize((512, 512)),  # Image of size 64x64 for faster training
    #lambda z: torch.from_numpy(np.array(z, copy=True)).to(dtype=torch.float32) / (torch.max(z) + 1e-8) # Normalize between 0 and 1
])

batch_size = 8
lr = 1e-4
epoch = 10


# Load data and generate batches
train_dataset = CropSegmentationDataset(set_type="train", transform=transform_images, target_transform=transform_masks)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CropSegmentationDataset(set_type="val", transform=transform_images, target_transform=transform_masks)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Implement VAE model:
# TODO: complete parameters and implement model forward pass + sampling
output_channels = train_dataset.get_class_number()
model = UNet(n_channels=3, n_classes=output_channels)

# Implement loss function:
# TODO: implement the loss function as presented in the course

loss = UNetLoss()

# Choose optimize:
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Implement the trainer
trainer = UNETTrainer(model=model, loss=loss, optimizer=optimizer)

# Do the training
trainer.fit(train_loader, epoch=epoch)

torch.save(model.state_dict(), 'model_unet.pth')

# Compute metrics
# TODO: implement metrics
#metrics = [CustomIoUMetric()]

#scores = trainer.eval(val_loader, metrics)
#print(scores[0])

print("job's done.")
