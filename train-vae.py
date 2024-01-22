
import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from dataset import CropSegmentationDataset
from vae import VariationalAutoEncoder
from loss import VAELoss
from trainer import VAETrainer
import numpy as np
from metrics import CustomIoUMetric
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

transform_images = Compose([
    Resize((64, 64)),
    lambda z: torch.from_numpy(np.array(z, copy=True)).to(dtype=torch.float32) / 255  # Normalize between 0 and 1
])

transform_masks = Compose([
    Resize((64, 64)),
    #lambda z: torch.from_numpy(np.array(z, copy=True)).to(dtype=torch.float32) / (torch.max(z) + 1e-8) # Normalize between 0 and 1
])

batch_size = 8
lr = 1e-4
epoch = 100
input_channel = 3
latent_dims = 2

# Load data and generate batches
train_dataset = CropSegmentationDataset(set_type="train", transform=transform_images, target_transform=transform_masks)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CropSegmentationDataset(set_type="val", transform=transform_images, target_transform=transform_masks)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Implement VAE model:
# TODO: complete parameters and implement model forward pass + sampling
output_channels = train_dataset.get_class_number()
model = VariationalAutoEncoder(latent_dim=latent_dims, 
                               input_channels=input_channel, 
                               output_channels=output_channels)

# Implement loss function:
# TODO: implement the loss function as presented in the course

loss = VAELoss()

# Choose optimize:
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Implement the trainer
trainer = VAETrainer(model=model, loss=loss, optimizer=optimizer)

# Do the training
trainer.fit(train_loader, epoch=epoch)

torch.save(model.state_dict(), 'model_vae.pth')
