
import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from dataset import CropSegmentationDataset
from vae import VariationalAutoEncoder
import numpy as np
from trainer import VAETrainer
from loss import VAELoss
import matplotlib.pyplot as plt
from skimage.transform import resize

transform = Compose([
    Resize((64, 64)),
    lambda z: torch.from_numpy(np.array(z, copy=True)).to(dtype=torch.float32) / 255
])

transform_masks = Compose([
    Resize((64, 64)),
])

batch_size = 8
lr = 1e-4
input_channel = 3
latent_dims = 2

val_dataset = CropSegmentationDataset(set_type="val", transform=transform, target_transform=transform_masks)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

output_channels = val_dataset.get_class_number()
model = VariationalAutoEncoder( input_channels=input_channel, 
                                 output_channels=output_channels)
model.load_state_dict(torch.load('model_vae.pth'))

loss = VAELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
trainer = VAETrainer(model=model, loss=loss, optimizer=optimizer)

image = val_dataset[410][0].unsqueeze(0)
image = image.cuda()

mask_true = val_dataset[410][1].squeeze(0)
mask_true = mask_true.cpu().numpy()

mask_predicted = model(image)
print(mask_predicted.shape)
mask_predicted = mask_predicted.squeeze(0)
mask_predicted = mask_predicted.cpu().detach().numpy()

mask_predicted = torch.from_numpy(mask_predicted)
mask_predicted = mask_predicted.long()
mask_single_channel = torch.argmax(mask_predicted, dim=0)

# Convert to numpy array
mask_predicted = mask_single_channel.cpu().numpy()

image = image.squeeze().cpu().numpy().transpose(1, 2, 0)

new_size = (1024, 1024)

image_resized = resize(image, new_size, anti_aliasing=False)
resized_mask_true = resize(mask_true, new_size, anti_aliasing=True)
resized_mask_np = resize(mask_predicted, new_size, anti_aliasing=True)

# Plot the resized true mask
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Input Image')

plt.subplot(1, 3, 2)
plt.imshow(mask_true, cmap='jet')
plt.title('True Mask')

# Plot the resized predicted mask
plt.subplot(1, 3, 3)
plt.imshow(mask_predicted, cmap='jet')
plt.title('Predicted Mask')

plt.show()
