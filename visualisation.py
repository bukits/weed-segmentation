import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import torch

target = Image.open('dataset/train/labels/05-15_00028_P0030852.png', "r")
target = torch.from_numpy(np.array(target))
target = target.long()
plt.imshow(target, cmap='hot')
plt.show()