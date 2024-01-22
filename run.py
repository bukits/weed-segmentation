import argparse
import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from dataset import CropSegmentationDataset
from vae import VariationalAutoEncoder
import numpy as np
import os
from unet import UNet
from trainer import VAETrainer, UNETTrainer
from metrics import CustomIoUMetric
import matplotlib.pyplot as plt
import random
from scipy.ndimage import binary_erosion

def extract_model_name(model_path):
    filename = os.path.basename(model_path)
    model_name = os.path.splitext(filename)[0]

    if model_name.startswith('model_'):
        model_name = model_name[len('model_'):]

    return model_name

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on test images.')
    parser.add_argument('--test_folder', type=str, required=True, help='Path to the test folder')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--batch_size', type=int, help='Batch size for the testing', default=8)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    test_folder_path = args.test_folder
    model_path = args.model_path
    model_name = extract_model_name(model_path)
    batch_size = args.batch_size
    input_channel = 3
    num_examples = 5

    if model_name == "vae":
        transform = Compose([
            Resize((64, 64)),
            lambda z: torch.from_numpy(np.array(z, copy=True)).to(dtype=torch.float32) / 255
        ])

        transform_masks = Compose([
            Resize((64, 64)),
        ])
    elif model_name == "unet":
        transform = Compose([
            Resize((512, 512)),
            lambda z: torch.from_numpy(np.array(z, copy=True)).to(dtype=torch.float32) / 255
        ])

        transform_masks = Compose([
            Resize((512, 512)),
        ])

    #!!!!!!!!!!!!!!change val to test
    test_dataset = CropSegmentationDataset(root_path=test_folder_path, set_type="val", transform=transform, target_transform=transform_masks)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dataset_length = len(test_dataset)

    output_channels = test_dataset.get_class_number()

    print('Loading model...')
    if model_name == "vae":
        model = VariationalAutoEncoder(input_channels=input_channel, 
                                    output_channels=output_channels)
        model.load_state_dict(torch.load(model_path))
        trainer = VAETrainer(model=model, loss=None, optimizer=None)
    elif model_name == "unet":
        model = UNet(n_channels=input_channel, n_classes=output_channels)
        model.load_state_dict(torch.load(model_path))
        trainer = UNETTrainer(model=model, loss=None, optimizer=None)
    print('Model is loaded.')

    print('Metric calculation has been started...')
    iou_metric = CustomIoUMetric()
    metrics = [iou_metric]
    scores, names = trainer.eval(test_loader, metrics)
    print('Metric calculation is done.')

    output_file_path = "metrics.txt"
    with open(output_file_path, 'w') as file:
        file.write('Metric calculation results:\n')
        for i, score in enumerate(scores):
            metric_name = names[i]
            file.write(f"{metric_name}: {score}\n")

    print(f"Scores written to {output_file_path}")

    selected_indices = random.sample(range(0, dataset_length), num_examples)
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 18))

    for i, index in enumerate(selected_indices):
        image = test_dataset[index][0].unsqueeze(0)
        image = image.cuda()
        image = image.squeeze().cpu().numpy().transpose(1, 2, 0)

        actual_batch_index = index % batch_size
        batch_index = index // batch_size

        mask_true = test_dataset[index][1].squeeze(0)
        mask_true = mask_true.cpu().numpy()

        mask_predicted_batch = trainer.predicted_masks[batch_index]
        mask_predicted = mask_predicted_batch[actual_batch_index, :, :, :]
        mask_predicted = mask_predicted.squeeze(0)
        mask_predicted = mask_predicted.cpu().detach().numpy()

        mask_predicted = torch.from_numpy(mask_predicted)
        mask_predicted = mask_predicted.float()
        mask_single_channel = torch.argmax(mask_predicted, dim=0)
        mask_predicted = mask_single_channel.cpu().numpy()

        if model_name == "unet":
            mask_predicted = binary_erosion(mask_predicted, structure=np.ones((5,5)))

        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Input Image')

        axes[i, 1].imshow(mask_true, cmap='jet')
        axes[i, 1].set_title('True Mask')

        axes[i, 2].imshow(mask_predicted, cmap='jet')
        axes[i, 2].set_title('Predicted Mask')

    plt.tight_layout()
    plt.show()