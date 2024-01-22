from loss import fixed_loss
from cbdnet import Network
from utils import AverageMeter
import os
import time
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, PILToTensor
from torch.utils.data import DataLoader

from dataset import TID2013
# from models import CBDNetwork, ReconstructionNetwork, NESubNetwork
# from trainer import BaselineTrainer
# from losses import TotalLoss, ReconstructionLoss
# from metrics import nMAE
transform = Compose([PILToTensor(),
                     lambda z: z.to(dtype=torch.float32) /
                     127.5 - 1  # Normalize between -1 and 1
                     ])

# from dataset.loader import Real, Syn


parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--bs', default=8, type=int, help='batch size')
parser.add_argument('--ps', default=224, type=int, help='patch size')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--epochs', default=30, type=int, help='sum of epochs')
args = parser.parse_args()


def train(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    model.train()

    for (noise_img, clean_img, sigma_img, flag) in train_loader:
        input_var = noise_img.cuda()
        target_var = clean_img.cuda()
        sigma_var = sigma_img.cuda()
        flag_var = flag.cuda()

        noise_level_est, output = model(input_var)

        loss = criterion(output, target_var, noise_level_est,
                         sigma_var, flag_var)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


if __name__ == '__main__':
    save_dir = './save_model/'

    model = Network()
    model.cuda()
    model = nn.DataParallel(model)

    # if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
    # 	# load existing model
    # 	model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
    # 	print('==> loading existing model:', os.path.join(save_dir, 'checkpoint.pth.tar'))
    # 	model.load_state_dict(model_info['state_dict'])
    # 	optimizer = torch.optim.Adam(model.parameters())
    # 	optimizer.load_state_dict(model_info['optimizer'])
    # 	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 	scheduler.load_state_dict(model_info['scheduler'])
    # 	cur_epoch = model_info['epoch']
    # else:
    # if not os.path.isdir(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # create model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    cur_epoch = 0

    criterion = fixed_loss()
    criterion.cuda()

    dataset = TID2013(transform=transform, crop_size=(224, 224))

    train_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    # train_dataset = Real('./data/SIDD_train/', 320, args.ps) + Syn('./data/Syn_train/', 100, args.ps)
    # train_loader = torch.utils.data.DataLoader(
    # 	train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    for epoch in range(cur_epoch, args.epochs + 1):
        print('Training started')
        loss = train(train_loader, model, criterion, optimizer)
        scheduler.step()

        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()},
            os.path.join(save_dir, 'checkpoint.pth.tar'))

        print('Epoch [{0}]\t'
              'lr: {lr:.6f}\t'
              'Loss: {loss:.5f}'
              .format(
                  epoch,
                  lr=optimizer.param_groups[-1]['lr'],
                  loss=loss))
