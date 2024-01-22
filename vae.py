import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_rate=0.5):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.conv(x)
    
class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_rate=0.5):
        super(ConvTransposeBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()

        self.conv1 = ConvBlock(input_channels, 64, 4, 2, 1)
        self.conv2 = ConvBlock(64, 128, 4, 2, 1)
        self.conv3 = ConvBlock(128, 256, 4, 2, 1)
        self.conv4 = ConvBlock(256, 256, 4, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_channels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(2, 256 * 4 * 4)
        self.deconv1 = ConvTransposeBlock(256, 256, 4, 2, 1)
        self.deconv2 = ConvTransposeBlock(256, 128, 4, 2, 1)
        self.deconv3 = ConvTransposeBlock(128, 64, 4, 2, 1)
        self.deconv4 = ConvTransposeBlock(64, output_channels, 4, 2, 1)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x

class VariationalAutoEncoder(nn.Module):
    def __init__(self,
                  input_channels: int,
                  output_channels: int):
        super().__init__()
        self.encoder_module = Encoder(input_channels=input_channels)
        self.decoder_module = Decoder(output_channels=output_channels)

    def forward(self, x: torch.Tensor):
        """Forward pass for the network.

        :return: A torch.Tensor.
        """
        # Encoding
        x = self.encoder_module(x)
        # Decoding
        segmentation_x = self.decoder_module(x)

        return segmentation_x