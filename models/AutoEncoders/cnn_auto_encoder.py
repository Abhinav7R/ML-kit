"""
This is the CNN Auto Encoder
"""

import torch.nn as nn

class CNNAutoEncoder(nn.Module):
    def __init__(self):
        super(CNNAutoEncoder, self).__init__()

        # Encoder path: 28x28 -> 14x14 -> 7x7 -> 4x4
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            
            # 14x14 -> 7x7
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            
            # 7x7 -> 4x4
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),  
        )

        # latent dim = 16*4*4 = 256

        # Decoder path: 4x4 -> 7x7 -> 14x14 -> 28x28
        self.decoder = nn.Sequential(
            
            # 4x4 -> 7x7
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(True),
            
            # 7x7 -> 14x14
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x