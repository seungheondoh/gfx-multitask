import torch
import torchaudio
from torch import nn
from gfx.modules.ops import Transformer, Res2DMaxPoolModule

class ResNet(nn.Module):
    def __init__(self,
                audio_representation,
                n_channels = 128,
                spec_type="mel",
                n_class=23
        ):
        super(ResNet, self).__init__()
        # Input preprocessing
        self.audio_representation = audio_representation
        self.spec_type = spec_type
        # Input embedding
        self.input_bn = nn.BatchNorm2d(1)
        self.layer1 = Res2DMaxPoolModule(1, n_channels, pooling=(2, 2))
        self.layer2 = Res2DMaxPoolModule(n_channels, n_channels, pooling=(2, 2))
        self.layer3 = Res2DMaxPoolModule(n_channels, n_channels*2, pooling=(2, 2))
        self.layer4 = Res2DMaxPoolModule(n_channels*2, n_channels*2, pooling=(2, 2))
        self.layer5 = Res2DMaxPoolModule(n_channels*2, n_channels*2, pooling=(2, 2))
        self.layer6 = Res2DMaxPoolModule(n_channels*2, n_channels*2, pooling=(2, 2))
        self.layer7 = Res2DMaxPoolModule(n_channels*2, n_channels*4, pooling=(2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to_latent = nn.Identity()
                # Dense
        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dense2 = nn.Linear(n_channels*4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        
    def forward(self, x):
        if self.spec_type == "mel":
            spec = self.audio_representation.melspec(x)
            spec = spec.unsqueeze(1)
        elif self.spec_type == "stft":
            spec = None
        spec = self.input_bn(spec) # B x L x D
        out = self.layer1(spec)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out) # B x D x F x T
        out = self.avg_pool(out)
        x = out.squeeze(-1).squeeze(-1)
        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)
        return x