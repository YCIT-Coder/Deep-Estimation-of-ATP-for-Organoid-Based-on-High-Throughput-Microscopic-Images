import torch
import torch.nn as nn
from Model.layers import Conv2D, Attention
import numpy as np


class MeshIns(nn.Module):
    def __init__(self, in_channel, out_channel, category, patches, num_layers, num_stack, image_size):
        super(MeshIns, self).__init__()
        self.patches = patches
        self.image_size = image_size
        feature_extractor = []
        for i in range(num_layers):
            feature_extractor.append(self._make_layer(in_channel, 32 * (2 ** i), num_stack))
            in_channel = 32 * (2 ** i)
            feature_extractor.append(nn.MaxPool2d(2))
        feature_extractor.append(Conv2D(in_channel, in_channel, ac=True))
        feature_extractor.append(nn.AvgPool2d(image_size // (2 ** num_layers)))
        feature_extractor.append(Conv2D(in_channel, out_channel, ac=True))
        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.attention = Attention(out_channel, out_channel * 2, patches)
        self.classifier = nn.Sequential(
            nn.Linear(out_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, category),
            nn.Sigmoid(),
        )

    def _make_layer(self, in_channel, out_channel, num_stack):
        layers = []
        for i in range(num_stack):
            layers.append(Conv2D(in_channel, out_channel, ac=True))
            in_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), np.int32(np.sqrt(self.patches)),
                   self.image_size, np.int32(np.sqrt(self.patches)),
                   self.image_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.contiguous()
        x = x.view(x.size(0), -1, x.size(-3), x.size(-2), x.size(-1))
        out = x.view(-1, x.size(2), x.size(3), x.size(4))
        out = self.feature_extractor(out).squeeze(-1).squeeze(-1)
        weight = self.attention(out).unsqueeze(-1)  # large weights refer to key instances
        out = out.view(-1, self.patches, out.size(-1))
        out = weight * out
        # print(weight)
        out = torch.sum(out, 1)
        out = self.classifier(out).squeeze(-1)
        return out


class DeepIns(nn.Module):
    def __init__(self, in_channel, hidden, category, num_layer, image_size, patches=64):
        super(DeepIns, self).__init__()
        feature_extractor = []
        temp = int(np.log2(patches)) // 2
        num_layer = 9 - temp
        factor = 2 ** (temp + 2)
        # print('temp{},patch{},layer{},factor{}'.format(temp, patches, num_layer, factor))
        for i in range(num_layer):
            feature_extractor.append(Conv2D(in_channel, factor * (2 ** i), ac=True))
            feature_extractor.append(nn.MaxPool2d(2, 2))
            in_channel = factor * (2 ** i)
            # print(in_channel)

        hidden = hidden // patches
        feature_extractor.append(Conv2D(in_channel, hidden, ac=True))
        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.attention = Attention(hidden, hidden * 2, (image_size // (2 ** num_layer)) ** 2)
        # print('patch', (image_size // (2 ** num_layer)) ** 2)
        hidden = hidden * patches

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden * 2, hidden * 2 * 2),
            nn.BatchNorm1d(hidden * 2 * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2 * 2, category),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.permute((0, 2, 3, 1))
        out = out.contiguous()
        out = out.view(-1, out.size(-1))
        # weight = self.attention(out).unsqueeze(-1)
        out = out.view(x.size(0), -1, out.size(-1))
        # out = weight * out
        out = torch.split(out, 1, 1)
        out = torch.cat(out, -1).squeeze(1)
        out = self.classifier(out).squeeze(-1)
        return out


if __name__ == '__main__':
    pass