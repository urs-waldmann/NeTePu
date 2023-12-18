import torch
from torch import nn
import torch.nn.functional as functional


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.LeakyReLU(0.1)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet18Enc(nn.Module):
    def __init__(self, num_Blocks=None, z_dim=1024, number_channels=3):
        super().__init__()
        if num_Blocks is None:
            num_Blocks = [2, 2, 2, 2, 2]
        self.in_planes = 16

        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(number_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlockEnc, 16, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 32, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 64, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 128, num_Blocks[3], stride=2)
        self.layer5 = self._make_layer(BasicBlockEnc, 256, num_Blocks[4], stride=2)
        self.linear = nn.Linear(256, 4 * z_dim)
        self.linear2 = nn.Linear(4 * z_dim, 2 * z_dim)

        self.relu = nn.LeakyReLU(0.1)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        z_texture = self.reparameterize(mu, logvar)

        return mu, logvar, z_texture

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, square root is divide by two
        epsilon = torch.randn_like(std)

        return epsilon * std + mean


def get_texture_encoder(CFG):
    CFG_enc = CFG['encoder']
    if CFG['texture']['data'] == 'h36m':
        encoder = ResNet18Enc(
            z_dim=CFG_enc.get('lat_dim'),
            number_channels=3
        )
    else:
        encoder = ResNet18Enc(
            z_dim=CFG_enc.get('lat_dim'),
            number_channels=6
        )

    return encoder
