"""
Model definitions for the SSL bearing fault diagnosis pipeline.

This module provides:
    - Plain CNN classifiers (CNN_128, CNN_64) for STFT-image inputs of
      different resolutions (SLRA uses 128x128, CWRU/HUST/PU use 64x64)
    - WideResNet, an alternative backbone selectable via --model wrn
    - Convolutional autoencoder classifiers (CAE_128, CAE_64) used by the
      'hcae' semi-supervised method, which jointly reconstructs unlabeled
      inputs and classifies labeled ones
    - ModelEMA, an exponential moving average shadow model used for
      evaluation/inference stability across all training methods
    - create_model(args), the single entry point main.py uses to build the
      right backbone for the chosen dataset/model/method combination
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


################### CNN
class CNN_128(nn.Module):
    """CNN classifier for 128x128 STFT inputs (used with the 'slra' dataset)."""

    def __init__(self, num_classes):
        super(CNN_128, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, 2))

        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024), nn.Tanh(),
            nn.Linear(1024, 256), nn.Tanh())
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.classifier(x)
        # returns (penultimate feature, class logits) — the feature is used
        # for t-SNE visualization and by the 'proposed' method's triplet loss
        return x, self.fc(x)


class CNN_64(nn.Module):
    """CNN classifier for 64x64 STFT inputs (used with 'slra'/'cwru'/'hust'/'pu')."""

    def __init__(self, num_classes):
        super(CNN_64, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, 2))

        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024), nn.Tanh(),
            nn.Linear(1024, 256), nn.Tanh())
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.classifier(x)
        return x, self.fc(x)


########################WRN
class BasicBlock(nn.Module):
    """Pre-activation residual block used to build WideResNet."""

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    """Stack of `nb_layers` BasicBlocks; the building unit of each WideResNet stage."""

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate,
                                      activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate,
                                activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """WideResNet-28-2 backbone, selectable via --model wrn (SLRA/128x128 only)."""

    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(512, num_classes)
        self.nChannels = 512

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)  # 64x64: 8  / 128x128: 16
        out = out.view(-1, self.nChannels)

        return out, self.fc(out)


class ModelEMA(object):
    """
    Exponential moving average of model weights.

    Kept as a separate deep-copied model (`self.ema`) that is updated after
    every optimizer step; args.use_ema controls whether this shadow model
    (rather than the live model) is used for validation/testing/checkpointing.
    """

    def __init__(self, args, model, decay):
        self.ema = deepcopy(model)
        self.ema.to(args.device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])


###########################################HCAE
class CAE_128(nn.Module):
    """
    Convolutional autoencoder + classifier for 128x128 inputs, used by the
    'hcae' method: forward() returns (reconstruction, class logits) so the
    same model can be trained with a reconstruction loss on unlabeled data
    and a classification loss on labeled data.
    """

    def __init__(self, num_classes):
        super(CAE_128, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(1, 32, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024), nn.Tanh(),
            nn.Linear(1024, 256), nn.Tanh()
        )

        self.classifier = nn.Linear(256, num_classes)

        self.decoder_fc = nn.Sequential(
            nn.Linear(256, 1024), nn.Tanh(),
            nn.Linear(1024, 128 * 8 * 8), nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.Tanh(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.Tanh(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.Tanh(),

            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.features(x)
        z = z.view(x.size(0), -1)
        z = self.encoder_fc(z)

        y = self.classifier(z)

        recon = self.decoder_fc(z)
        recon = recon.view(z.size(0), 128, 8, 8)
        recon = self.decoder(recon)  

        return recon, y


class CAE_64(nn.Module):
    """Same as CAE_128 but sized for 64x64 inputs (CWRU/HUST/PU datasets)."""

    def __init__(self, num_classes):
        super(CAE_64, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2,2)
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(128*8*8, 1024), nn.Tanh(),
            nn.Linear(1024, 256), nn.Tanh()
        )

        self.classifier = nn.Linear(256, num_classes)

        self.decoder_fc = nn.Sequential(
            nn.Linear(256, 1024), nn.Tanh(),
            nn.Linear(1024, 128*8*8), nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.features(x)
        z = z.view(x.size(0), -1)
        z = self.encoder_fc(z)
        y = self.classifier(z)

        recon = self.decoder_fc(z)
        recon = recon.view(z.size(0), 128, 8, 8)
        recon = self.decoder(recon)

        return recon, y

def create_model(args):
    """
    Build the backbone selected by args.dataset/args.model/args.method.

    All datasets now use 64x64 STFT inputs, so CNN_64/CAE_64 is the default
    backbone everywhere. WideResNet (--model wrn) is available for 'slra'.
    The 'hcae' method always uses the CAE_64 autoencoder variant regardless
    of --model. (CNN_128/CAE_128 are kept in this file for 128x128 STFT
    inputs, unused now that 'slra' switched to 64x64 — see get_data() /
    main.py's data_path for 'slra' if that ever changes back.)
    """
    if args.dataset == 'slra' and args.model == 'wrn':
        model = WideResNet(num_classes=args.num_classes)
    elif args.method == 'hcae':
        model = CAE_64(num_classes=args.num_classes)
    else:
        model = CNN_64(num_classes=args.num_classes)
    model = model.to(args.device)
    return model