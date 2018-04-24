#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

# import torch as th
# import torchvision as tv

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

from layers import *


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    """
    Basic block adopted from torchvision(https://github.com/pytorch/vision).
    x -> conv3x3_bn_relu -> conv3x3_bn_<+x[downsample]>_relu
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block adopted from torchvision.
    x -> conv1x1_bn -> conv3x3_bn -> conv_bn_<+x_[downsample]>_relu
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet adopted from torchvision.
    """
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetAsBlock(nn.Module):
    """
        Resnet block modified from ResNet class of torchvision.
    """
    residual_layers = []

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetAsBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def block_pass(self, x, block_func, residual=True):
        x = block_func(x)
        if residual:
            self.residual_layers.append(x)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block_pass(x, self.layer1)
        x = self.block_pass(x, self.layer2)
        x = self.block_pass(x, self.layer3)
        x = self.block_pass(x, self.layer4, False)

        return x


class DeconvByBilinearUpsamplingConcat(nn.Module):
    """
    Deconvolution layer, consisting a convolution layer to reduce channels,
    a bilinear upsampling op and concatenating layers from ResNet block.
    This idea is adopted from [FOTS]() paper, though we are not able to know
    how did they implement this, thus this class comes with NO GUARANTEE.
    """

    def __init__(self, in_planes, residual_layers, repeat, channel_shrunk_factor, stride=1):
        super().__init__()
        self.channel_shrunk_factor = channel_shrunk_factor
        self.now_planes = in_planes
        self.bilinear_upsampling = bilinear_upsampling_2x()
        self.residual_layers = residual_layers
        self.repeat = repeat

    def forward(self, x):
        for i in range(self.repeat):
            self.conv = conv3x3(self.now_planes, self.now_planes // self.channel_shrunk_factor)
            x = self.conv(x)
            x = self.bilinear_upsampling(x)
            if i < len(self.residual_layers):
                residual = self.residual_layers[- (i + 1)]
                # Pad residuals with odd image h/w
                if x.size()[2] > residual.size()[2]:
                    residual = nn.ZeroPad2d((0, 0, x.size()[2] - residual.size()[2], 0))(residual)
                if x.size()[3] > residual.size()[3]:
                    residual = nn.ZeroPad2d((0, 0, 0, x.size()[3] - residual.size()[3]))(residual)
                x = torch.cat((x, residual), 1)
            self.now_planes //= 2
        return x


class DeconvByBilinearUpsampling(nn.Module):
    """
    Deconvolution layer, consisting a convolution layer to reduce channels,
    a bilinear upsampling op and concatenating layers from ResNet block.
    This idea is adopted from [FOTS]() and [FPN]() paper, though we are not
    able to know how FOTS implements this so we followed the FPN style,
    thus this class comes with NO GUARANTEE.
    """

    def __init__(self, out_channel=256, stride=1):
        super().__init__()
        self.out_channel = out_channel
        # self.conv3 = conv3x3(out_channel, out_channel)
        self.bilinear_upsampling = bilinear_upsampling_2x()
        self.conv1_1 = [conv1x1(3072, 512), conv1x1(1024, 256), conv1x1(512, 128)]
        self.conv3_3 = [conv3x3(512, 512), conv3x3(256, 256), conv3x3(128, 128)]
        for i in range(len(self.conv1_1)):
            self.add_module("conv1_1-" + str(i), self.conv1_1[i])
            self.add_module("conv3_3-" + str(i), self.conv3_3[i])

    def forward(self, x, residual_layers):
        repeat = len(residual_layers)
        # if x.size()[1] != self.out_channel:
        #     x = conv1x1(x.size()[1], self.out_channel)(x)
        for i in range(repeat):
            x = self.bilinear_upsampling(x)
            # Adjust residual channels
            residual = residual_layers[- (i + 1)]
            # Pad residuals with odd image h/w - no need since images are resized to resolution of 32-multiples
            # if x.size()[2] > residual.size()[2]:
            #     residual = nn.ZeroPad2d((0, 0, x.size()[2] - residual.size()[2], 0))(residual)
            # if x.size()[3] > residual.size()[3]:
            #     residual = nn.ZeroPad2d((0, 0, 0, x.size()[3] - residual.size()[3]))(residual)
            # # FPN said they perform element-wise addition, while FOTS uses the word "concatenate"
            # We adopt concatenating of EAST style.
            # x = torch.add(x, residual)
            x = torch.cat((x, residual), 1)
            x = self.conv1_1[i](x)
            x = self.conv3_3[i](x)
            # x = conv1x1(x.size()[1], x.size()[1] // 3)(x)
            # x = conv3x3(x.size()[1], x.size()[1])(x)
        return x


def resnet50_block():
    """
    Constructs a ResNet-50 block, removing avg_pool and fc layers.
    """
    return ResNetAsBlock(Bottleneck, [3, 4, 6, 3])


# def deconv_block_previous(residual_layers, in_planes, repeat=3, channel_shrunk_factor=2):
#     return DeconvByBilinearUpsampling(in_planes, residual_layers, repeat, channel_shrunk_factor)


def deconv_block():
    return DeconvByBilinearUpsampling()


# The following methods are adopted from torchvision codes.
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
