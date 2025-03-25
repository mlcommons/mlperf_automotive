"""
Modifications by MLCommons from SSD-Pytorch (https://github.com/uvipen/SSD-pytorch) author: Viet Nguyen (nhviet1009@gmail.com)
Copyright 2024 MLCommons Association and Contributors

MIT License

Copyright (c) 2021 Viet Nguyen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
#from torchvision.models.mobilenet import mobilenet_v2, InvertedResidual
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.mobilenetv2 import InvertedResidual

class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.num_classes, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs


class ResNet(nn.Module):
    def __init__(self, model_config, pretrained=False):
        super().__init__()
        # Grigori removed pretrained since we don't need it for ABTF
        backbone = resnet50(pretrained=pretrained)
        if 'feature_out_channels' in model_config:
            self.out_channels = model_config['feature_out_channels']
        else:
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])
        for block in model_config['backbone']:
            if block['block_no'] == 0:
                conv_layer = self.feature_extractor[block['block_no']]
                conv_layer.kernel_size = block['conv']['kernel_size']
                conv_layer.stride = block['conv']['stride']
                conv_layer.padding = block['conv']['padding']
            else:
                conv_block = self.feature_extractor[block['block_no']][block['layer']]
                for conv in block['conv']:
                    for k, v in conv.items():
                        conv_layer = getattr(conv_block, 'conv' + str(k))
                        conv_layer.kernel_size = v['kernel_size']
                        conv_layer.stride = v['stride']
                        conv_layer.padding = v['padding']
            if 'downsample' in block and block['downsample'] is not None:
                conv_block.downsample[0].stride = block['downsample']['stride']
        #conv4_block1 = self.feature_extractor[-1][0]
        #conv4_block1.conv1.stride = (1, 1)
        #conv4_block1.conv2.stride = (1, 1)
        #conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD(Base):
    def __init__(self, model_config, backbone, num_classes=81):
        super().__init__()

        self.feature_extractor = backbone
        self.num_classes = num_classes
        self.config = model_config
        self.pre_blocks = None
        if 'pre_backbone' in self.config and self.config['pre_backbone']:
            self.pre_backbone([3, 3, 3], [3, 3, 3], self.config['pre_backbone'])
        if 'feature_in_channels' in model_config:
            in_channels = model_config['feature_out_channels']
        else:
            in_channels = [256, 256, 128, 128, 128]
        self._build_additional_features(self.feature_extractor.out_channels, in_channels)
        self.num_defaults = self.config['head']['num_defaults']
        self.loc = []
        self.conf = []

        for nd, oc, loc_config, conf_config in zip(self.num_defaults, self.feature_extractor.out_channels, self.config['head']['loc'], self.config['head']['conf']):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=loc_config['kernel_size'], stride=conf_config['stride'], padding=loc_config['padding']))
            self.conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=conf_config['kernel_size'], stride=conf_config['stride'], padding=conf_config['padding']))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self.init_weights()

    def pre_backbone(self, input_channels, output_channels, pre_layers):
        self.pre_blocks = []
        blocks = []
        for i, conv in enumerate(pre_layers):
            blocks.append(nn.Conv2d(input_channels[i], output_channels[i], conv['kernel_size'], conv['stride'], conv['padding']))
            blocks.append(nn.BatchNorm2d(output_channels[i]))
            blocks.append(nn.ReLU(inplace=True))
        self.pre_blocks = nn.Sequential(*blocks)
    
    def _build_additional_features(self, input_size, output_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(
                zip(input_size[:-1], input_size[1:], output_size)):
            middle_block = self.config['middle_blocks'][i]
            layer = nn.Sequential(
                nn.Conv2d(input_size, channels, kernel_size=middle_block['kernel_size'][0], padding=middle_block['padding'][0], stride=middle_block['stride'][0], bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, output_size, kernel_size=middle_block['kernel_size'][1], padding=middle_block['padding'][1], stride=middle_block['stride'][1], bias=False),
                nn.BatchNorm2d(output_size),
                nn.ReLU(inplace=True),
            )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)


    def forward(self, x):
        if self.pre_blocks is not None:
            x = self.pre_blocks(x)
        x = self.feature_extractor(x)
        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return locs, confs


feature_maps = {}


class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Grigori removed pretrained since we don't need it for ABTF
        self.feature_extractor = mobilenet_v2(pretrained=False).features
        self.feature_extractor[14].conv[0][2].register_forward_hook(self.get_activation())

    def get_activation(self):
        def hook(self, input, output):
            feature_maps[0] = output.detach()

        return hook

    def forward(self, x):
        x = self.feature_extractor(x)
        return feature_maps[0], x


def SeperableConv2d(in_channels, out_channels, kernel_size=3):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                  groups=in_channels, padding=padding),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def StackedSeperableConv2d(ls_channels, multiplier):
    out_channels = 6 * multiplier
    layers = [SeperableConv2d(in_channels=in_channels, out_channels=out_channels) for in_channels in ls_channels]
    layers.append(nn.Conv2d(in_channels=ls_channels[-1], out_channels=out_channels, kernel_size=1))
    return nn.ModuleList(layers)


class SSDLite(Base):
    def __init__(self, backbone=MobileNetV2(), num_classes=81, width_mul=1.0):
        super(SSDLite, self).__init__()
        self.feature_extractor = backbone
        self.num_classes = num_classes

        self.additional_blocks = nn.ModuleList([
            InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
            InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
            InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
            InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
        ])
        header_channels = [round(576 * width_mul), 1280, 512, 256, 256, 64]
        self.loc = StackedSeperableConv2d(header_channels, 4)
        self.conf = StackedSeperableConv2d(header_channels, self.num_classes)
        self.init_weights()


    def forward(self, x):
        y, x = self.feature_extractor(x)
        detection_feed = [y, x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return locs, confs
