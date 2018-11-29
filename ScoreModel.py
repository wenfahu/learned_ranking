import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Block(nn.Module):

    """Docstring for Block. """

    def __init__(self, planes, dilation=1):
        """TODO: to be defined1. """
        super(Block, self).__init__()
        self.conv_h = nn.Conv2d(planes, planes,
                kernel_size=(3, 1),
                dilation = dilation,
                groups=planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv_w = nn.Conv2d(planes, planes,
                kernel_size=(1, 3),
                stride=(1, 2),
                dilation = dilation,
                groups=planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        """TODO: Docstring for forward.

        :arg1: TODO
        :returns: TODO

        """
        x = self.conv_h(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv_w(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x

        

class ScoreNetwork(nn.Module):

    """Docstring for ScoreNetwork. """

    def __init__(self, planes):
        """TODO: to be defined1. """
        super(ScoreNetwork, self).__init__()
        self.layer1 = Block(planes)
        self.layer2 = Block(planes)
        self.layer3 = Block(planes)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        """TODO: Docstring for forward.

        :x: TODO
        :returns: TODO

        """
        pdb.set_trace()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.sigmoid(x)
        return x.squeeze()
        
