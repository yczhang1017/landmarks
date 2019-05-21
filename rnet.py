
import torch.nn as nn
import math
import collections
import torch.utils.model_zoo as model_zoo

__all__ = ['RNet', 'rnet18', 'rnet34', 'rnet50', 'rnet101','rnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
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


class RNet(nn.Module):

    def __init__(self, block, layers, num_classes=[491,499]):
        self.inplanes = 64
        super(RNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.num_classes=num_classes
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1, self.inplanes = self._make_layer(block, 64, layers[0])
        self.layer2, self.inplanes = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3, self.inplanes = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc=nn.ModuleList()
        self.layer4 =nn.ModuleList()
        for c in num_classes:
            self.layer4.append(self._make_layer(block, 512, layers[3], stride=2)[0])
            self.fc.append(nn.Linear(512 * block.expansion, c))
            
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

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes))

        return nn.Sequential(*layers), planes * block.expansion

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        y=[None]*2
        for i,c in enumerate(self.num_classes): 
            y[i] = self.layer4[i](x)
            y[i] = self.avgpool(y[i])
            y[i] = y[i].view(y[i].size(0), -1)
            y[i] = self.fc[i](y[i])
            
        return y


def load(model,url,num_classes):
    stateo=model_zoo.load_url(url)
    state=collections.OrderedDict()
    for _ in range(len(stateo)):
        k,v=stateo.popitem(last=False)
        if k.startswith("layer4."):
            for i,c in enumerate(num_classes):
                state[k[:6]+'.'+str(i)+k[6:]]=v
        elif k.startswith("fc.") and k.endswith('weight'):
            for i,c in enumerate(num_classes):
                state[k[:2]+'.'+str(i)+k[2:]]=v[:c,:]
        elif k.startswith("fc.") and k.endswith('bias'):
            for i,c in enumerate(num_classes):
                state[k[:2]+'.'+str(i)+k[2:]]=v[:c]
        else:
            state[k]=v
    model.load_state_dict(state)
    
    

def rnet18(pretrained=False,num_classes=[491,499], **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RNet(BasicBlock, [2, 2, 2, 2],num_classes=num_classes, **kwargs)
    if pretrained:
        load(model,model_urls['resnet18'],num_classes)
    return model


def rnet34(pretrained=False,num_classes=[491,499], **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RNet(BasicBlock, [3, 4, 6, 3],num_classes=num_classes, **kwargs)
    if pretrained:
        load(model,model_urls['resnet34'],num_classes)
    return model


def rnet50(pretrained=False,num_classes=[491,499], **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RNet(Bottleneck, [3, 4, 6, 3],num_classes=num_classes, **kwargs)
    if pretrained:
        load(model,model_urls['resnet50'],num_classes)
    return model



def rnet101(pretrained=False,num_classes=[491,499], **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RNet(Bottleneck, [3, 4, 23, 3],num_classes=num_classes, **kwargs)
    if pretrained:
        load(model,model_urls['resnet101'],num_classes)
    return model


def rnet152(pretrained=False,num_classes=[491,499], **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RNet(Bottleneck, [3, 8, 36, 3],num_classes=num_classes, **kwargs)
    if pretrained:
        load(model,model_urls['resnet152'],num_classes)
    return model
