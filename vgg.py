import torch.nn as nn

class VGG(nn.Module):
    '''
    creating VGG-like architechtures taking as input pattern
    variable described in __init__
    '''
    def __init__(self, pattern):
        '''
        pattern - list of integers and strings "pooling". Integer
        value is for number of convolutionals and "pooling" is for
        pooling layer
        '''
        super(VGG, self).__init__()
        self.vgg = self.create_layers(pattern)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.vgg(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def create_layers(self, pattern):
        layers = []
        in_channels = 3
        for out_channels in pattern:
            if out_channels == 'pooling':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                assert type(out_channels) == int
                layers.extend([nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(inplace=True)])
                in_channels = out_channels
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        
        return nn.Sequential(*layers)