import torch
from torch import nn
from torch.nn import  functional as F
from MobileNetV2 import MobileNetV2
import numpy
from itertools import chain
import timeit


class HeadDetector(nn.Module):

    def __init__(self, dropout=0.1):
        super(HeadDetector, self).__init__()

        self.net = MobileNetV2(include_top=False)
        state_dict = torch.load('/home/aaditya/PycharmProjects/Vistas/mobilenet_v2.pth.tar')
        self.net.load_state_dict(state_dict, strict=False)

        self.dropout = dropout
        self.attention = nn.Sequential(
            # nn.Dropout2d(p=self.dropout),
            nn.Conv2d(1280, 640, 1),
            nn.ReLU(),
            nn.BatchNorm2d(640),
            nn.Conv2d(640, 1, 1),
            nn.Sigmoid(),
            # nn.BatchNorm2d(1),
            # nn.Dropout2d(p=self.dropout),
        )

        self.common = nn.Sequential(
            nn.Conv2d(1280, 640, 1),
            nn.ReLU(),
            nn.BatchNorm2d(640)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(640, 2, 1),
            nn.LogSoftmax(dim=1)
        )

        self.regressor = nn.Sequential(
            nn.Conv2d(640, 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        # x *= self.attention(x)
        x = self.common(x)
        c = self.classifier(x)
        box = self.regressor(x)
        return c, box

    def unfreeze(self, choice):
        for p in self.net.parameters():
            p.requires_grad = choice

if __name__ == '__main__':
    x = torch.rand((8*13, 2))
    x = F.log_softmax(x, dim=1)
    y = torch.LongTensor(numpy.random.randint(0, 2, (8*13,)))
    print(F.nll_loss(x, y))
    # model = HeadDetector()
    # print(type(chain(model.classifier.parameters(), model.regressor.parameters())))
    # print(model(x)[0].data)
    # print(timeit.timeit('test()', setup='from __main__ import test', number=3))