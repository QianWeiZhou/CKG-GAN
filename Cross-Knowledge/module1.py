# @Time    : 2019/12/30 下午3:53
# @Author  : Yibo Liu
# @Email   : 17767094198@163.com
# @File    : module1.py


import torch.nn as nn


class Module1(nn.Module):
    def __init__(self):
        super(Module1, self).__init__()
        self.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2),
            padding=(3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                # print(m)
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # m.weight = nn.init.zeros_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        return x


def generate_model1():

    model1 = Module1()
    return model1


generate_model1()

