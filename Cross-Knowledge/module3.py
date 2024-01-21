import torch.nn as nn


class Module3(nn.Module):
    def __init__(self):
        super(Module3, self).__init__()
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=3, stride=2)
        self.group1 = nn.GroupNorm(32, 1024)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(1024, 512, kernel_size=3, stride=2)
        self.group2 = nn.GroupNorm(32, 512)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(512, 256, kernel_size=3, stride=2)
        self.group3 = nn.GroupNorm(32, 256)
        self.relu4 = nn.ReLU(inplace=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=5)
        self.fc1 = nn.Linear(256, 2)
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
        x = self.conv2(x)
        x = self.group1(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.group2(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.group3(x)
        x = self.relu4(x)
        print(x.size())
        x = self.maxpool2(x)
        print(x.size())
        x = x.view(-1, 256)
        x = self.fc1(x)
        return x


def generate_model3():

    model3 = Module3()
    return model3


generate_model3()
