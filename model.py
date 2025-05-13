import torch
import torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        self.conv1 = self.make_block(3, 8)
        self.conv2 = self.make_block(8, 16)
        self.conv3 = self.make_block(16, 32)
        self.conv4 = self.make_block(32, 64)
        self.conv5 = self.make_block(64, 128)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_class)
        )

    def make_block(self, intput_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(intput_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        self.conv1 = self.make_block(3, 16)
        self.conv2 = self.make_block(16, 32)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_class)

    def make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == "__main__":
    model = SimpleCNN()
    input_data = torch.rand(8, 3, 512, 512)
    output = model(input_data)
    # print(output.shape)
