import torch  # for using torch.sigmoid
import torch.nn as nn  # for using nn.Module
from torchsummary import summary  # for checking amount of model parameter



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 5, stride=1)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=1)
        self.conv2_bn = nn.BatchNorm2d(16)


        self.fc1 = nn.Linear(61 * 61 * 16, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = Flatten()
        self.drop_out = nn.Dropout(0.2)

    def forward(self, img):
        output = self.conv1(img)
        output = self.maxpool(self.relu(output))
        output = self.conv2(output)
        output = self.conv2_bn(output)
        output = self.maxpool(self.relu(output))

        output = self.flatten(output)
        output = self.drop_out(output)
        output = self.fc1(output)
        output = self.fc1_bn(output)
        output = self.relu(output)
        output = self.drop_out(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.drop_out(output)
        output = self.fc3(output)

        return output

    def summary(self):
        summary(self, (1, 300, 300))


if __name__ == "__main__":
    # Caution! If don't have cuda device, using CNN(). Not CNN().cuda().
    # Also CustomMLP have to upper line.
    graphic_device = 'gpu'
    if graphic_device == 'cpu':
        summary(CNN(), (3, 256, 256), device=graphic_device)
    else:
        summary(CNN().cuda(), (3, 256, 256), device='cuda')
