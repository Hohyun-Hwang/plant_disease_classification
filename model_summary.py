import torch
from model import CNN
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchsummary import summary

if __name__ == '__main__':
    graphic_device = 'cpu'
    if graphic_device == 'cpu':
        summary(CNN(), (3, 256, 256), device=graphic_device)
        summary(EfficientNet.from_pretrained('efficientnet-b7', num_classes=21), (3,256,256),device=graphic_device)
        model = torch.hub.load("pytorch/vision:v0.6.0", 'resnet18', pretrained=True)
        model.fc = nn.Linear(512, 21, bias=True)
        summary(model, (3,256,256), device=graphic_device)
    else:
        summary(CNN().cuda(), (3, 256, 256), device='cuda')
        summary(EfficientNet.from_pretrained('efficientnet-b7', num_classes=21).cuda(), (3,256,256) , device='cuda')
        model = torch.hub.load("pytorch/vision:v0.6.0", 'resnet18', pretrained=True)
        model.fc = nn.Linear(512, 21, bias=True)
        summary(model.to('cuda'),(3,256,256) ,device='cuda')
