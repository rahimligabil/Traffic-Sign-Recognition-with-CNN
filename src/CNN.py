import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self,num_classes = 43):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size = 3,padding = 1,stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            

            nn.Conv2d(32,64,kernel_size = 3,padding = 1,stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),
            


            nn.Conv2d(64,128,kernel_size = 3,padding = 1,stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),
            
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(128,num_classes)



    def forward(self,x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
        