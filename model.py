import torch.nn as nn

class ZFNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ZFNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            # layer1 
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(5),
            # layer2
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(5),
            # layer3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # layer4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # layer5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classification_layer = nn.Sequential(
            # layer6
            nn.Linear(9216, 4096),
            nn.Dropout(),
            # layer7
            nn.Linear(4096, num_classes),
            nn.Dropout()
        )

    #     self.init_weight()


    # # define weight initialization function
    # def init_weight(self):
    #     for layer in self.feature_extractor:
    #         if isinstance(layer, nn.Conv2d):
    #             nn.init.normal_(layer.weight, mean=0, std=0.01)
    #             nn.init.constant_(layer.bias, 0)
    #     # in paper, initialize bias to 1 for conv2, 4, 5 layer
    #     nn.init.constant_(self.feature_extractor[4].bias, 1)
    #     nn.init.constant_(self.feature_extractor[10].bias, 1)
    #    nn.init.constant_(self.feature_extractor[12].bias, 1)
    
    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(-1, 9216)
        out = self.classification_layer(out)
        return out

# model = ZFNet().cuda()
# from torchsummary import summary
# summary(model, input_size = (3, 224, 224))