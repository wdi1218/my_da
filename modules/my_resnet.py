from torchvision.models.resnet import resnet50

class MyResnet_wofc(object):
    def __init__(self):
        # print('Creating CNN instance.')
        self.resnet = resnet50(pretrained=True)
        # Remove final classifier layer
        del self.resnet.fc

    def resnet_forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        res4f_relu = self.resnet.layer3(x)
        res5e_relu = self.resnet.layer4(res4f_relu)
        res5ee = res5e_relu.permute(0, 2, 3, 1)

        return res5ee






