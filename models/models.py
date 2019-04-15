import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#################
# Autoencoder
#################
class VGG16_AE(nn.Module):
    def __init__(self, in_channels):
        super(VGG16_AE, self).__init__()
        #############
        # Encoder
        #############
        self.eblock1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pooling1 = nn.MaxPool2d(2,2)
        self.eblock2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pooling2 = nn.MaxPool2d(2, 2)
        self.eblock3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.pooling3 = nn.MaxPool2d(2, 2)
        self.eblock4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.pooling4 = nn.MaxPool2d(2, 2)
        self.eblock5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        ###############
        # Decoder
        ###############
        self.dblock1=nn.Sequential(
            nn.Conv2d(512,512,3,1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.dblock2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.dblock3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.dblock4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.dblock5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 3, 1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.eblock1(x)
        out = self.pooling1(out)
        out = self.eblock2(out)
        out = self.pooling2(out)
        out = self.eblock3(out)
        out = self.pooling3(out)
        out = self.eblock4(out)
        out = self.pooling4(out)
        out = self.eblock5(out)

        out = self.dblock1(out)
        out = nn.functional.interpolate(out, scale_factor=2)
        out = self.dblock2(out)
        out = nn.functional.interpolate(out, scale_factor=2)
        out = self.dblock3(out)
        out = nn.functional.interpolate(out, scale_factor=2)
        out = self.dblock4(out)
        out = nn.functional.interpolate(out, scale_factor=2)
        out = self.dblock5(out)
        return out

    def encode(self, x):
        out = self.eblock1(x)
        out = self.pooling1(out)
        out = self.eblock2(out)
        out = self.pooling2(out)
        out = self.eblock3(out)
        out = self.pooling3(out)
        out = self.eblock4(out)
        out = self.pooling4(out)
        out = self.eblock5(out)
        print(out.shape)
        return out

    def decode(self, x):
        out = self.dblock1(x)
        out = nn.functional.interpolate(out, scale_factor=2)
        out = self.dblock2(out)
        out = nn.functional.interpolate(out, scale_factor=2)
        out = self.dblock3(out)
        out = nn.functional.interpolate(out, scale_factor=2)
        out = self.dblock4(out)
        out = nn.functional.interpolate(out, scale_factor=2)
        out = self.dblock5(out)
        return out

class AEVGG19(nn.Module):
    def __init__(self):
        super(AEVGG19, self).__init__()
        ######################
        # Encoder
        ######################
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace=True)

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace=True)

        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace=True)

        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu6 = nn.ReLU(inplace=True)

        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu7 = nn.ReLU(inplace=True)

        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu8 = nn.ReLU(inplace=True)

        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu9 = nn.ReLU(inplace=True)

        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu10 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu11 = nn.ReLU(inplace=True)

        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu12 = nn.ReLU(inplace=True)

        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu13 = nn.ReLU(inplace=True)

        self.maxPool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu14 = nn.ReLU(inplace=True)
        #################
        # Decoder
        #################
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu16 = nn.ReLU(inplace=True)

        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu17 = nn.ReLU(inplace=True)

        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu18 = nn.ReLU(inplace=True)

        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu19 = nn.ReLU(inplace=True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad20 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu20 = nn.ReLU(inplace=True)

        self.reflecPad21 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu22 = nn.ReLU(inplace=True)

        self.reflecPad23 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad24 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu24 = nn.ReLU(inplace=True)

        self.reflecPad25 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv25 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu26 = nn.ReLU(inplace=True)

        self.reflecPad27 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv27 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out, pool_idx = self.maxPool(out)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        out = self.relu5(out)
        out, pool_idx2 = self.maxPool2(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out, pool_idx3 = self.maxPool3(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)
        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out, pool_idx4 = self.maxPool4(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        #### Decoder ####
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        out = self.relu19(out)
        out = self.unpool2(out)
        out = self.reflecPad20(out)
        out = self.conv20(out)
        out = self.relu20(out)
        out = self.reflecPad21(out)
        out = self.conv21(out)
        out = self.relu21(out)
        out = self.reflecPad22(out)
        out = self.conv22(out)
        out = self.relu22(out)
        out = self.reflecPad23(out)
        out = self.conv23(out)
        out = self.relu23(out)
        out = self.unpool3(out)
        out = self.reflecPad24(out)
        out = self.conv24(out)
        out = self.relu24(out)
        out = self.reflecPad25(out)
        out = self.conv25(out)
        out = self.relu25(out)
        out = self.unpool4(out)
        out = self.reflecPad26(out)
        out = self.conv26(out)
        out = self.relu26(out)
        out = self.reflecPad27(out)
        out = self.conv27(out)
        return out

###################
# Classifiers
###################
class Net(nn.Module):
    def __init__(self, in_channels):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3 ,1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3 ,1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3 ,1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], 3*3*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def encode1(self, x):
        x = self.conv1(x)
        return x

    def encode2(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def encode3(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def decode1(self, x):
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def decode2(self, x):
        x = self.conv3(x)
        x = x.view(x.shape[0], 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def decode3(self, x):
        x = x.view(x.shape[0], 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class VGG19(nn.Module):
    def __init__(self, in_channels):
        super(VGG19, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2,2)

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
        )
        self.pool5 = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        out = self.block1(x)
        out = self.pool1(out)
        out = self.block2(out)
        out = self.pool2(out)
        out = self.block3(out)
        out = self.pool3(out)
        out = self.block4(out)
        out = self.pool4(out)
        out = self.block5(out)
        out = self.pool5(out)

        out = out.view(out.shape[0], -1)
        out = self.classifier(out)
        return out

    def encode3(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        return x

    def encode4(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.pool4(x)
        return x

    def encode5(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.pool4(x)
        x = self.block5(x)
        return x

    def decode3(self, x):
        x = self.block4(x)
        x = self.pool4(x)
        x = self.block5(x)
        x = self.pool5(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def decode4(self, x):
        x = self.block5(x)
        x = self.pool5(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def decode5(self, x):
        x = self.pool5(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x



class VGG16(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(VGG16, self).__init__()
        self.pad0 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(2, 2)

        ##########################
        self.pad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.pad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool2d(2, 2)

        ###########################
        self.pad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.pad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool3 = nn.MaxPool2d(2, 2)

        ###########################
        self.pad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.pad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool4 = nn.MaxPool2d(2, 2)

        ###########################
        self.pad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.pad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.pad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool5 = nn.MaxPool2d(2, 2)

        ############################
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.pad0(x)
        out = self.conv1(out)
        out = self.pad1(out)
        out = self.conv2(out)
        out = self.maxpool1(out)

        out = self.pad2(out)
        out = self.conv3(out)
        out = self.pad3(out)
        out = self.conv4(out)
        out = self.maxpool2(out)

        out = self.pad4(out)
        out = self.conv5(out)
        out = self.pad5(out)
        out = self.conv6(out)
        out = self.pad6(out)
        out = self.conv7(out)
        out = self.maxpool3(out)

        out = self.pad7(out)
        out = self.conv8(out)
        out = self.pad8(out)
        out = self.conv9(out)
        out = self.pad9(out)
        out = self.conv10(out)
        out = self.maxpool4(out)

        out = self.pad10(out)
        out = self.conv11(out)
        out = self.pad11(out)
        out = self.conv12(out)
        out = self.pad12(out)
        out = self.conv13(out)
        out = self.maxpool5(out)

        out = out.view(x.shape[0], -1)
        out = self.classifier(out)
        return out

    def encode1(self, x):
        out = self.pad0(x)
        out = self.conv1(out)
        out = self.pad1(out)
        out = self.conv2(out)
        out = self.maxpool1(out)

        out = self.pad2(out)
        out = self.conv3(out)
        out = self.pad3(out)
        out = self.conv4(out)
        out = self.maxpool2(out)

        out = self.pad4(out)
        out = self.conv5(out)
        out = self.pad5(out)
        out = self.conv6(out)
        out = self.pad6(out)
        out = self.conv7(out)
        out = self.maxpool3(out)
        return out

    def encode2(self, x):
        out = self.pad0(x)
        out = self.conv1(out)
        out = self.pad1(out)
        out = self.conv2(out)
        out = self.maxpool1(out)

        out = self.pad2(out)
        out = self.conv3(out)
        out = self.pad3(out)
        out = self.conv4(out)
        out = self.maxpool2(out)

        out = self.pad4(out)
        out = self.conv5(out)
        out = self.pad5(out)
        out = self.conv6(out)
        out = self.pad6(out)
        out = self.conv7(out)
        out = self.maxpool3(out)

        out = self.pad7(out)
        out = self.conv8(out)
        out = self.pad8(out)
        out = self.conv9(out)
        out = self.pad9(out)
        out = self.conv10(out)
        out = self.maxpool4(out)

        return out

    def encode3(self, x):
        out = self.pad0(x)
        out = self.conv1(out)
        out = self.pad1(out)
        out = self.conv2(out)
        out = self.maxpool1(out)

        out = self.pad2(out)
        out = self.conv3(out)
        out = self.pad3(out)
        out = self.conv4(out)
        out = self.maxpool2(out)

        out = self.pad4(out)
        out = self.conv5(out)
        out = self.pad5(out)
        out = self.conv6(out)
        out = self.pad6(out)
        out = self.conv7(out)
        out = self.maxpool3(out)

        out = self.pad7(out)
        out = self.conv8(out)
        out = self.pad8(out)
        out = self.conv9(out)
        out = self.pad9(out)
        out = self.conv10(out)
        out = self.maxpool4(out)

        out = self.pad10(out)
        out = self.conv11(out)
        out = self.pad11(out)
        out = self.conv12(out)
        out = self.pad12(out)
        out = self.conv13(out)
        return out

    def decode1(self, x):
        out = self.pad7(x)
        out = self.conv8(out)
        out = self.pad8(out)
        out = self.conv9(out)
        out = self.pad9(out)
        out = self.conv10(out)
        out = self.maxpool4(out)

        out = self.pad10(out)
        out = self.conv11(out)
        out = self.pad11(out)
        out = self.conv12(out)
        out = self.pad12(out)
        out = self.conv13(out)
        out = self.maxpool5(out)

        out = out.view(x.shape[0], -1)
        out = self.classifier(out)
        return out

    def decode2(self, x):
        out = self.pad10(x)
        out = self.conv11(out)
        out = self.pad11(out)
        out = self.conv12(out)
        out = self.pad12(out)
        out = self.conv13(out)
        out = self.maxpool5(out)

        out = out.view(x.shape[0], -1)
        out = self.classifier(out)
        return out

    def decode3(self, x):
        out = self.maxpool5(x)
        out = out.view(x.shape[0], -1)
        out = self.classifier(out)
        return out

    def forward1(self, x):
        out = self.pad0(x)
        out = self.conv1(out)
        out = self.pad1(out)
        out = self.conv2(out)
        out = self.maxpool1(out)
        return out

    def forward2(self, x):
        out = self.pad2(x)
        out = self.conv3(out)
        out = self.pad3(out)
        out = self.conv4(out)
        out = self.maxpool2(out)
        return out

    def forward3(self, x):
        out = self.pad4(x)
        out = self.conv5(out)
        out = self.pad5(out)
        out = self.conv6(out)
        out = self.pad6(out)
        out = self.conv7(out)
        out = self.maxpool3(out)
        return out

    def forward4(self, x):
        out = self.pad7(x)
        out = self.conv8(out)
        out = self.pad8(out)
        out = self.conv9(out)
        out = self.pad9(out)
        out = self.conv10(out)
        out = self.maxpool4(out)
        return out

    def forward5(self, x):
        out = self.pad10(x)
        out = self.conv11(out)
        out = self.pad11(out)
        out = self.conv12(out)
        out = self.pad12(out)
        out = self.conv13(out)
        return out

    def classify(self, x):
        out = self.maxpool5(x)
        out = out.view(x.shape[0], -1)
        out = self.classifier(out)
        return out

####################
# Helper Functions
####################
def save_model(model ,epoch, optim , filename):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
    }, filename)

def load_model(model, root):
    checkpoint = torch.load(root)
    #epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint)
    #optimizer.load_state_dict(checkpoint['optimizer'])
    return model#, optimizer, epoch