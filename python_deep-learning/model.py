# coding=UTF-8
import torch
import pytorch_ssim


# 需要大改出分割网络(3D-CNN、3D-Unet)和预测网络(3D-CNN或用lstm)
# 分割网出掩码，预测网出预测图，分别独立训练，然后掩码乘以预测图。
# 不采用FNN等限制输入输出尺寸的神经层（参考FCN等），因此也就不需要规定输入输出的尺寸遵从某个预设大小，
# 但是受网络结构限制，边长以及时间序列数都需要为4的倍数，否则输出可能不是原边长或输出时间序列缺失一部分
# 理论上具有兼并可能，应该可以有两个出口，但是这样的话，如何回传合并两种label是个问题，可能是高难度操作。放在第三阶段。
# 如果加入时空模块，可以考虑命名为2.5D-CNN-UNet?

# 3D卷积层：大核卷->卷
class Cnn3D_layer(torch.nn.Module):
    def __init__(self, n_input, n_out):
        super(Cnn3D_layer, self).__init__()
        self.network1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=n_input, out_channels=n_out, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                            stride=(1, 1, 1), bias=False),
            torch.nn.BatchNorm3d(n_out), torch.nn.LeakyReLU(inplace=True))
        self.network2 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=n_out, out_channels=n_out, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                            stride=(1, 1, 1), bias=True), torch.nn.BatchNorm3d(n_out),
            torch.nn.Conv3d(in_channels=n_out, out_channels=n_out, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                            stride=(1, 1, 1), bias=False), torch.nn.BatchNorm3d(n_out))
        self.output = torch.nn.Sequential(torch.nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x1 = self.network1(x)
        x2 = self.network2(x1) + x1
        x3 = self.output(x2)
        return x3

class Cnn3D_layer2(torch.nn.Module):
    def __init__(self, n_input, n_out):
        super(Cnn3D_layer2, self).__init__()
        self.network1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=n_input, out_channels=n_out, kernel_size=(1, 1, 5), padding=(0, 0, 2),
                            stride=(1, 1, 1), bias=False),
            torch.nn.BatchNorm3d(n_out), torch.nn.LeakyReLU(inplace=True))
        self.network2 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=n_out, out_channels=n_out, kernel_size=(1, 1, 5), padding=(0, 0, 2),
                            stride=(1, 1, 1), bias=False), torch.nn.BatchNorm3d(n_out),
            torch.nn.Conv3d(in_channels=n_out, out_channels=n_out, kernel_size=(1, 1, 5), padding=(0, 0, 2),
                            stride=(1, 1, 1), bias=False), torch.nn.BatchNorm3d(n_out), torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv3d(in_channels=n_out, out_channels=n_out, kernel_size=(1, 1, 5), padding=(0, 0, 2),
                            stride=(1, 1, 1), bias=False), torch.nn.BatchNorm3d(n_out),
            torch.nn.Conv3d(in_channels=n_out, out_channels=n_out, kernel_size=(1, 1, 5), padding=(0, 0, 2),
                            stride=(1, 1, 1), bias=False), torch.nn.BatchNorm3d(n_out))
        self.output = torch.nn.Sequential(torch.nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x1 = self.network1(x)
        x2 = self.network2(x1) + x1
        x3 = self.output(x2)
        return x3

class Cnn3D_spatial_layer(torch.nn.Module):
    def __init__(self, n_input, n_out):
        super(Cnn3D_spatial_layer, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=n_input, out_channels=n_out, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                            stride=(1, 1, 1), bias=False), torch.nn.BatchNorm3d(n_out),
            torch.nn.Conv3d(in_channels=n_out, out_channels=n_out, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                            stride=(1, 1, 1), bias=False), torch.nn.BatchNorm3d(n_out),
            torch.nn.Conv3d(in_channels=n_out, out_channels=n_out, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                            stride=(1, 1, 1), bias=True), torch.nn.Sigmoid())

    def forward(self, x):
        x1 = self.network(x) * x
        return x1


class up_sample_conv(torch.nn.Module):
    def __init__(self, n_channel):
        super(up_sample_conv, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv3d(in_channels=n_channel, out_channels=n_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                            stride=(1, 1, 1), bias=True))

    def forward(self, x):
        return self.network(x)


class down_sample_conv(torch.nn.Module):
    def __init__(self, n_channel):
        super(down_sample_conv, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=n_channel, out_channels=n_channel, kernel_size=(2, 2, 2), padding=(0, 0, 0),
                            stride=(2, 2, 2), bias=True))

    def forward(self, x):
        return self.network(x)


class Cnn2D_layer(torch.nn.Module):
    def __init__(self, n_input, n_out):
        super(Cnn2D_layer, self).__init__()
        self.network1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=n_input, out_channels=n_out, kernel_size=(3, 3), padding=(1, 1),
                            stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(n_out), torch.nn.LeakyReLU(inplace=True))
        self.network2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=(3, 3), padding=(1, 1),
                            stride=(1, 1), bias=True), torch.nn.BatchNorm2d(n_out),
            torch.nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=(3, 3), padding=(1, 1),
                            stride=(1, 1), bias=False), torch.nn.BatchNorm2d(n_out))
        self.output = torch.nn.Sequential(torch.nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x1 = self.network1(x)
        x2 = self.network2(x1) + x1
        x3 = self.output(x2)
        return x3


# 预测滤波后的值的
# 总体为3D CNN U-net结构，但是中心部分（U的谷底）加一个时空模块，残差连接的方式是把旧张量作为额外通道
# 时空模块：参照SVD的物理意义，把2个空间维reshape到一起，则3维降2维，做2D CNN，然后再reshape回3维。该模块是否有用需要做消融测试来判断
# 可能要考虑不做池化来防止细节丢失。。。如果改了平均池化和sum loss还是失败的话，就不做池化了，这样的话通道数一直维持到8就可以了
# 这个结构做不了频率关注
# 这个是仿体模型的结构，千万不要修改！！！！
class UNet3D_regression(torch.nn.Module):  # 简易u-net
    def __init__(self):
        super(UNet3D_regression, self).__init__()
        self.add_module("conv1", Cnn3D_layer(2, 8))  # n*n
        self.add_module("downS1", torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))  # down_sample_conv(8)
        self.add_module("conv2", Cnn3D_layer(8, 16))  # n*n
        self.add_module("downS2", torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))
        self.add_module("conv3", Cnn3D_layer(16, 32))  # n*n
        self.add_module("downS3", torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))
        self.add_module("middle_conv", Cnn3D_layer(32, 32))  # n*n
        self.add_module("upS4", torch.nn.Upsample(scale_factor=2, mode="nearest"))  # up_sample_conv(16)
        self.add_module("conv4", Cnn3D_layer(64, 16))  # U-net: 32+32
        self.add_module("upS5",  torch.nn.Upsample(scale_factor=2, mode="nearest"))  #torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.add_module("conv5", Cnn3D_layer(32, 8))  # U-net: 16+16
        self.add_module("upS6", torch.nn.Upsample(scale_factor=2, mode="nearest"))
        self.add_module("conv6", Cnn3D_layer(16, 4))  # U-net: 8+8
        self.add_module("output", torch.nn.Conv3d(in_channels=4, out_channels=2, kernel_size=(1, 1, 1), padding=(0, 0, 0),
                                                        stride=(1, 1, 1), bias=True))

    def forward(self, x):
        assert x.shape[2] % 8 == 0 and x.shape[3] % 8 == 0 and x.shape[4] % 8 == 0, "w, h, should be the fold of 8. t should be the fold of 4"
        # x = torch.as_tensor(x, dtype=torch.float32)  # 以 batch*2*64*176*200的样本输入为例（不一定要遵循该尺寸）
        x1 = self.conv1(x)  # 8*64*176*200
        x1d = self.downS1(x1)
        x2 = self.conv2(x1d)  # 16*32*88*100
        x3 = self.conv3(self.downS2(x2))  # 32*16*44*50

        x_middle = self.middle_conv(self.downS3(x3))  # 32*8*22*25

        x4 = self.conv4(torch.cat([self.upS4(x_middle), x3], dim=1))  # 16*16*44*50
        x5 = self.conv5(torch.cat([self.upS5(x4), x2], dim=1))  # 8*32*88*100
        x6 = self.output(self.conv6(torch.cat([self.upS6(x5), x1], dim=1)))  # 4*64*176*200
        # 2*64*176*200; 主要起到调整幅度的作用，所以全是1X1X1的卷积
        return x6

'''
class UNet3D_regression(torch.nn.Module):  # 简易u-net
    def __init__(self):
        super(UNet3D_regression, self).__init__()
        self.add_module("conv1", Cnn3D_layer2(2, 8))  # n*n
        self.add_module("conv1s", Cnn3D_spatial_layer(8, 8))  # n*n
        self.add_module("downS1", torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), return_indices=True))  # down_sample_conv(8)
        self.add_module("conv2", Cnn3D_layer2(8, 16))  # n*n
        self.add_module("conv2s", Cnn3D_spatial_layer(16, 16))  # n*n
        self.add_module("downS2", torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), return_indices=True))
        self.add_module("conv3", Cnn3D_layer2(16, 32))  # n*n
        self.add_module("conv3s", Cnn3D_spatial_layer(32, 32))  # n*n
        self.add_module("downS3", torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), return_indices=True))
        self.add_module("middle_conv", Cnn3D_layer2(32, 32))  # n*n
        self.add_module("upS4", torch.nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))  # up_sample_conv(16)
        self.add_module("conv4", Cnn3D_layer2(64, 16))  # U-net: 32+32
        self.add_module("upS5",  torch.nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))  #torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.add_module("conv5", Cnn3D_layer2(32, 8))  # U-net: 16+16
        self.add_module("upS6", torch.nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))
        self.add_module("conv6", Cnn3D_layer2(16, 4))  # U-net: 8+8
        self.add_module("output", torch.nn.Conv3d(in_channels=4, out_channels=2, kernel_size=(1, 1, 1), padding=(0, 0, 0),
                                                        stride=(1, 1, 1), bias=True))

    def forward(self, x):
        assert x.shape[2] % 8 == 0 and x.shape[3] % 8 == 0 and x.shape[4] % 8 == 0, "w, h, should be the fold of 8. t should be the fold of 4"
        # x = torch.as_tensor(x, dtype=torch.float32)  # 以 batch*2*64*176*200的样本输入为例（不一定要遵循该尺寸）
        x1 = self.conv1s(self.conv1(x))  # 8*64*176*200
        x1d, ind1 = self.downS1(x1)
        x2 = self.conv2s(self.conv2(x1d))  # 16*32*88*100
        x2d, ind2 = self.downS2(x2)
        x3 = self.conv3s(self.conv3(x2d))  # 32*16*44*50
        x3d, ind3 = self.downS3(x3)

        x_middle = self.middle_conv(x3d)  # 32*8*22*25

        x4 = self.conv4(torch.cat([self.upS4(x_middle, ind3), x3], dim=1))  # 16*16*44*50
        x5 = self.conv5(torch.cat([self.upS5(x4, ind2), x2], dim=1))  # 8*32*88*100
        x6 = self.output(self.conv6(torch.cat([self.upS6(x5, ind1), x1], dim=1)))  # 4*64*176*200
        # 2*64*176*200; 主要起到调整幅度的作用，所以全是1X1X1的卷积
        return x6
'''


class UNet2D_regression(torch.nn.Module):  # 简易u-net
    def __init__(self):
        super(UNet2D_regression, self).__init__()
        self.add_module("conv1", Cnn2D_layer(2, 8))  # n*n
        self.add_module("downS1", torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)))  # down_sample_conv(8)
        self.add_module("conv2", Cnn2D_layer(8, 16))  # n*n
        self.add_module("downS2", torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)))
        self.add_module("conv3", Cnn2D_layer(16, 32))  # n*n
        self.add_module("downS3", torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)))
        self.add_module("middle_conv", Cnn2D_layer(32, 32))  # n*n
        self.add_module("upS4", torch.nn.Upsample(scale_factor=2, mode="nearest"))  # up_sample_conv(16)
        self.add_module("conv4", Cnn2D_layer(64, 16))  # U-net: 32+32
        self.add_module("upS5",  torch.nn.Upsample(scale_factor=2, mode="nearest"))  #torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.add_module("conv5", Cnn2D_layer(32, 8))  # U-net: 16+16
        self.add_module("upS6", torch.nn.Upsample(scale_factor=2, mode="nearest"))
        self.add_module("conv6", Cnn2D_layer(16, 4))  # U-net: 8+8
        self.add_module("output", torch.nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(1, 1), padding=(0, 0),
                                                        stride=(1, 1), bias=True))

    def forward(self, x):
        assert x.shape[2] % 8 == 0 and x.shape[3] % 8 == 0, "w, h, should be the fold of 8. t should be the fold of 4"
        # x = torch.as_tensor(x, dtype=torch.float32)  # 以 batch*2*64*176*200的样本输入为例（不一定要遵循该尺寸）
        x1 = self.conv1(x)  # 8*64*176*200
        x1d = self.downS1(x1)
        x2 = self.conv2(x1d)  # 16*32*88*100
        x3 = self.conv3(self.downS2(x2))  # 32*16*44*50

        x_middle = self.middle_conv(self.downS3(x3))  # 32*8*22*25

        x4 = self.conv4(torch.cat([self.upS4(x_middle), x3], dim=1))  # 16*16*44*50
        x5 = self.conv5(torch.cat([self.upS5(x4), x2], dim=1))  # 8*32*88*100
        x6 = self.output(self.conv6(torch.cat([self.upS6(x5), x1], dim=1)))  # 4*64*176*200
        # 2*64*176*200; 主要起到调整幅度的作用，所以全是1X1X1的卷积
        return x6


# 分割任务，选做，网络结构和regression一致，但是评价函数换成分类任务的crossEntropy，且需要在模型外自行flattern之后送入评价，
# 再flattern传回来，因此需要包装一下pytorch的crossEntropy
class UNet3D_segmentation(torch.nn.Module):  # 简易u-net
    def __init__(self):
        super(UNet3D_segmentation, self).__init__()
        self.add_module("conv1", Cnn3D_layer(2, 8))  # n*n
        self.add_module("downS1", torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),
                                                     padding=(0, 0, 0)))  # down_sample_conv(8)
        self.add_module("conv2", Cnn3D_layer(8, 16))  # n*n
        self.add_module("downS2", torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))
        self.add_module("conv3", Cnn3D_layer(16, 32))  # n*n
        self.add_module("downS3", torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))
        self.add_module("middle_conv", Cnn3D_layer(32, 32))  # n*n
        self.add_module("upS4", torch.nn.Upsample(scale_factor=2, mode="nearest"))  # up_sample_conv(16)
        self.add_module("conv4", Cnn3D_layer(64, 16))  # U-net: 32+32
        self.add_module("upS5", torch.nn.Upsample(scale_factor=2,
                                                  mode="nearest"))  # torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.add_module("conv5", Cnn3D_layer(32, 8))  # U-net: 16+16
        self.add_module("upS6", torch.nn.Upsample(scale_factor=2, mode="nearest"))
        self.add_module("conv6", Cnn3D_layer(16, 1))  # U-net: 8+8
        self.add_module("output", torch.nn.Sigmoid())

    def forward(self, x):
        assert x.shape[2] % 8 == 0 and x.shape[3] % 8 == 0 and x.shape[
            4] % 8 == 0, "w, h, should be the fold of 8. t should be the fold of 4"
        # x = torch.as_tensor(x, dtype=torch.float32)  # 以 batch*2*64*176*200的样本输入为例（不一定要遵循该尺寸）
        x1 = self.conv1(x)  # 8*64*176*200
        x1d = self.downS1(x1)
        x2 = self.conv2(x1d)  # 16*32*88*100
        x3 = self.conv3(self.downS2(x2))  # 32*16*44*50

        x_middle = self.middle_conv(self.downS3(x3))  # 32*8*22*25

        x4 = self.conv4(torch.cat([self.upS4(x_middle), x3], dim=1))  # 16*16*44*50
        x5 = self.conv5(torch.cat([self.upS5(x4), x2], dim=1))  # 8*32*88*100
        x6 = self.output(self.conv6(torch.cat([self.upS6(x5), x1], dim=1)))  # 4*64*176*200
        # 2*64*176*200; 主要起到调整幅度的作用，所以全是1X1X1的卷积
        return x6


class UNet2D_segmentation(torch.nn.Module):  # 简易u-net
    def __init__(self):
        super(UNet2D_segmentation, self).__init__()
        self.add_module("conv1", Cnn2D_layer(2, 8))  # n*n
        self.add_module("downS1", torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0),
                                                     return_indices=True))
        self.add_module("conv2", Cnn2D_layer(8, 16))  # n*n
        self.add_module("downS2", torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0),
                                                     return_indices=True))
        self.add_module("middle_conv", Cnn2D_layer(16, 16))  # n*n
        # self.add_module("middle_conv", Cnn2D_reshape_layer(16))  # 2D CNN, 其物理意义为时空模块，将空间与时间分开，文章将会消融该核心，探讨时空特性
        self.add_module("upS3", torch.nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)))
        self.add_module("conv3", Cnn2D_layer(32, 8))  # U-net: 32+32
        self.add_module("upS4", torch.nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)))
        self.add_module("conv4", Cnn2D_layer(16, 2))  # U-net: 16+16
        self.add_module("output", torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(3, 3), padding=(1, 1),
                                                  stride=(1, 1), bias=True))
        self.add_module("classifier", torch.nn.Sigmoid())

    def forward(self, x_ori):
        assert (x_ori.shape[2] * x_ori.shape[3]) % 4 == 0 and x_ori.shape[4] % 4 == 0, "w*h, t, should be the fold of 4"
        x_ori = torch.as_tensor(x_ori, dtype=torch.float32)  # 以 batch*2*128*380*1000的样本输入为例（不一定要遵循该尺寸）
        x = torch.reshape(x_ori, (x_ori.shape[0], x_ori.shape[1],
                                  x_ori.shape[2] * x_ori.shape[3], x_ori.shape[4]))
        x1 = self.conv1(x)  # 8*(128*380)*1000
        x1d, ind1 = self.downS1(x1)  # 8*(64*380)*500
        x2 = self.conv2(x1d)  # 16*64*190*500
        x2d, ind2 = self.downS2(x2)  # 16*(32*380)*250
        x_middle = self.middle_conv(x2d)  # small resi 16*(32*380)*250
        x3u = torch.cat([self.upS3(x_middle, ind2), x2], dim=1)  # 16*64*190*500
        x3 = self.conv3(x3u)  # 8*64*190*500
        x4u = torch.cat([self.upS4(x3, ind1), x1], dim=1)  # 8*128*380*1000
        x4 = self.conv4(x4u)  # 2*(128*380)*1000
        x_out = self.output(x4)
        x_final = self.classifier(torch.reshape(x_out, (x_ori.shape[0], x_ori.shape[1],
                                                x_ori.shape[2], x_ori.shape[3], x_ori.shape[4])))
        return x_final


# 根据气象图预测文献定义的2D网络, CNN部分结构仍然采取u-net，但是多了lstm接入末尾（此结构无法使用时空模块，但是lstm会承担时空功能）
class UNet2D_LSTM(torch.nn.Module):  #
    def __init__(self):
        super(UNet2D_LSTM, self).__init__()

    def forward(self, x):
        pass


class model_demo(torch.nn.Module):
    def __init__(self):
        super(model_demo,self).__init__()
        self.add_module("demo1", Cnn3D_layer(1, 12))  # n*n
        self.add_module("demo2", Cnn3D_layer(12, 1))  # n*n

    def forward(self, x):
        # print(x.shape) -> [batch_size, 1, 100, 380, sequence_dim]
        x2 = self.demo1(x)
        # print(x2.shape)
        x3 = self.demo2(x2)
        # print(x3.shape)
        return x3


class hyt_SSIM_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, label):
        # 复现2patch SSIM的计算，输入数据预计为[6，2，64，176，200]的形式，
        # 批次和通道维度可以计算完后单独叠加，因为时间慢动连贯，时间维度可以当作3维图处理
        predict2 = torch.permute(predict, dims=[0, 4, 1, 2, 3])
        label2 = torch.permute(label, dims=[0, 4, 1, 2, 3])
        # [批, 时, 通, 高，长]
        print(predict2.shape)
        # print(predict.shape)

        predict3 = torch.reshape(predict2, shape=[predict2.shape[0] * predict2.shape[1], predict2.shape[2],
                                                  predict2.shape[3], predict2.shape[4]])
        # [批*时, 通，高，长]
        label3 = torch.reshape(label2, shape=[label2.shape[0] * label2.shape[1], label2.shape[2],
                                              label2.shape[3], label2.shape[4]])

        predict4 = torch.unsqueeze(torch.sqrt(((torch.pow(torch.squeeze(predict3[:, 0, :, :]), 2.0)
                       + torch.pow(torch.squeeze(predict3[:, 1, :, :]), 2.0)))), dim=1)
        label4 = torch.unsqueeze(torch.sqrt(((torch.pow(torch.squeeze(label3[:, 0, :, :]), 2.0)
                        + torch.pow(torch.squeeze(label3[:, 1, :, :]), 2.0)))), dim=1)

        #min_factor = min(torch.min(predict4).item(), torch.min(label4).item())
        #max_factor = max(torch.max(predict4).item(), torch.max(label4).item())
        #print(min_factor)
        #print(max_factor)
        #predict5 = (predict4 - min_factor) / (max_factor - min_factor)#torch.unsqueeze(torch.sqrt(((torch.pow(torch.squeeze(predict3[:, 0, :, :]), 2.0)
        #           #             + torch.pow(torch.squeeze(predict3[:, 1, :, :]), 2.0)))).cpu(), dim=1)#255*(predict3 - torch.min(predict3)) / (torch.max(predict3) - torch.min(predict3))
        #label5 = (label4 - min_factor) / (max_factor - min_factor)# torch.unsqueeze(torch.sqrt(((torch.pow(torch.squeeze(label3[:, 0, :, :]), 2.0)
        #         #                 + torch.pow(torch.squeeze(label3[:, 1, :, :]), 2.0)))).cpu(),dim=1)#255*(label3 - torch.min(label3)) / (torch.max(label3) - torch.min(label3))
#
        # print(predict4.shape)
        output1 = pytorch_ssim.ssim(predict4, label4, window_size=7, size_average=False) # torch.zeros(size=predict3.shape)
        # print(output1.mean())
        return 1-output1.mean()


class hyt_SSIM_Huber_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, label):
        ssim_cri = hyt_SSIM_loss()
        ssim_part = ssim_cri(predict, label)

        huber_cri = torch.nn.modules.loss.SmoothL1Loss(reduction="mean")
        huber_part = huber_cri(predict, label)
        # ss = torch.nn.modules.loss.SmoothL1Loss(reduction="mean")
        defined_loss = 0.1*ssim_part + 0.9*huber_part

        print(ssim_part)
        print(huber_part)
        print(defined_loss)

        return defined_loss



if __name__ == "__main__":
    cri = hyt_SSIM_loss()
    a = torch.rand(size=[6, 2, 64, 176, 200], requires_grad=True)*10
    b = a #torch.rand(size=[6, 2, 64, 176, 200], requires_grad=True)

    y = cri(a,b)
    print(y)
    y.backward()