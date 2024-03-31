import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_in_gelu(nn.Module):
    def __init__(self, in_channels, out_channels, norm, act, kernel, stride, pad, group):
        super(conv_in_gelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel,stride=stride,padding=pad,groups=group)
        self.ins = nn.InstanceNorm2d(out_channels)
        self.gelu = nn.GELU()
        #self.ins = nn.BatchNorm2d(out_channels)
        #self.gelu = nn.ReLU()
        self.norm = norm
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        if self.norm == True:
            x = self.ins(x)
        if self.act == True:
            x = self.gelu(x)
        return x


class sas(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(sas, self).__init__()
        self.conv1 = conv_in_gelu(in_channels,out_channels,True,True,1,1,0,1)
        self.conv2 = conv_in_gelu(in_channels,in_channels,False,False,3,1,1,in_channels)
        self.conv3 = conv_in_gelu(in_channels,out_channels,True,True,1,1,0,1)
        self.conv4 = conv_in_gelu(2*out_channels,out_channels//2,False,False,1,1,0,1)
        self.conv5 = conv_in_gelu(out_channels//2,out_channels,True,True,1,1,0,1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.conv3(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channel=200, out_channel=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=1,stride=1,padding=0)
        self.conv2 = sas(in_channel,128)
        self.conv3 = sas(128,64)
        self.conv4 = sas(64,32)
        self.conv5 = nn.Conv2d(in_channels=32,out_channels=out_channel,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        return x3,x5


class Decoder(nn.Module):
    def __init__(self, in_channel=32, out_channel=200):
        super().__init__()
        self.conv1 = conv_in_gelu(in_channel,64,False,False,1,1,0,1)
        self.conv2 = conv_in_gelu(64,64,True,True,1,1,0,1)

        self.conv3 = conv_in_gelu(64,128,False,False,1,1,0,1)
        self.conv4 = conv_in_gelu(128,128,True,True,1,1,0,1)
        self.conv5 = conv_in_gelu(128,out_channel,False,False,1,1,0,1)
        self.conv6 = conv_in_gelu(out_channel,out_channel,True,True,1,1,0,1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        return x2,x6


class Auto(nn.Module):
    def __init__(self,in_channel=200,mid_channel=32):
        super(Auto,self).__init__()
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        
        self.encoder = Encoder(in_channel=in_channel,out_channel=mid_channel)
        self.decoder = Decoder(in_channel=mid_channel,out_channel=in_channel)

    def forward(self, x):
        enc64,enc32 = self.encoder(x)
        dec64,dec200 = self.decoder(enc32)

        return enc64,enc32,dec64,dec200
