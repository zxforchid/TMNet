 # coding=utf-8
from inspect import classify_class_attrs
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.conv import Conv2d
#from model.resattention import res_cbam
import torchvision.models as models
#from model.res2fg import res2net
import torch.nn.functional as F
import math
from model import vgg
class SelfAttention_1(nn.Module):
    def __init__(self,in_channels):
        super(SelfAttention_1, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels
        #max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        #bn = nn.BatchNorm2d

        self.g = nn.Conv2d(in_channels,in_channels,1)
        self.theta = nn.Conv2d(in_channels,in_channels,1)
        self.phi = nn.Conv2d(in_channels,in_channels,1)
    def forward(self,x3):

        batch_size = x3.size(0)

        g_x = self.g(x3).view(batch_size, self.inter_channels, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)#[b,wh,c]

        theta_x = self.theta(x3).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x3).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x3.size()[2:])
        out = y + x3
        
        return out



class TIU(nn.Module):
    def __init__(self,in_channels):
        super(TIU,self).__init__()
        self.scf1 = SCF(in_channels)
        self.scf2 = SCF(in_channels)
        self.scf3 = SCF(in_channels)
        self.cf1 = CF(in_channels)
        self.cf2 = CF(in_channels)
        self.cf3 = CF(in_channels)
        self.cf4 = CF(in_channels)

        self.conv2 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)

        self.SA = SpatialAttention()
    def forward(self,x1,x2,x3):
        r1,r2,r3,r4 = self.scf1(x1)
        d1,d2,d3,d4 = self.scf2(x2)
        t1,t2,t3,t4 = self.scf3(x3)
        m1 = self.cf1(r1,d1,t1)
        m2 = self.cf2(m1+r2,d2+m1,t2+m1)
        m3 = self.cf4(m2+r3,d3+m2,t3+m2)
        m4 = self.cf4(m3+r4,d4+m3,t4+m3)
        M = torch.cat((m1,m2,m3,m4),1)
        M_s = self.SA(M)
        out = x1*M_s+x2*M_s+x3*M_s+M
        return out
    

class CF(nn.Module):
    def __init__(self,in_channels):
        super(CF,self).__init__()
        self.CA_r = ChannelAttention_2(in_channels)
        self.SA_r = SpatialAttention()
        self.CA_d = ChannelAttention_2(in_channels)
        self.CA_t = ChannelAttention_2(in_channels)
        self.SA_dt = SpatialAttention()
        self.conv1 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(3*in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,r,d,t):
        sa_r = self.SA_r(self.CA_r(r)*r)
        d_r = d+sa_r*d
        t_r = t+sa_r*t
        dt = self.conv1(torch.cat((self.CA_d(d_r)*d_r,self.CA_t(t_r)*t_r),1))
        sa_dt = self.SA_dt(dt)
        out = r*sa_dt+dt*sa_dt
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)*x

class ChannelAttention_1(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_1, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class ChannelAttention_2(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_2, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 2, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 2, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)#dim=1是因为传入数据是多加了一维unsqueeze(0)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)


#mid level
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()
        self.att1 = ChannelAttention_1(in_channels)
        self.att2 = ChannelAttention_2(in_channels)
        self.conv = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,x):
        a1 = x*self.att1(x)
        a2 = x*self.att2(x)
        out = self.conv(torch.cat((a1,a2),1))
        return out
class SCF(nn.Module):# Pyramid Feature Module
    def __init__(self,in_channels):
        super(SCF,self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=7, dilation=7)
        )

        #self.atten0 = ChannelAttention_2(in_channels)
        #self.atten1 = ChannelAttention_2(in_channels)
        #self.atten2 = ChannelAttention_2(in_channels)

        #self.conv = nn.Conv2d(4*in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,x):
        x0,x1,x2,x3 = torch.split(x,x.size()[1]//4,dim=1)
        y0 = self.branch0(x0)
        y1 = self.branch1(x1)
        y2 = self.branch2(x2)
        y3 = self.branch3(x3)

        #y = self.conv(torch.cat((y0,y1,y2,y3),1))
        return y0,y1,y2,y3


class CR(nn.Module):
    def __init__(self,in_channels):
        super(CR,self).__init__()
        self.ca1 = ChannelAttention_2(in_channels)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention_2(in_channels)
        self.sa2 = SpatialAttention()
        self.ca3 = ChannelAttention_2(in_channels)
        self.sa3 = SpatialAttention()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv11 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv22 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv33 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)

        self.conv4 = nn.Conv2d(3*in_channels,in_channels,kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv_out = nn.Conv2d(in_channels,1,kernel_size=1)
    def forward(self,x1,x2,x3):
        a = self.conv11(self.conv1(x1))
        aa = self.ca1(a)*a+self.sa1(a)*a
        b = self.conv22(self.conv2(x2))
        bb = self.ca2(b)*b+self.sa2(b)*b
        c = self.conv33(self.conv3(x3))
        cc = self.ca3(c)*c+self.sa3(c)*c
        out = self.conv5(self.conv4(torch.cat((aa,bb,cc),1)))
        #out = self.conv_out(out)
        
        return out
    
class CBR(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CBR,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.Ba = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.relu(self.Ba(self.conv(x)))

        return out
class TMNet(nn.Module):#输入三通道
    def __init__(self, in_channels):
        super(TMNet, self).__init__()
        #resnet_RGB = models.resnet18(pretrained=True)
        #resnet_D = models.resnet18(pretrained=True)
        #resnet_T = models.resnet18(pretrained=True)

        self.vgg_RGB = vgg.a_vgg16()
        self.vgg_D = vgg.a_vgg16()
        self.vgg_T = vgg.a_vgg16()
        #resnet2 = models.resnet18(pretrained=True)
        #self.weight=nn.Parameter(torch.FloatTensor(1))
        #
        #reanet = res2net()
        #res2n
        # ************************* Encoder ***************************
        # input conv3*3,64
        
        self.conv_1x1_0_RGB = nn.Conv2d(64,128,kernel_size=1)
        self.conv_1x1_1_RGB = nn.Conv2d(128,128,kernel_size=1)
        self.conv_1x1_2_RGB = nn.Conv2d(256,128,kernel_size=1)
        self.conv_1x1_3_RGB = nn.Conv2d(512,128,kernel_size=1)
        self.conv_1x1_4_RGB = nn.Conv2d(512,128,kernel_size=1)
        
        self.conv_1x1_0_D = nn.Conv2d(64,128,kernel_size=1)
        self.conv_1x1_1_D = nn.Conv2d(128,128,kernel_size=1)
        self.conv_1x1_2_D = nn.Conv2d(256,128,kernel_size=1)
        self.conv_1x1_3_D = nn.Conv2d(512,128,kernel_size=1)
        self.conv_1x1_4_D = nn.Conv2d(512,128,kernel_size=1)

        self.conv_1x1_0_T = nn.Conv2d(64,128,kernel_size=1)
        self.conv_1x1_1_T = nn.Conv2d(128,128,kernel_size=1)
        self.conv_1x1_2_T = nn.Conv2d(256,128,kernel_size=1)
        self.conv_1x1_3_T = nn.Conv2d(512,128,kernel_size=1)
        self.conv_1x1_4_T = nn.Conv2d(512,128,kernel_size=1)

        self.tiu0 = TIU(32)
        self.tiu1 = TIU(32)
        self.tiu2 = TIU(32)
        self.tiu3 = TIU(32)
        self.tiu4 = TIU(32)
        # ************************* Decoder ***************************
        #self.Se_D = SE()
        #self.Se_T = SE()
        self.conv_rd1 = nn.Conv2d(256,128,kernel_size=1)
        self.conv_rd0 = nn.Conv2d(256,128,kernel_size=1)

        self.conv_rt1 = nn.Conv2d(256,128,kernel_size=1)
        self.conv_rt0 = nn.Conv2d(256,128,kernel_size=1)

        self.conv_dt1 = nn.Conv2d(256,128,kernel_size=1)
        self.conv_dt0 = nn.Conv2d(256,128,kernel_size=1)

        self.conv_rdt1 = nn.Conv2d(384,128,kernel_size=1)
        self.conv_rdt0 = nn.Conv2d(384,128,kernel_size=1)

        self.conv4_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv3_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv2_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv1_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv0_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)

        self.conv4_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv3_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv1_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv0_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)

        self.cr = CR(128)
        
        #self.pfm_0_1 = PFM(128)
        #self.pfm_1_1 = PFM(128)
        #self.pfm_0_2 = PFM(128)
        #self.pfm_1_2 = PFM(128)
        #self.pfm_0_3 = PFM(128)
        #self.pfm_1_3 = PFM(128)


        self.conv_3x3_0_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_1_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_2_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_3_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_4_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)

        self.conv_3x3_0_T = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_1_T = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_2_T = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_3_T = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_4_T = nn.Conv2d(256,128,kernel_size=3,padding=1)

        #self.decoder_RGB = Decoder()
        #self.decoder_T = Decoder()

        #self.loc = location(128)
        #self.Fuse = FF()
        #self.fff_4 = FF(128)
        #self.fff_3 = FFF(128)
        #self.fff_2 = FFF(128)
        #self.fff_1 = FFF(128)
        #self.fff_0 = FFF(128)
        self.conv_a2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv_a1 = nn.Conv2d(512,128,kernel_size=3,padding=1)
        self.conv_a0 = nn.Conv2d(640,128,kernel_size=3,padding=1)
        self.se4_attention = SelfAttention_1(128)
        self.se3_attention = SelfAttention_1(128)
        self.at2 = Attention(128)
        self.at1 = Attention(128)
        self.at0 = Attention(128)
        self.conv_2 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.conv_1 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.conv_0 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.conv_e = nn.Conv2d(128,1,kernel_size=1)

        # ************************* Feature Map Upsample ***************************
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.downsample2 = nn.Upsample(scale_factor=0.25, mode='bilinear')
        self.downsample3 = nn.Upsample(scale_factor=0.125, mode='bilinear')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample5 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.conv_out_4 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_out_3 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_out_2 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_out_1 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_out_0 = nn.Conv2d(128,1,kernel_size=1)
        
    def load_pretrained_model(self):
        st=torch.load("./model/vgg16.pth")
        st2={}
        for key in st.keys():
            st2['base.'+key]=st[key]
        self.vgg_RGB.load_state_dict(st2)
        self.vgg_T.load_state_dict(st2)
        self.vgg_D.load_state_dict(st2)
        print('loading pretrained model success!')
        
    def forward(self, x_rgb,x_d,x_t):
        # ************************* Encoder ***************************
        #D>>R
        d = self.vgg_D(x_d)
        
        s0_d = self.conv_1x1_0_D(d[0])
        s1_d = self.conv_1x1_1_D(d[1])
        s2_d = self.conv_1x1_2_D(d[2])
        s3_d = self.conv_1x1_3_D(d[3])
        s4_d = self.conv_1x1_4_D(d[4])


         #T>>R
        t = self.vgg_T(x_t)
        s0_t = self.conv_1x1_0_T(t[0])
        s1_t = self.conv_1x1_1_T(t[1])
        s2_t = self.conv_1x1_2_T(t[2])
        s3_t = self.conv_1x1_3_T(t[3])
        s4_t = self.conv_1x1_4_T(t[4])
        
        

        #R
        r = self.vgg_RGB(x_rgb)

        s0_r = self.conv_1x1_0_RGB(r[0])
        s1_r = self.conv_1x1_1_RGB(r[1])
        s2_r = self.conv_1x1_2_RGB(r[2])
        s3_r = self.conv_1x1_3_RGB(r[3])
        s4_r = self.conv_1x1_4_RGB(r[4])
        ##################################
        
        ###1
        E0 = self.tiu0(s0_r,s0_d,s0_t)
        E1 = self.tiu1(s1_r,s1_d,s1_t)
        E2 = self.tiu2(s2_r,s2_d,s2_t)
        E3 = self.tiu3(s3_r,s3_d,s3_t)
        E4 = self.tiu4(s4_r,s4_d,s4_t)

        e = self.cr(E0,self.upsample1(E1),self.upsample2(E2))


        s4 = self.se4_attention(E4)
        s3 = self.se3_attention(E3)
        s2 = self.at2(self.conv_a2(torch.cat((E2,self.upsample1(s3),self.upsample2(s4)),1)))
        s2 = self.conv_2(s2+s2*self.downsample2(e)+self.downsample2(e))
        s1 = self.at1(self.conv_a1(torch.cat((E1,self.upsample1(s2),self.upsample2(s3),self.upsample3(s4)),1)))
        s1 = self.conv_1(s1+s1*self.downsample(e)+self.downsample(e))
        s0 = self.at0(self.conv_a0(torch.cat((E0,self.upsample1(s1),self.upsample2(s2),self.upsample3(s3),self.upsample4(s4)),1)))
        s0 = self.conv_0(s0+s0*e+e)
        Sal0 = F.sigmoid(self.conv_out_0(s0))
        Sal1 = F.sigmoid(self.conv_out_1(s1))
        Sal2 = F.sigmoid(self.conv_out_2(s2))
        Sal3 = F.sigmoid(self.conv_out_3(s3))
        Sal4 = F.sigmoid(self.conv_out_4(s4))
        E = F.sigmoid(self.conv_e(e))
        return Sal0,self.upsample1(Sal1),self.upsample2(Sal2),self.upsample3(Sal3),self.upsample4(Sal4),E#,G_d,G_t


