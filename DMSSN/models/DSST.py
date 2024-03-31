from re import T
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from timm.models.layers import DropPath


class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DEPTHWISECONV, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class SSAA(nn.Module):
    def __init__(self, inp_channel, num_heads, stride, attn_drop, proj_drop):
        super().__init__()
        self.num_heads = num_heads
        head_dim = inp_channel // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Conv2d(inp_channel, 2*inp_channel, 1, 1, 0)
        self.stride = nn.Conv2d(inp_channel, inp_channel, stride, stride)
        self.spak = DEPTHWISECONV(inp_channel, inp_channel)
        self.spav = DEPTHWISECONV(inp_channel, inp_channel)
        self.spek = nn.Conv2d(in_channels=inp_channel, out_channels=inp_channel, kernel_size=1, stride=1, padding=0, groups=1)
        self.spev = nn.Conv2d(in_channels=inp_channel, out_channels=inp_channel, kernel_size=1, stride=1, padding=0, groups=1)
        self.proj = nn.Conv2d(2*inp_channel, inp_channel, 1, 1, 0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B,2,self.num_heads,C//self.num_heads,H*W).permute(1, 0, 2, 4, 3).contiguous()#b,c,h,w-b,2c,h,w-b,2,n,c/n,hw-2,b,n,hw,c/n
        qa, qe = q[0], q[1]#2,b,n,hw,c/n-b,n,hw,c/n
        x = self.stride(x)#b,c,h,w-b,c,h',w'
        spak = self.spak(x).reshape(B,self.num_heads,C//self.num_heads,-1).permute(0, 1, 3, 2).contiguous()#b,c,h',w'-b,c,h',w'-b,n,c/n,h'w'-b,n,h'w',c/n
        spav = self.spav(x).reshape(B,self.num_heads,C//self.num_heads,-1).permute(0, 1, 3, 2).contiguous()
        spek = self.spek(x).reshape(B,self.num_heads,C//self.num_heads,-1).permute(0, 1, 3, 2).contiguous()
        spev = self.spev(x).reshape(B,self.num_heads,C//self.num_heads,-1).permute(0, 1, 3, 2).contiguous()
        attna = (qa @ spak.transpose(-2, -1).contiguous()) * self.scale#b,n,hw,h'w'
        attna = attna.softmax(dim=-1)
        attna = self.attn_drop(attna)
        outa = (attna @ spav).transpose(2, 3).contiguous().reshape(B, C, H, W)#b,n,hw,c/n-b,n,c/n,hw-b,c,h,w
        attne = (qe @ spek.transpose(-2, -1).contiguous()) * self.scale
        attne = attne.softmax(dim=-1)
        attne = self.attn_drop(attne)
        oute = (attne @ spev).transpose(2, 3).contiguous().reshape(B, C, H, W)
        out = torch.cat((outa,oute),1)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, stride, mlp_ratio=4, drop=0.5, attn_drop=0.5, drop_path=0.5):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=dim)
        self.attn = SSAA(dim, num_heads=num_heads, stride=stride, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class CGG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, pad, group):
        super(CGG, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel,stride=stride,padding=pad,groups=group)
        num = int(out_channels // 32)
        self.groupnorm = nn.GroupNorm(num_groups=num, num_channels=out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.groupnorm(x)
        x = self.gelu(x)
        return x


class SimpleHead(nn.Module):
    def __init__(self, channels, out_cla=1):
        super().__init__()
        self.stages = len(channels)
        self.upconvs = nn.ModuleList()
        for i in range(self.stages):
            self.upconvs.append(CGG(channels[i],channels[0],kernel=1,stride=1,pad=0,group=1))
        self.conv_seg = nn.Conv2d(channels[0], out_cla, kernel_size=1,stride=1,padding=0)

    def forward(self, inputs, H, W):
        for i in range(self.stages):
            if i == 0:
                outs = self.upconvs[i](inputs[i])
                outs = resize(outs, size=(H,W), mode='bilinear', align_corners=False)
            else:
                out = self.upconvs[i](inputs[i])
                #out = resize(out, size=outs.size()[2:], mode='bilinear', align_corners=False)
                out = resize(out, size=(H,W), mode='bilinear', align_corners=False)
                outs += out
        x = self.conv_seg(outs)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=32, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size // 2, patch_size // 2))
        num = int(embed_dim // 32)
        self.norm = nn.GroupNorm(num_groups=num, num_channels=embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)

        return x


class Backbone(nn.Module):
    def __init__(self, in_channel=32, out_cla=1, depths=[2, 2, 2, 2], channels=[32, 64, 96, 128], num_groups=[1, 2, 4, 8], 
                strides=[8, 4, 2, 1], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], drop_rate=0.3, attn_drop_rate=0.3, drop_path_rate=0.3):
        super(Backbone,self).__init__()
        self.num_stages = len(depths)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(self.num_stages):
            patch_embed = PatchEmbed(patch_size=7 if i == 0 else 3,
                                    stride=4 if i == 0 else 2,
                                    in_chans=in_channel if i == 0 else channels[i - 1],
                                    embed_dim=channels[i])
            block = nn.ModuleList([Block(
                dim=channels[i], num_heads=num_heads[i], stride=strides[i], mlp_ratio=mlp_ratios[i],
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = nn.GroupNorm(num_groups=num_groups[i], num_channels=channels[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.decoder = SimpleHead(channels,out_cla)

    def forward(self, x):
        _,_,H,W = x.shape
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = norm(x)
            outs.append(x)

        result = self.decoder(outs,H,W)

        return F.sigmoid(result)


class CIG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, pad, group):
        super(CIG, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel,stride=stride,padding=pad,groups=group)
        self.ins = nn.InstanceNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.ins(x)
        x = self.gelu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channel=200, out_channel=32):
        super().__init__()
        self.conv1 = CIG(in_channel,64,1,1,0,1)
        self.conv2 = CIG(64,out_channel,1,1,0,1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x1,x2


class Decoder(nn.Module):
    def __init__(self, in_channel=32, out_channel=200):
        super().__init__()
        self.conv1 = CIG(in_channel,64,1,1,0,1)
        self.conv2 = CIG(64,out_channel,1,1,0,1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x1,x2


class Auto_stu(nn.Module):
    def __init__(self,in_channel=200,mid_channel=32):
        super(Auto_stu,self).__init__()
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        
        self.encoder = Encoder(in_channel=in_channel,out_channel=mid_channel)
        self.decoder = Decoder(in_channel=mid_channel,out_channel=in_channel)

    def forward(self, x):
        enc64,enc32 = self.encoder(x)
        dec64,dec200 = self.decoder(enc32)

        return enc64,enc32,dec64,dec200


class ACEN(nn.Module):
    def __init__(self,in_channel=200,mid_channel=32):
        super(ACEN,self).__init__()
        self.encoder = Auto_stu(in_channel=in_channel,mid_channel=mid_channel)
        self.decoder = Backbone(in_channel=mid_channel)

    def forward(self, x):
        enc64,enc32,dec64,dec200 = self.encoder(x)
        out = self.decoder(enc32)
        return enc64, enc32, dec64, dec200, out