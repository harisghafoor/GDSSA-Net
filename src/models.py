import torch
import torch.nn.functional as F
import torch.nn as nn

""" Multi Output Attention Unet"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNetppGradual(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttentionUNetppGradual, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        # Additional convolutions for deep supervision
        self.Conv_ds5 = nn.Conv2d(512, output_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_ds4 = nn.Conv2d(256, output_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_ds3 = nn.Conv2d(128, output_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_ds2 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
#         print(f"Input Image Shape: {x.shape}")

        e1 = self.Conv1(x)
        print(f"e1: {e1.shape}")

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)
        print(f"e2: {e2.shape}")

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)
        print(f"e3: {e3.shape}")

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)
        print(f"e4: {e4.shape}")

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)
        print(f"e5: {e5.shape}")

        d5 = self.Up5(e5)
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)
        d5 = self.UpConv5(d5)
        print(f"d5: {d5.shape}")

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        e3_upsampled = F.interpolate(e3, size=d4.size()[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)
        print(f"d4: {d4.shape}")

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        e2_upsampled = F.interpolate(e2, size=d3.size()[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)
        print(f"d3 (Second Last Decoder Output): {d3.shape}")

        d2_intermediate = self.Up2(d3)
        s1 = self.Att2(gate=d2_intermediate, skip_connection=e1)
        e1_upsampled = F.interpolate(e1, size=d2_intermediate.size()[2:], mode='bilinear', align_corners=True)
        d2_intermediate = torch.cat((s1, d2_intermediate), dim=1)
        d2_intermediate = self.UpConv2(d2_intermediate)
        print(f"d2_intermediate: {d2_intermediate.shape}")

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        e1_upsampled = F.interpolate(e1, size=d2.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)
        print(f"d2: {d2.shape}")

        out = self.final_conv(d2)
#         print(f"out: {out.shape}")

        # Deep supervision outputs
        ds_out5 = self.Conv_ds5(d5)
        ds_out4 = self.Conv_ds4(d4)
        ds_out3 = self.Conv_ds3(d3)
        ds_out2 = self.Conv_ds2(d2_intermediate)
#         print("x.size()[2:]",x.size()[2:])
        
#         ds_out3 = F.interpolate(ds_out3, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=True)
        print(f"ds_out5: {ds_out5.shape}")
        print(f"ds_out4: {ds_out4.shape}")
        print(f"ds_out3: {ds_out3.shape}")
        print(f"ds_out2: {ds_out2.shape}")
        # return out, [ds_out2,ds_out3,ds_out4,ds_out5]  
        return out, [ds_out3,ds_out4,ds_out5] 


if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256))
    f = AttentionUNetppGradual()
    main_output,ds_outputs = f(x)
#     print("Main Output Shape:", ds_outputs.shape)
#     print("Second Last Decoder Output Shape:", ds_outputs[0].shape)
#     print("Third Last Decoder Output Shape:", ds_outputs[1].shape)

