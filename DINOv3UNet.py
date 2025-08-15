import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

def load_dinov3_model(dinov3_project_path, weight_path):
    print("Loading DINOv3 model...")
    try:
        model = torch.hub.load(
            repo_or_dir=dinov3_project_path,
            model="dinov3_vitl16",
            source="local",
            pretrained=False,
            trust_repo=True
        )
        if os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=True)
            print("✓ Local weights loaded")
        else:
            print(f"Warning: Weight file not found at {weight_path}")
        
        return model
        
    except Exception as e:
        print(f"Error loading DINOv3 model: {e}")
        raise

class DiNOv3UNet(nn.Module):
    def __init__(self, dinov3_project_path=None, 
                 dinov3_weight_path=None):
        super(DiNOv3UNet, self).__init__()

        # ===== DINOv3 Encoder =====
        self.dino = load_dinov3_model(dinov3_project_path, dinov3_weight_path)
        for param in self.dino.parameters():
            param.requires_grad = False

        self.reduce = nn.Conv2d(1024, 128, 1)

        self.up1 = (Up(256, 128))
        self.up2 = (Up(256, 128))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(256, 128))

        self.head = nn.Conv2d(128, 1, 1)
        

    def forward(self, x):
        features = self.dino.get_intermediate_layers(x, n=[23], reshape=True, norm=True)
        dino_feat = features[0]

        dino_feat = self.reduce(dino_feat) #torch.Size([1, 128, 22, 22])

        feat_2x = F.interpolate(dino_feat, scale_factor=2, mode='bilinear') #torch.Size([1, 128, 44, 44])
        feat_4x = F.interpolate(dino_feat, scale_factor=4, mode='bilinear') #torch.Size([1, 128, 88, 88])
        feat_8x = F.interpolate(dino_feat, scale_factor=8, mode='bilinear') #torch.Size([1, 128, 176, 176])
        feat_16x = F.interpolate(dino_feat, scale_factor=16, mode='bilinear') #torch.Size([1, 128, 352, 352])
    
        x = self.up1(dino_feat, feat_2x) #torch.Size([1, 128, 44, 44])
        x = self.up2(x, feat_4x) #torch.Size([1, 128, 88, 88])
        x = self.up3(x, feat_8x) #torch.Size([1, 128, 176, 176])
        x = self.up4(x, feat_16x) #torch.Size([1, 128, 352, 352])

        out = self.head(x) #torch.Size([1, 1, 352, 352])
        return out
    
if __name__ == "__main__":
    model = DiNOv3UNet(
        dinov3_project_path=None,
        dinov3_weight_path=None
    )
    model.cuda()
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 352, 352).cuda()
        out = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")

