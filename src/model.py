import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import config

#building block
class DoubleConv(nn.Module):
    """
    Two consecutive  Conv2d → BatchNorm → ReLU  blocks.
    Used in both encoder and decoder layers.
    """
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__()
        self.block=nn.Sequential(nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
            return self.block(x)
        
class EncoderBlock(nn.Module):
    """
    One encoder stage:
        DoubleConv  →  MaxPool2d(2)
    Returns both the skip-connection feature map (before pooling)
    and the pooled output (passed to the next stage).
    """
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__()
        self.conv=DoubleConv(in_ch, out_ch)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x:torch.Tensor):
        skip=self.conv(x)
        down=self.pool(skip)
        return skip, down

class DecoderBlock(nn.Module):
    """
    One decoder stage:
        ConvTranspose2d (upsample x2)  →  concat skip  →  DoubleConv
    """
    def __init__(self, in_ch:int, skip_ch:int, out_ch:int):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv=DoubleConv(in_ch//2+skip_ch, out_ch)
        
    def forward(self, x:torch.Tensor, skip:torch.Tensor) -> torch.Tensor:
        x=self.up(x)
        # Handle potential off-by-one size mismatch after transpose conv
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
 
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
    
# main model

class SegNet(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        
        # Pretrained ResNet18 encoder — loads ImageNet weights automatically
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Encoder layers from ResNet18
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64ch
        self.pool  = resnet.maxpool
        self.enc2  = resnet.layer1   # 64ch  — encoder layer 2
        self.enc3  = resnet.layer2   # 128ch — encoder layer 3
        self.enc4  = resnet.layer3   # 256ch — encoder layer 4

        # Decoder layers
        # Channel math:
        #   dec1: enc4(256) → 128
        #   dec2: dec1(128) + s3(128) = 256 → 64
        #   dec3: dec2(64)  + s2(64)  = 128 → 64
        #   dec4: dec3(64)  + s1(64)  = 128 → 32   ← s1 skip now used
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )

        # Output head
        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        self._init_decoder_weights()

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)        # (B, 64,  128, 128)
        x  = self.pool(s1)       # (B, 64,   64,  64)
        s2 = self.enc2(x)        # (B, 64,   64,  64)
        s3 = self.enc3(s2)       # (B, 128,  32,  32)
        x  = self.enc4(s3)       # (B, 256,  16,  16)

        # Decoder with skip connections
        x = self.dec1(x)                              # (B, 128,  32,  32)
        x = self.dec2(torch.cat([x, s3], dim=1))      # (B,  64,  64,  64)
        x = self.dec3(torch.cat([x, s2], dim=1))      # (B,  64, 128, 128)
        x = self.dec4(torch.cat([x, s1], dim=1))      # (B,  32, 256, 256) — s1 skip

        x = F.interpolate(x, size=(config.ANN_H, config.ANN_W),
                         mode='bilinear', align_corners=False)

        return self.output_conv(x)
    
    def _init_decoder_weights(self):
        """Kaiming He init for decoder layers only.
        Scoped to dec1-dec4 and output_conv so the pretrained ResNet18
        encoder weights are never overwritten."""
        decoder_parts = [self.dec1, self.dec2, self.dec3, self.dec4, self.output_conv]
        for module in decoder_parts:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = SegNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
 
    dummy = torch.zeros(2, 3, config.ANN_H, config.ANN_W).to(config.DEVICE)
    out   = model(dummy)
 
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}   (expected: [2, {config.NUM_CLASSES}, 256, 256])")
    print(f"Trainable parameters : {count_parameters(model):,}")
    assert out.shape == (2, config.NUM_CLASSES, config.ANN_H, config.ANN_W), \
        "Output shape mismatch!"

