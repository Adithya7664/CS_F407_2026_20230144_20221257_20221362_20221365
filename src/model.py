import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.double_conv=DoubleConv(in_ch, out_ch)
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
    """
    U-Net with exactly 4 encoder layers and 4 decoder layers.
 
    Encoder channel progression : 3 → 64 → 128 → 256 → 512
    Bottleneck                  :          512 → 1024
    Decoder channel progression : 1024 → 512 → 256 → 128 → 64
    Output head                 : 64  → NUM_CLASSES (23)
 
    Spatial sizes at each stage (256×256 input):
        enc1 skip : 256×256,  down : 128×128
        enc2 skip : 128×128,  down :  64×64
        enc3 skip :  64×64,   down :  32×32
        enc4 skip :  32×32,   down :  16×16
        bottleneck:  16×16
        dec1 out  :  32×32
        dec2 out  :  64×64
        dec3 out  : 128×128
        dec4 out  : 256×256
    """
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        # 4 encoder stages
        self.enc1 = EncoderBlock(in_ch=3,   out_ch=64)    # layer 1
        self.enc2 = EncoderBlock(in_ch=64,  out_ch=128)   # layer 2
        self.enc3 = EncoderBlock(in_ch=128, out_ch=256)   # layer 3
        self.enc4 = EncoderBlock(in_ch=256, out_ch=512)   # layer 4
        #Bottleneck (bridges encoder ↔ decoder, not counted as a layer)
        self.bottleneck = DoubleConv(in_ch=512, out_ch=1024)
        #4 decoder layers
        # in_ch = bottleneck/previous decoder out_ch
        # skip_ch = matching encoder out_ch
        self.dec1 = DecoderBlock(in_ch=1024, skip_ch=512, out_ch=512)   # layer 1
        self.dec2 = DecoderBlock(in_ch=512,  skip_ch=256, out_ch=256)   # layer 2
        self.dec3 = DecoderBlock(in_ch=256,  skip_ch=128, out_ch=128)   # layer 3
        self.dec4 = DecoderBlock(in_ch=128,  skip_ch=64,  out_ch=64)    # layer 4
        #output head
        self.output_conv=nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        #weights initializsation
        self._init_weights()
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 256, 256)
 
        Returns
        -------
        logits : (B, NUM_CLASSES, 256, 256)
        """
        # Encoder
        s1, x = self.enc1(x)    # skip1: (B,  64, 256, 256)
        s2, x = self.enc2(x)    # skip2: (B, 128, 128, 128)
        s3, x = self.enc3(x)    # skip3: (B, 256,  64,  64)
        s4, x = self.enc4(x)    # skip4: (B, 512,  32,  32)
 
        # Bottleneck
        x = self.bottleneck(x)  # (B, 1024, 16, 16)
 
        # Decoder
        x = self.dec1(x, s4)    # (B, 512,  32,  32)
        x = self.dec2(x, s3)    # (B, 256,  64,  64)
        x = self.dec3(x, s2)    # (B, 128, 128, 128)
        x = self.dec4(x, s1)    # (B,  64, 256, 256)
 
        # Output
        return self.output_conv(x)  # (B, 23, 256, 256)
    
    def _init_weights(self):
        """Kaiming He initialisation for Conv layers; 1/0 for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
                
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

