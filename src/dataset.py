import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import config

# ── RGB → class-ID lookup table ───────────────────────────────────────────────
# Build once at import time from config.CLASS_COLORS so the per-sample
# conversion is a fast vectorised numpy operation.
 
def _build_rgb_to_id_map():
    """Return a dict {(R,G,B): class_id} from the palette in config."""
    return {rgb: idx for idx, rgb in enumerate(config.CLASS_COLORS)}

RGB_TO_ID = _build_rgb_to_id_map()
# Build once at module level
_LUT = np.zeros((256, 256, 256), dtype=np.int64)
for _idx, (_r, _g, _b) in enumerate(config.CLASS_COLORS):
    _LUT[_r, _g, _b] = _idx

def rgb_mask_to_class_ids(rgb_mask: np.ndarray) -> np.ndarray:
    return _LUT[rgb_mask[:,:,0], rgb_mask[:,:,1], rgb_mask[:,:,2]]

#image transforms

# Applied to the input image before feeding the ANN (256×256, normalised)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((config.ANN_H, config.ANN_W)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
 
# Applied to the raw image when we need the 800×600 anchor (geometry, display)
MAIN_RESIZE = transforms.Resize((config.MAIN_H, config.MAIN_W))

#dataset class

class GrazDataset(Dataset):
    """
    PyTorch Dataset for the TU Graz Semantic Drone Dataset.
 
    Directory layout expected
    <image_dir>/
        *.jpg   (original aerial images, any resolution)
    <mask_dir>/
        *.png   (RGB colour annotation masks, same stem as images)
 
    Each __getitem__ returns
    image_tensor : torch.Tensor  shape (3, 256, 256)  — ANN input
    mask_tensor  : torch.Tensor  shape (256, 256)      — class IDs (int64)
    main_image   : torch.Tensor  shape (3, 600, 800)   — 800x600 display copy
    filename     : str                                  — base filename
    """
    def __init__(self, image_dir: str=config.TRAIN_IMAGE_DIR, mask_dir: str=config.TRAIN_MASK_DIR, image_transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.image_transform = image_transform or IMAGE_TRANSFORM
        self.filenames=sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if len(self.filenames) == 0:
            raise FileNotFoundError(
                f"No images found in '{image_dir}'. "
                "Check TRAIN_IMAGE_DIR in config.py."
            )
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index:int):
        fname=self.filenames[index]
        stem=os.path.splitext(fname)[0]
        
        #load image
        img_path = os.path.join(self.image_dir, fname)
        image_pil = Image.open(img_path).convert("RGB")
        
        # 800×600 copy kept for geometry / visualisation (C×H×W float tensor)
        main_pil    = MAIN_RESIZE(image_pil)
        main_tensor = transforms.ToTensor()(main_pil)   # (3, 600, 800)
        
        # 256×256 normalised tensor for ANN
        image_tensor = self.image_transform(image_pil)  # (3, 256, 256)
        
        #load mask
        # Try common extensions: .png first, then .jpg
        mask_path = self._find_mask(stem)
        mask_pil  = Image.open(mask_path).convert("RGB")
 
        # Resize mask to 256×256 using NEAREST to preserve label values
        mask_pil_resized = mask_pil.resize(
            (config.ANN_W, config.ANN_H), resample=Image.NEAREST
        )
 
        # Convert RGB annotation → class-ID map
        mask_np     = np.array(mask_pil_resized, dtype=np.uint8)
        class_ids   = rgb_mask_to_class_ids(mask_np)               # (256,256)
        mask_tensor = torch.from_numpy(class_ids)                   # int64
 
        return image_tensor, mask_tensor, main_tensor, fname
    
    #helpers
    def _find_mask(self, stem: str):
        """Locate the annotation mask file for the given image stem."""
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
            candidate = os.path.join(self.mask_dir, stem + ext)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"No mask found for '{stem}' in '{self.mask_dir}'.\n"
            "Ensure the mask filename stem matches the image filename stem."
        )
        
    #Dataloader
def get_dataloaders(image_dir: str=config.TRAIN_IMAGE_DIR, mask_dir: str=config.TRAIN_MASK_DIR, batch_size: int=config.BATCH_SIZE, val_split: float=config.VAL_SPLIT, num_workers: int=config.NUM_WORKERS, seed: int=42):
    """
    Build train and validation DataLoaders from the Graz dataset.

    Parameters
    ----------
    image_dir   : path to original aerial images
    mask_dir    : path to RGB annotation masks
    batch_size  : samples per batch
    val_split   : fraction held out for validation (default 0.2)
    num_workers : parallel data-loading workers
    seed        : random seed for reproducible splits

    Returns
    -------
    train_loader, val_loader : torch.utils.data.DataLoader
    """
    full_dataset=GrazDataset(image_dir=image_dir, mask_dir=mask_dir)
    total     = len(full_dataset)
    val_count = int(total * val_split)
    trn_count = total - val_count

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        full_dataset, [trn_count, val_count], generator=generator
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    print(f"[dataset] Total samples : {total}")
    print(f"[dataset] Train samples : {trn_count}  |  Val samples : {val_count}")
    print(f"[dataset] Batch size    : {batch_size}")

    return train_loader, val_loader
    
if __name__ == "__main__":
    print("Running dataset sanity check…")
    train_loader, val_loader = get_dataloaders()
 
    img_t, mask_t, main_t, fname = next(iter(train_loader))
 
    print(f"  image_tensor shape : {img_t.shape}   dtype: {img_t.dtype}")
    print(f"  mask_tensor  shape : {mask_t.shape}  dtype: {mask_t.dtype}")
    print(f"  main_tensor  shape : {main_t.shape}  dtype: {main_t.dtype}")
    print(f"  filename           : {fname[0]}")
    print(f"  unique class IDs   : {mask_t.unique().tolist()}")