import os
import random
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF

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

# Final tensor normalisation applied to the image only (not the mask).
# Flips and geometric augmentations are done manually in __getitem__ so that
# the exact same transformation is applied to both image and mask.
_TO_TENSOR_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Colour-only jitter — safe to apply to image only, never to mask.
_COLOR_JITTER = transforms.ColorJitter(
    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
)

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
    def __init__(self, image_dir: str=config.TRAIN_IMAGE_DIR, mask_dir: str=config.TRAIN_MASK_DIR, augment: bool=True):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.augment   = augment
        self.filenames=sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if len(self.filenames) == 0:
            raise FileNotFoundError(
                f"No images found in '{image_dir}'. "
                "Check TRAIN_IMAGE_DIR in config.py."
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        fname = self.filenames[index]
        stem  = os.path.splitext(fname)[0]

        # ── Load image ───────────────────────────────────────────────────────
        img_path  = os.path.join(self.image_dir, fname)
        image_pil = Image.open(img_path).convert("RGB")

        # 800×600 copy for geometry / visualisation (no augmentation)
        main_pil    = MAIN_RESIZE(image_pil)
        main_tensor = transforms.ToTensor()(main_pil)   # (3, 600, 800)

        # ── Load mask ────────────────────────────────────────────────────────
        mask_path = self._find_mask(stem)
        mask_pil  = Image.open(mask_path).convert("RGB")

        # ── Resize both to 256×256 before augmentation ───────────────────────
        image_pil = image_pil.resize((config.ANN_W, config.ANN_H), resample=Image.BILINEAR)
        mask_pil  = mask_pil.resize( (config.ANN_W, config.ANN_H), resample=Image.NEAREST)

        # ── Geometric augmentations — SAME decision applied to image & mask ──
        # This is the critical part: both image and mask must be transformed
        # identically so the labels stay aligned with the pixels.
        if self.augment:
            if random.random() < 0.5:
                image_pil = TF.hflip(image_pil)
                mask_pil  = TF.hflip(mask_pil)

            if random.random() < 0.5:
                image_pil = TF.vflip(image_pil)
                mask_pil  = TF.vflip(mask_pil)

            # Small random rotation (±10°) — keeps most of the frame intact
            if random.random() < 0.3:
                angle     = random.uniform(-10, 10)
                image_pil = TF.rotate(image_pil, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask_pil  = TF.rotate(mask_pil,  angle, interpolation=TF.InterpolationMode.NEAREST)

            # Colour jitter on image only — mask is class IDs, not colour
            image_pil = _COLOR_JITTER(image_pil)

        # ── Convert to tensors ───────────────────────────────────────────────
        image_tensor = _TO_TENSOR_NORM(image_pil)   # (3, 256, 256) normalised

        mask_np     = np.array(mask_pil, dtype=np.uint8)
        class_ids   = rgb_mask_to_class_ids(mask_np)     # (256, 256) int64
        mask_tensor = torch.from_numpy(class_ids)

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
    # Train set gets augmentation; val set does not — we want clean val metrics.
    train_full = GrazDataset(image_dir=image_dir, mask_dir=mask_dir, augment=True)
    val_full   = GrazDataset(image_dir=image_dir, mask_dir=mask_dir, augment=False)

    total     = len(train_full)
    val_count = int(total * val_split)
    trn_count = total - val_count

    generator = torch.Generator().manual_seed(seed)
    indices   = torch.randperm(total, generator=generator).tolist()
    trn_indices = indices[:trn_count]
    val_indices = indices[trn_count:]

    from torch.utils.data import Subset
    train_ds = Subset(train_full, trn_indices)
    val_ds   = Subset(val_full,   val_indices)

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