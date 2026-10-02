import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import SegNet, count_parameters
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


class FocalLoss(nn.Module):
    """
    Focal Loss for dense semantic segmentation with class imbalance.

    Standard CrossEntropyLoss gives equal weight to every pixel.
    Focal Loss multiplies each pixel's loss by (1 - p_t)^gamma, where
    p_t is the model's confidence in the correct class:
      - If the model is already confident (p_t ≈ 1, easy pixel) → factor ≈ 0 → loss suppressed
      - If the model is wrong / uncertain (p_t ≈ 0, hard pixel) → factor ≈ 1 → loss preserved

    Effect: training effort automatically shifts toward the rare, hard classes
    (person, ar-marker, obstacle) that standard CE ignores once dominant classes converge.

    gamma=2 is the standard value from the original Focal Loss paper (Lin et al. 2017).
    """
    def __init__(self, gamma: float = 2.0,
                 weight: torch.Tensor = None,
                 ignore_index: int = 23):
        super().__init__()
        self.gamma        = gamma
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-pixel cross-entropy (unreduced) — shape (B, H, W)
        ce = F.cross_entropy(inputs, targets,
                             weight=self.weight,
                             ignore_index=self.ignore_index,
                             reduction="none")

        # p_t = probability assigned to the correct class
        pt = torch.exp(-ce)

        # Focal weight: (1 - p_t)^gamma
        focal = ((1.0 - pt) ** self.gamma) * ce

        # Average only over non-ignored pixels
        mask = targets != self.ignore_index
        return focal[mask].mean() if mask.any() else focal.mean()


class DiceLoss(nn.Module):
    """
    Soft Dice Loss averaged over all non-ignored classes present in the batch.

    Dice directly optimises overlap (numerator = 2·|A∩B|, denominator = |A|+|B|),
    so it is a proxy for IoU.  Pairing it with Focal Loss (which is a better
    cross-entropy proxy) gives a hybrid that improves both pixel accuracy and mIoU.
    """
    def __init__(self, ignore_index: int = 23, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.shape[1]
        probs       = F.softmax(inputs, dim=1)          # (B, C, H, W)

        valid = targets != self.ignore_index             # (B, H, W)
        dice_scores = []
        for cls in range(num_classes):
            if cls == self.ignore_index:
                continue
            target_cls = ((targets == cls) & valid).float()   # (B, H, W)
            pred_cls   = probs[:, cls] * valid.float()        # (B, H, W)
            if target_cls.sum() == 0 and pred_cls.sum() == 0:
                continue
            intersection = (pred_cls * target_cls).sum()
            dice = (2.0 * intersection + self.smooth) / (pred_cls.sum() + target_cls.sum() + self.smooth)
            dice_scores.append(1.0 - dice)

        return torch.stack(dice_scores).mean() if dice_scores else inputs.sum() * 0.0


class HybridLoss(nn.Module):
    """Focal Loss + Dice Loss weighted sum."""
    def __init__(self, focal: nn.Module, dice: nn.Module, alpha: float = 0.5):
        super().__init__()
        self.focal = focal
        self.dice  = dice
        self.alpha = alpha   # weight on focal; (1-alpha) on dice

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.focal(inputs, targets) + (1.0 - self.alpha) * self.dice(inputs, targets)


#Metrics
def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Fraction of correctly classified pixels.
 
    Parameters
    ----------
    preds   : (B, C, H, W) raw logits
    targets : (B, H, W)   integer class IDs
    """
    pred_classes=preds.argmax(dim=1)  # (B, H, W)
    correct = (pred_classes == targets).sum().item()
    total=targets.numel()
    return correct / total

def mean_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int=config.NUM_CLASSES) -> float:
    """
    Mean Intersection-over-Union across all classes present in the batch.
    Classes absent from both pred and target are excluded from the mean.
    """
    pred_classes = preds.argmax(dim=1).cpu().numpy().flatten()
    targets_np   = targets.cpu().numpy().flatten()
 
    ious = []
    for cls in range(num_classes):
        pred_mask   = pred_classes == cls
        target_mask = targets_np   == cls
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union        = np.logical_or(pred_mask,  target_mask).sum()
        if union > 0:
            ious.append(intersection / union)
 
    return float(np.mean(ious)) if ious else 0.0

def run_epoch(model:nn.Module, loader, criterion:nn.Module, optimizer:torch.optim.Optimizer, device:torch.device, training:bool) -> dict:
    """
    Run one full pass over `loader`.
 
    Returns
    -------
    dict with keys: loss, pixel_acc, mean_iou
    """
    model.train(training)
    desc = "  Train" if training else "  Val  "
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    n_batches = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in tqdm(loader, desc=desc, leave=False, ncols=80):
            # dataset returns: image_tensor, mask_tensor, main_tensor, filename
            images, masks, _, _ = batch
            images = images.to(device, non_blocking=True)  # (B, 3, 256, 256)
            masks  = masks.to(device,  non_blocking=True)  # (B, 256, 256) int64
 
            logits = model(images)                          # (B, 23, 256, 256)
            loss   = criterion(logits, masks)
 
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
 
            total_loss += loss.item()
            total_acc  += pixel_accuracy(logits.detach(), masks)
            total_iou  += mean_iou(logits.detach(), masks)
            n_batches  += 1
 
    return {
        "loss":       total_loss / max(n_batches, 1),
        "pixel_acc":  total_acc  / max(n_batches, 1),
        "mean_iou":   total_iou  / max(n_batches, 1),
    }
    
#Main training function
def train(num_epochs:int = config.NUM_EPOCHS, batch_size:int= config.BATCH_SIZE, lr:float = config.LEARNING_RATE, checkpoint:str= config.CHECKPOINT_PATH, resume:bool  = False, tb_log_dir:str= "runs/segnet"):
    """
    Full training + validation loop.
 
    Parameters
    ----------
    num_epochs  : total number of training epochs
    batch_size  : samples per batch
    lr          : initial learning rate for Adam
    checkpoint  : path to save / load best_model.pth
    resume      : if True and checkpoint exists, resume from it
    tb_log_dir  : TensorBoard log directory (ignored if TB not installed)
    """
    device = config.DEVICE
    print(f"\n{'='*60}")
    print(f"  CS F407 — SegNet Training")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {num_epochs}")
    print(f"  Batch size  : {batch_size}")
    print(f"  LR          : {lr}")
    print(f"  Checkpoint  : {checkpoint}")
    print(f"{'='*60}\n")
    #data
    train_loader, val_loader = get_dataloaders(
        image_dir=config.TRAIN_IMAGE_DIR, 
        mask_dir=config.TRAIN_MASK_DIR, 
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
    )
    #Model
    model = SegNet(num_classes=config.NUM_CLASSES).to(device)
    print(f"  Trainable params : {count_parameters(model):,}\n")
    print(f"Model device: {next(model.parameters()).device}")
    torch.backends.cudnn.benchmark = True
    # ── Loss: Focal Loss with class weights ──────────────────────────────────
    # Focal Loss focuses training on hard/rare pixels automatically via the
    # (1-p_t)^gamma modulation. Class weights provide an additional manual
    # boost to the most important rare classes on top of that.
    class_weights = torch.ones(config.NUM_CLASSES, device=device)
    class_weights[0]  = 0.3   # unlabeled    — mostly annotation borders
    class_weights[1]  = 2.0   # paved-area   — SAFE
    class_weights[2]  = 2.0   # dirt         — SAFE
    class_weights[3]  = 2.0   # grass        — SAFE
    class_weights[4]  = 2.0   # gravel       — SAFE
    class_weights[15] = 3.0   # person       — must not land on people
    class_weights[17] = 2.5   # car          — obstacle
    class_weights[21] = 3.0   # ar-marker    — very rare, critical
    class_weights[22] = 2.5   # obstacle     — must avoid
    focal     = FocalLoss(gamma=2.0, weight=class_weights, ignore_index=23)
    dice      = DiceLoss(ignore_index=23)
    criterion = HybridLoss(focal, dice, alpha=0.5)
    optimiser = Adam([
    {'params': model.enc1.parameters(), 'lr': lr * 0.1},
    {'params': model.enc2.parameters(), 'lr': lr * 0.1},
    {'params': model.enc3.parameters(), 'lr': lr * 0.1},
    {'params': model.enc4.parameters(), 'lr': lr * 0.1},
    {'params': model.dec1.parameters(), 'lr': lr * 0.5},
    {'params': model.dec2.parameters(), 'lr': lr * 0.5},
    {'params': model.dec3.parameters(), 'lr': lr * 0.5},
    {'params': model.dec4.parameters(), 'lr': lr * 0.5},
    {'params': model.output_conv.parameters(), 'lr': lr},
], weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimiser, T_max=num_epochs, eta_min=1e-6)
    #TensorBoard
    writer = None
    if _TB_AVAILABLE:
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"  TensorBoard logs : {tb_log_dir}")
    #Optional resume
    start_epoch   = 1
    best_val_loss = float("inf")
 
    if resume and os.path.isfile(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optimiser_state"])
        start_epoch   = ckpt.get("epoch", 1) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch - 1}  "
              f"(best val loss: {best_val_loss:.4f})\n")
    #Training loop
    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  [],
               "train_iou":  [], "val_iou":  []}
    for epoch in range(start_epoch, num_epochs + 1):
        t0=time.time()
        trn=run_epoch(model, train_loader, criterion, optimiser, device, training=True)
        val = run_epoch(model, val_loader, criterion,
                        None, device, training=False)
 
        scheduler.step()
        elapsed = time.time() - t0
        history["train_loss"].append(trn["loss"])
        history["val_loss"].append(val["loss"])
        history["train_acc"].append(trn["pixel_acc"])
        history["val_acc"].append(val["pixel_acc"])
        history["train_iou"].append(trn["mean_iou"])
        history["val_iou"].append(val["mean_iou"])
 
        if writer:
            writer.add_scalars("Loss",       {"train": trn["loss"],      "val": val["loss"]},      epoch)
            writer.add_scalars("PixelAcc",   {"train": trn["pixel_acc"], "val": val["pixel_acc"]}, epoch)
            writer.add_scalars("mIoU",       {"train": trn["mean_iou"],  "val": val["mean_iou"]},  epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
 
        print(
            f"Epoch [{epoch:03d}/{num_epochs}]  "
            f"TrainLoss: {trn['loss']:.4f}  "
            f"ValLoss: {val['loss']:.4f}  "
            f"ValAcc: {val['pixel_acc']*100:.2f}%  "
            f"ValIoU: {val['mean_iou']*100:.2f}%  "
            f"({elapsed:.1f}s)"
        )
        #save best checkpoint
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            _save_checkpoint(
                path=checkpoint,
                model=model,
                optimiser=optimiser,
                epoch=epoch,
                best_val_loss=best_val_loss,
                val_metrics=val,
            )
            print(f"  ✓ Saved best checkpoint  (val loss: {best_val_loss:.4f})")
            
    if writer:
        writer.close()
 
    print(f"\nTraining complete.")
    print(f"Best validation loss : {best_val_loss:.4f}")
    print(f"Checkpoint saved to  : {checkpoint}")
 
    return history

def _save_checkpoint(path, model, optimiser, epoch, best_val_loss, val_metrics):
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimiser_state": optimiser.state_dict(),
        "best_val_loss":   best_val_loss,
        "val_metrics":     val_metrics,
        "num_classes":     config.NUM_CLASSES,
        "ann_h":           config.ANN_H,
        "ann_w":           config.ANN_W,
    }, path)
 
 
def load_model(checkpoint: str = config.CHECKPOINT_PATH,
               device: torch.device = config.DEVICE) -> SegNet:
    """
    Load a saved SegNet from a checkpoint file.
    Used by main.py at inference time.
 
    Returns
    -------
    model : SegNet  (eval mode, on `device`)
    """
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint}' not found.\n"
            "Run train.py first to generate best_model.pth."
        )
 
    ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SegNet(num_classes=ckpt.get("num_classes", config.NUM_CLASSES))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    torch.backends.cudnn.benchmark = True
    print(f"[train] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"(val loss: {ckpt.get('best_val_loss', float('nan')):.4f})")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegNet on Graz Drone Dataset")
    parser.add_argument("--epochs",     type=int,   default=config.NUM_EPOCHS)  # now 100
    parser.add_argument("--batch-size", type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=config.LEARNING_RATE)
    parser.add_argument("--checkpoint", type=str,   default=config.CHECKPOINT_PATH)
    parser.add_argument("--resume",     action="store_true",
                        help="Resume training from existing checkpoint")
    parser.add_argument("--tb-log-dir", type=str,   default="runs/segnet")
    args = parser.parse_args()
 
    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint=args.checkpoint,
        resume=args.resume,
        tb_log_dir=args.tb_log_dir,
    )
        