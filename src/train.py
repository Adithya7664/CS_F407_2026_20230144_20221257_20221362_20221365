import os
import time
import argparse
 
import numpy as np
import torch
import torch.nn as nn
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
    #Loss, optimizer, scheuler
    # ignore_index=-1 allows masking unknown pixels if needed
    criterion  = nn.CrossEntropyLoss(ignore_index=-1)
    from torch.optim import SGD
    optimiser = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
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
 
    ckpt  = torch.load(checkpoint, map_location=device)
    model = SegNet(num_classes=ckpt.get("num_classes", config.NUM_CLASSES))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    torch.backends.cudnn.benchmark = True
    print(f"[train] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"(val loss: {ckpt.get('best_val_loss', float('nan')):.4f})")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegNet on Graz Drone Dataset")
    parser.add_argument("--epochs",     type=int,   default=config.NUM_EPOCHS)
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
        